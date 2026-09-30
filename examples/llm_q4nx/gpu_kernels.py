# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton kernels for the iGPU half of a hybrid run.

The GPU counterpart to `kernels.py`. Both exist because torch is this
repository's CPU reference, as `README.md` says, rather than how an example
computes on a device; `examples/gpt2` and `examples/qwen2_5` have had
`*_kernel_gpu` since they grew a GPU path.

The decode kernels take one row of activations, and that shapes them. Every
projection is a GEMV with nothing for `tl.dot` to contract over, and a forward
issues one per projection per layer, so launch count matters more than
arithmetic: `_gemv_kernel` reduces over K by hand, and the elementwise kernels
are written to fuse into as few launches as the forward allows.

The prefill kernels at the bottom take N > 1 and look nothing like them -- a
tiled `tl.dot`, and an attention carrying an online softmax so the score matrix
never exists at once.

Nothing that varies per call is a constexpr. A tile derived from the context
length, in particular, recompiles the kernel as the context grows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: The GeGLU constants, spelled as `kernels._geglu_kernel` spells them: `C2` is
#: *twice* sqrt(2/pi) because `tanh(z) = 2*sigmoid(2z) - 1` turns
#: gelu_pytorch_tanh into the exact `x * sigmoid(2z)`. Not an approximation, and
#: not a second definition -- mlir-air's decode runs gelu_tanh in `glu.cc`, and
#: all three have to agree about the model.
_GELU_2C = 1.5957691216057308
_GELU_K = 0.044715


def _pow2(n):
    return 1 << (max(int(n), 1) - 1).bit_length()


class gpu_driver:
    """Make Triton's AMD GPU backend active for the duration.

    Required, not hygiene. `kernels._npu_driver` restores the previously active
    driver only when there *was* one, and on the first NPU launch of a process
    there is not -- so after any prefill the NPU driver stays active, and a
    kernel launched here would be handed to `amd_triton_npu` and fail in aircc
    (`failed to legalize operation 'memref.copy'`) rather than compiling for
    gfx1151.

    Enter it once around a whole generation, not per step. Re-entering is free;
    it no-ops when the GPU backend is already active. Switching is not:
    `set_active` drops Triton's compiled-kernel cache, so a scope entered per
    token recompiles everything that token touches.
    """

    _AMD = "triton.backends.amd.driver"

    def __enter__(self):
        from triton.backends import backends

        self._prev = getattr(triton.runtime.driver, "_active", None)
        self._switched = self._prev is None or type(self._prev).__module__ != self._AMD
        if self._switched:
            triton.runtime.driver.set_active(backends["amd"].driver())

    def __exit__(self, *exc):
        if self._switched and self._prev is not None:
            triton.runtime.driver.set_active(self._prev)


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------
@triton.jit
def _gemv_kernel(
    A,
    B,
    C,
    K,
    N,
    KS,
    stride_bk,
    stride_bn,
    U,
    C2,
    KC,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_PARTS: tl.constexpr,
    GLU: tl.constexpr,
):
    """`[1, K] @ [K, N] -> [1, N]`, f32 accumulation.

    A reduction over K rather than `tl.dot`: with one activation row there is
    no M to tile, and a dot would pad M to the hardware's minimum and throw the
    work away.

    Both of B's strides are taken rather than assuming the N axis is packed.
    The LM head is reached as `lm_head.T`, a transposed view whose *K* axis is
    the contiguous one; assuming otherwise reads the wrong elements and still
    produces plausible logits -- the decode ran and emitted fluent nonsense.

    Split-K, for the narrow projections that otherwise run as a handful of
    programs: grid axis 1 takes `KS` of the K range each and stores its
    partial row at `C + s * N`, unreduced. The consumer reduces them as it
    loads: `A_PARTS` rows of `A` are summed on the way in -- and, with `GLU`,
    the sum `a` is replaced by `gelu_tanh(a) * U`. Both are elementwise on the
    activation, so neither adds a serial step to this kernel.
    """
    pid_n = tl.program_id(0)
    s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    k_end = tl.minimum(K, s * KS + KS)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(s * KS, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_end
        a = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for p in tl.static_range(A_PARTS):
            a += tl.load(A + p * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        if GLU:
            u = tl.load(U + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            a = a * tl.sigmoid(C2 * (a + KC * a * a * a)) * u
        # Rounded to bf16 here rather than by the caller: see `gemv`.
        a = a.to(tl.bfloat16).to(tl.float32)
        b = tl.load(
            B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(a[:, None] * b, axis=0)
    tl.store(C + s * N + offs_n, acc, mask=mask_n)


def _gemv_blocks(K, N):
    """Tile for one GEMV. Measured on gfx1151, against torch as the baseline.

    A wide N wants a wide tile so the few programs each carry enough work; a
    deep K wants a deep one so the reduction loop is short. Picked from a sweep
    over (64..512) x (32..256) on the four shapes this model actually issues --
    `qkv`, `gate_up`, `down` and the head -- rather than autotuned, because
    `triton.autotune` runs every candidate on the first call of each shape and
    there are eight of them per token.

    The defaults this replaced (128, 64) were faster than torch on two of the
    four shapes and slower on the others; these beat it on all four.
    """
    return (512 if N >= 16384 else 64), (256 if K >= 8192 else 64)


def gemv(x, w, block_n=None, block_k=None, glu_up=None, split_k=1):
    """x: [1, K] -> [1, N] against w: [K, N]. f32 out, as `_mm` returns.

    `w` may be a transposed view -- the LM head is -- so its strides are read
    rather than assumed, and it is deliberately NOT forced contiguous: that
    would copy a 262144x1536 weight on every token.

    `split_k` > 1 returns `[split_k, N]` partial sums, unreduced; pass them
    straight on as the next GEMV's `x`, which sums the rows as it loads them.
    `glu_up` makes the activation `gelu_tanh(x) * glu_up`, applied after that
    sum -- the GeGLU fused into the consumer rather than the producer, which
    is what lets the producer split its K.
    """
    K, N = w.shape
    if block_n is None or block_k is None:
        bn, bk = _gemv_blocks(K, N)
        block_n = block_n or bn
        block_k = block_k or bk
    # bf16 in, f32 accumulation -- `LlamaPrefill._matmul` rounds the activation
    # before the product and the NPU GEMM does the same, so feeding f32 here
    # would make the GPU decode compute from different inputs than the path it
    # is meant to reproduce, and drift a token at a time. The rounding happens
    # in the kernel's load: a `.to(bfloat16)` here is a launch of its own, and
    # at one per GEMV it was ~250 launches a token.
    parts = x.shape[0] if x.dim() == 2 else 1
    x = x.contiguous()
    if x.numel() != parts * K:
        raise ValueError(f"activation {tuple(x.shape)} does not match K={K}")
    ks = triton.cdiv(triton.cdiv(K, split_k), block_k) * block_k
    out = torch.empty((split_k, N), dtype=torch.float32, device=x.device)
    _gemv_kernel[(triton.cdiv(N, block_n), split_k)](
        x,
        w,
        out,
        K,
        N,
        ks,
        w.stride(0),
        w.stride(1),
        glu_up.reshape(-1).contiguous() if glu_up is not None else out,
        _GELU_2C,
        _GELU_K,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        A_PARTS=parts,
        GLU=glu_up is not None,
    )
    return out


# ---------------------------------------------------------------------------
# Projections, W4A16: Q4NX Codec B read in place
# ---------------------------------------------------------------------------
#: Codec B chunk geometry, as mlir-air's `proj_qmm_pack` defines it. A chunk is
#: 32 output rows by 256 K columns, 5120 bytes: 256 bf16 scales, 256 bf16 mins
#: -- one per (32-column group, row), group-major -- then 4096 bytes of nibbles.
Q4NX_ROWS = 32
Q4NX_COLS = 256
Q4NX_CHUNK_BYTES = 5120

#: Codec K: the same chunk -- same geometry, same 4096 bytes of nibbles, byte
#: for byte -- with the per-(group, row) bf16 scale and min replaced by 6-bit
#: codes, 4608 bytes a chunk: 4.5 bits a weight, as llama.cpp's Q4_K, against
#: Codec B's 5.0. Both codes are LOGARITHMIC over a range stored per row:
#:
#:     scale = 2^(s_lo + k_s * (s_hi - s_lo) / 63)
#:     min   = -scale * 2^(r_lo + k_r * (r_hi - r_lo) / 63)
#:
#: the min as a ratio to its own group's scale. Each endpoint is one byte, a
#: log2 in 1/16-octave steps (scales from 2^-16, ratios from 2^-8), rounded
#: outward so the range brackets the row.
#:
#:     [0, 128)     per row, 4 bytes: s_lo, s_hi, r_lo, r_hi
#:     [128, 384)   low 8 bits of each (group, row)'s 12-bit code k_s | k_r << 6
#:     [384, 512)   high 4 bits of the same, two per byte
#:     [512, 4608)  the Codec B nibbles, unchanged
#:
#: (group, row) is at `v = 32 * group + row`, Codec B's own scale order, so
#: both codecs index the same way. Laid out for the kernel: a thread needs one
#: 32-bit load for its row's endpoints and two byte loads for its code, where
#: a plane per field took eight. Lossy; see `Q4NXWeight.to_codec_k` for how
#: much and why this encoding.
Q4K_CHUNK_BYTES = 4608
#: Where the endpoint bytes' log2 scales start: scales from 2^-16, ratios 2^-8.
_Q4K_S_OFF = 16.0
_Q4K_R_OFF = 8.0


class Q4NXWeight:
    """A Codec B projection `[N, K]` (y = W x), kept packed on the device.

    `data` is the bundle's chunks verbatim, as uint8, in the bundle's
    block-major order: chunk `i * (K // 256) + j` covers output rows
    `[32i, 32i + 32)` and K columns `[256j, 256j + 256)`. That order is also
    what makes fusing projections free -- stacking two matrices along N is
    concatenating their chunk arrays, so Q|K|V and gate|up need no repacking.
    """

    def __init__(self, data, N, K, codec="b"):
        if N % Q4NX_ROWS or K % Q4NX_COLS:
            raise ValueError(f"Codec B needs N % 32 == K % 256 == 0, got {N}x{K}")
        chunk = {"b": Q4NX_CHUNK_BYTES, "k": Q4K_CHUNK_BYTES}[codec]
        nb = (N // Q4NX_ROWS) * (K // Q4NX_COLS)
        if data.numel() != nb * chunk or data.dtype != torch.uint8:
            raise ValueError(
                f"{N}x{K} is {nb} chunks = {nb * chunk} bytes of uint8, "
                f"got {data.numel()} of {data.dtype}"
            )
        self.data, self.N, self.K, self.codec = data, N, K, codec
        self.chunk_bytes = chunk

    @property
    def shape(self):
        """`[K, N]`, the orientation `gemv` takes its bf16 weights in."""
        return (self.K, self.N)

    def to(self, device):
        return Q4NXWeight(self.data.to(device), self.N, self.K, self.codec)

    def _scales_mins(self):
        """Per-chunk scale and min as float32 `[nb, 8 groups, 32 rows]`."""
        c = self.data.view(-1, self.chunk_bytes)
        if self.codec == "b":
            sc = c[:, :512].contiguous().view(torch.bfloat16).float()
            mn = c[:, 512:1024].contiguous().view(torch.bfloat16).float()
            return sc.view(-1, 8, 32), mn.view(-1, 8, 32)
        e = c[:, :128].float().view(-1, 1, 32, 4) / 16  # [nb, 1, row, endpoint]
        s_lo, s_hi = e[..., 0] - _Q4K_S_OFF, e[..., 1] - _Q4K_S_OFF
        r_lo, r_hi = e[..., 2] - _Q4K_R_OFF, e[..., 3] - _Q4K_R_OFF
        hi = c[:, 384:512]
        hi = torch.stack([hi & 0xF, hi >> 4], -1).view(-1, 256)
        code = c[:, 128:384].int() | (hi.int() << 8)
        ks = (code & 63).float().view(-1, 8, 32)
        kr = (code >> 6).float().view(-1, 8, 32)
        sc = torch.exp2(s_lo + ks * ((s_hi - s_lo) / 63))
        return sc, -sc * torch.exp2(r_lo + kr * ((r_hi - r_lo) / 63))

    def to_codec_k(self):
        """This Codec B weight re-encoded as Codec K: 10% fewer bytes, lossy.

        The nibbles are kept byte for byte; only each chunk's scales and mins
        are requantized. Re-picking the 4-bit codes against the requantized
        scales moved <0.1% of them, so the codes stay the bundle's own.

        Why logarithmic, and why the min as a ratio: measured on the real
        bundle, in units of the RMS error Codec B's 4-bit codes already carry,

            Q4_K's rule (6-bit linear against the row max)      0.22
            6-bit linear scale, min as a ratio of its scale     0.13
            this encoding                                        0.08
            8-bit linear (4.625 bits a weight, for reference)    0.055

        Linear codes against the row maximum give a small-scale group a large
        relative error, and a linear min is quantized against the row's
        largest min -- which can exceed a small group's whole step. On the
        decode, the Q4_K rule cost KL 1.4e-2 against Codec B; nearly all of it
        from the projections, not the LM head.

        Codec B's mins are all <= 0 in this model (so the ratio is positive);
        a positive one raises rather than being clipped.
        """
        if self.codec != "b":
            raise ValueError(f"already codec {self.codec!r}")
        sc, mn = self._scales_mins()
        if (mn > 0).any():
            raise ValueError("a positive Codec B min: Codec K stores -min / scale")

        def log_code(x, off):
            lx = torch.log2(x.clamp(min=2.0**-off))
            lo = torch.floor((lx.amin(1, keepdim=True) + off) * 16).clamp(0, 255)
            hi = torch.ceil((lx.amax(1, keepdim=True) + off) * 16).clamp(0, 255)
            step = ((hi - lo) / 16 / 63).clamp(min=1e-9)
            k = torch.round((lx - (lo / 16 - off)) / step).clamp(0, 63)
            return lo.to(torch.uint8), hi.to(torch.uint8), k.to(torch.uint8)

        s_lo, s_hi, ks = log_code(sc, _Q4K_S_OFF)
        r_lo, r_hi, kr = log_code(-mn / sc.clamp(min=2.0**-_Q4K_S_OFF), _Q4K_R_OFF)
        code = ks.view(-1, 256).int() | (kr.view(-1, 256).int() << 6)
        hi = (code >> 8).to(torch.uint8)
        ends = torch.stack([s_lo, s_hi, r_lo, r_hi], -1).view(-1, 128)
        c = self.data.view(-1, Q4NX_CHUNK_BYTES)
        out = torch.cat(
            [
                ends,
                (code & 0xFF).to(torch.uint8),
                hi[:, 0::2] | (hi[:, 1::2] << 4),
                c[:, 1024:],
            ],
            1,
        )
        return Q4NXWeight(out.reshape(-1), self.N, self.K, "k")


@triton.jit
def _q4nx_rows(
    X, W, blk, NBJ, j0, J, C2, KC, GLU_IN: tl.constexpr, CODEC_K: tl.constexpr
):
    """One 32-row block of `[1, K] @ Codec B [N, K]^T`, over chunks [j0, j0+J).

    Returns the block's partial rows as `[2, 16]` (row `16g + r`).

    The tile is `(g, grp, c, r)`: output row `16g + r`, K column `32grp + c`.
    Two facts about the chunk make that the natural shape:

    * the nibble byte at `(g, col, k)` holds rows `16g + 2k` (low) and
      `16g + 2k + 1` (high), so `join(lo, hi)` along `k` IS the row axis `r`,
      with no shuffle;
    * the scales and mins are `[grp][g][r]`, so they load as a contiguous
      `[2, 8, 1, 16]` and broadcast over `c`.

    Not split per 16-row half `g`, though the chunk would allow it without
    atomics: that doubles the program count and measured 1.16-1.41x faster in
    isolation, and 5% slower in the decode (in-process A/B, 48 tokens each).

    The first version indexed the scales per element instead, which compiled
    to ~65 two-byte gathers per thread per chunk and capped it at 15-32 GB/s;
    this one streams at the device's ~43 GB/s read ceiling on the large shapes.

    Every weight is `scale * q + min`, the codec's definition, accumulated
    elementwise across chunks with one cross-lane reduction at the end. The
    weights are not rounded to bf16 on the way, unlike the host-dequantized
    path -- so the two agree to ~1e-3 relative, not bitwise.

    `GLU_IN` takes `X` as a gate|up row, `[2K]`, and projects
    `gelu_tanh(gate) * up` -- the FFN's GeGLU, computed as each chunk of the
    activation is loaded. Elementwise, so it adds no serial step: fusing
    anything that needs a reduction (the RMSNorm, say) into this kernel's
    prologue or epilogue was measured slower than the launch it saved.
    """
    g = tl.arange(0, 2)[:, None, None, None]
    grp = tl.arange(0, 8)[None, :, None, None]
    c = tl.arange(0, 32)[None, None, :, None]
    k = tl.arange(0, 8)[None, None, None, :]
    r = tl.arange(0, 16)[None, None, None, :]
    boff = g * 2048 + (grp * 32 + c) * 8 + k  # [2, 8, 32, 8] nibble bytes
    soff = grp * 32 + g * 16 + r  # [2, 8, 1, 16] scales / mins
    xoff = grp * 32 + c  # [1, 8, 32, 1] activation
    acc = tl.zeros((2, 8, 32, 16), tl.float32)
    for jj in range(0, J):
        j = j0 + jj
        x = tl.load(X + j * 256 + xoff).to(tl.float32)
        if GLU_IN:
            u = tl.load(X + NBJ * 256 + j * 256 + xoff).to(tl.float32)
            x = x * tl.sigmoid(C2 * (x + KC * x * x * x)) * u
        x = x.to(tl.bfloat16).to(tl.float32)
        if CODEC_K:
            base = W + (blk * NBJ + j).to(tl.int64) * 4608
            ep = base.to(tl.pointer_type(tl.uint32))
            e = tl.load(ep + g * 16 + r)  # [2, 1, 1, 16]: this row's endpoints
            s_lo = (e & 0xFF).to(tl.float32) / 16 - 16.0
            s_st = (((e >> 8) & 0xFF).to(tl.float32) / 16 - 16.0 - s_lo) / 63
            r_lo = ((e >> 16) & 0xFF).to(tl.float32) / 16 - 8.0
            r_st = ((e >> 24).to(tl.float32) / 16 - 8.0 - r_lo) / 63
            hi = (tl.load(base + 384 + soff // 2) >> ((soff % 2) * 4)) & 0xF
            code = tl.load(base + 128 + soff).to(tl.int32) | (hi.to(tl.int32) << 8)
            ks = code & 63
            kr = code >> 6
            s = tl.exp2(s_lo + ks.to(tl.float32) * s_st)
            m = -s * tl.exp2(r_lo + kr.to(tl.float32) * r_st)
            b = tl.load(base + 512 + boff)
        else:
            base = W + (blk * NBJ + j).to(tl.int64) * 5120
            sp = base.to(tl.pointer_type(tl.bfloat16))
            s = tl.load(sp + soff).to(tl.float32)
            m = tl.load(sp + 256 + soff).to(tl.float32)
            b = tl.load(base + 1024 + boff)
        q = tl.reshape(tl.join(b & 0xF, b >> 4), (2, 8, 32, 16)).to(tl.float32)
        acc += x * (q * s + m)
    return tl.sum(tl.sum(acc, axis=2), axis=1)


@triton.jit
def _gemv_q4nx_kernel(
    X,
    W,
    C,
    NBJ,
    J_PER,
    C2,
    KC,
    SPLIT: tl.constexpr,
    GLU_IN: tl.constexpr,
    CODEC_K: tl.constexpr,
):
    """`[1, K] @ Codec B [N, K]^T -> [1, N]`, one program per 32 output rows.

    See `_q4nx_rows` for the tile. `SPLIT` > 1 gives each program `J_PER` of
    the row's `NBJ` chunks and reduces the partial rows with atomics into a
    zeroed `C`.
    """
    i = tl.program_id(0)
    js = tl.program_id(1)
    out = _q4nx_rows(X, W, i, NBJ, js * J_PER, J_PER, C2, KC, GLU_IN, CODEC_K)
    rows = i * 32 + tl.arange(0, 2)[:, None] * 16 + tl.arange(0, 16)[None, :]
    if SPLIT == 1:
        tl.store(C + rows, out)
    else:
        tl.atomic_add(C + rows, out, sem="relaxed")


def _q4nx_split(nbj):
    """Split-K for one GEMV: only on the deep `down` projections.

    Measured on gfx1103 over every shape this model issues. At N=1536 there
    are 48 row programs, too few to keep DRAM busy through a 24- or 48-chunk
    row; splitting it 4 or 8 ways took `down` from 30-37 to 38-39 GB/s. The
    other shapes gained under 0.05 ms or lost, once the zeroing launch the
    atomics need is paid for.

    Always a divisor of `nbj`: each split takes `nbj // split` chunks, so any
    remainder would be chunks no program reads. This model's `down` has 24 or
    48 and keeps its 4 and 8; a K of 6400 (25 chunks) does not split at all.
    """
    if nbj < 24:
        return 1
    split = min(8, nbj // 6)
    while nbj % split:
        split -= 1
    return split


def gemv_q4nx(x, w, glu_in=False):
    """x: [1, K] -> [1, N] against a `Q4NXWeight`. f32 out, as `gemv` returns.

    The decode's weight-bound half at 5 bits a weight rather than 16: every
    projection it issues is a GEMV that streams its matrix once, so bytes are
    the cost, and reading the bundle's own encoding cuts them 3.2x.

    `glu_in`: `x` is a gate|up row, `[1, 2K]`, and the product is taken
    against `gelu_tanh(gate) * up` -- the FFN's GeGLU, fused into the load.
    """
    # bf16-rounded in the kernel, as `gemv` does, and for the same reasons.
    x = x.reshape(-1).contiguous()
    if x.numel() != w.K * (2 if glu_in else 1):
        raise ValueError(f"activation is {x.numel()} wide, weight K is {w.K}")
    nbj = w.K // Q4NX_COLS
    split = _q4nx_split(nbj)
    alloc = torch.zeros if split > 1 else torch.empty
    out = alloc(w.N, dtype=torch.float32, device=x.device)
    _gemv_q4nx_kernel[(w.N // Q4NX_ROWS, split)](
        x,
        w.data,
        out,
        nbj,
        nbj // split,
        _GELU_2C,
        _GELU_K,
        SPLIT=split,
        GLU_IN=glu_in,
        CODEC_K=w.codec == "k",
        num_warps=8,
    )
    return out.reshape(1, w.N)


def dequant_q4nx(w):
    """A `Q4NXWeight` as float32 `[K, N]`, in torch -- the reference for tests."""
    N, K = w.N, w.K
    nbi, nbj = N // Q4NX_ROWS, K // Q4NX_COLS
    c = w.data.view(nbi * nbj, w.chunk_bytes)
    sc, mn = w._scales_mins()
    qb = c[:, w.chunk_bytes - 4096 :].view(-1, 2, 256, 8)
    q = torch.stack([qb & 0xF, qb >> 4], -1).float()  # [nb, g, col, k, bit]
    q = q.permute(0, 1, 3, 4, 2).reshape(-1, 32, 256)  # row = 16g + 2k + bit
    s = sc.permute(0, 2, 1).repeat_interleave(32, 2)
    m = mn.permute(0, 2, 1).repeat_interleave(32, 2)
    wt = (s * q + m).view(nbi, nbj, 32, 256).permute(0, 2, 1, 3).reshape(N, K)
    return wt.T


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
@triton.jit
def _rmsnorm_kernel(
    X,
    W,
    Y,
    N,
    eps,
    HAS_W: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """RMSNorm over the last axis of `[rows, N]`, one program per row.

    `HAS_W` because Gemma4's value norm is weightless -- the reference applies
    no scale at all there, so a kernel that always multiplied would need a ones
    vector allocated and multiplied by for nothing.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x) / N + eps)
    y = x * inv
    if HAS_W:
        y = y * tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * N + offs, y, mask=mask)


def rmsnorm(x, weight, eps):
    """x: [rows, N] -> same, f32. `weight=None` is the weightless norm."""
    x = x.contiguous()
    rows, N = x.shape
    out = torch.empty_like(x, dtype=torch.float32)
    _rmsnorm_kernel[(rows,)](
        x,
        weight if weight is not None else x,  # unread when HAS_W is False
        out,
        N,
        eps,
        HAS_W=weight is not None,
        BLOCK_N=_pow2(N),
    )
    return out


@triton.jit
def _rmsnorm_residual_kernel(
    X,
    W,
    R,
    Y,
    W2,
    Y2,
    N,
    eps,
    scale,
    BLOCK_N: tl.constexpr,
    HAS_NEXT: tl.constexpr,
):
    """`(residual + rmsnorm(x) * weight) * scale`, one row, one launch.

    The Gemma4 block closes each of its three sublayers this way, so fusing it
    turns nine launches per layer into three. `scale` is the per-layer
    `out_scale` on the sublayer that carries one and 1.0 on the others -- a
    scalar the kernel multiplies by anyway, so it costs nothing to always take
    it and saves a separate pass over the row where it is not 1.

    `HAS_NEXT` also stores `rmsnorm(result) * W2` to `Y2`: the pre-norm of the
    sublayer that follows. This program already holds the whole row, so the
    next norm is free here -- and not free anywhere else: fused into the
    projection that consumes it, every program of that GEMV would have to
    reduce the row before its first weight load, which measured slower than
    the launch it saved.
    """
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x) / N + eps)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + offs, mask=mask, other=0.0).to(tl.float32)
    y = (r + x * inv * w) * scale
    tl.store(Y + offs, y, mask=mask)
    if HAS_NEXT:
        inv2 = 1.0 / tl.sqrt(tl.sum(y * y) / N + eps)
        w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y2 + offs, y * inv2 * w2, mask=mask)


def rmsnorm_residual(x, weight, eps, residual, scale=1.0, next_norm=None):
    """`(residual + rmsnorm(x) * weight) * scale` over one row, f32.

    With `next_norm`, returns `(result, rmsnorm(result) * next_norm)`.
    """
    x = x.reshape(-1).contiguous()
    N = x.numel()
    out = torch.empty(N, dtype=torch.float32, device=x.device)
    nxt = torch.empty_like(out) if next_norm is not None else out
    _rmsnorm_residual_kernel[(1,)](
        x,
        weight.contiguous(),
        residual.reshape(-1).contiguous(),
        out,
        next_norm.contiguous() if next_norm is not None else out,
        nxt,
        N,
        eps,
        scale,
        BLOCK_N=_pow2(N),
        HAS_NEXT=next_norm is not None,
    )
    if next_norm is None:
        return out.reshape(1, N)
    return out.reshape(1, N), nxt.reshape(1, N)


@triton.jit
def _ple_combine_kernel(P, W, T, Y, eps, s_in, s_out, N: tl.constexpr):
    """`(rmsnorm(p * s_in) * w + t) * s_out` on one layer's row."""
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    p = tl.load(P + row * N + offs) * s_in
    inv = 1.0 / tl.sqrt(tl.sum(p * p) / N + eps)
    w = tl.load(W + offs).to(tl.float32)
    t = tl.load(T + row * N + offs).to(tl.float32)
    tl.store(Y + row * N + offs, (p * inv * w + t) * s_out)


def ple_combine(proj, weight, tbl, s_in, s_out, eps):
    """Gemma4's per-layer inputs for one token, all layers in one launch.

    `proj` is `[1, L * n]`, every layer's `model_proj` output side by side;
    `tbl` is this token's `[L, n]` rows of the per-layer embedding table. Row
    `l` of the result is `(rmsnorm(proj_l * s_in) * weight + tbl_l) * s_out`,
    which was five launches per layer as separate torch and Triton ops.
    """
    L, n = tbl.shape
    out = torch.empty((L, n), dtype=torch.float32, device=proj.device)
    _ple_combine_kernel[(L,)](
        proj.reshape(-1).contiguous(),
        weight.contiguous(),
        tbl.contiguous(),
        out,
        eps,
        float(s_in),
        float(s_out),
        N=n,
    )
    return out


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
@triton.jit
def _rope_kernel(X, ROW, Y, half, BLOCK: tl.constexpr):
    """Half-split rotary on one token's heads, against one table row.

    The pairing is (i, i + dh/2), matching `LlamaPrefill._rope` and the device's
    `rope.cc`; the partial rotary on Gemma4's full-attention layers lives in the
    table, not here.
    """
    h = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < half
    cos = tl.load(ROW + offs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(ROW + half + offs, mask=mask, other=0.0).to(tl.float32)
    base = h * 2 * half
    x1 = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X + base + half + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + base + offs, x1 * cos - x2 * sin, mask=mask)
    tl.store(Y + base + half + offs, x1 * sin + x2 * cos, mask=mask)


def rope(x, lut_row, n_heads, dh):
    """x: [1, n_heads*dh] -> same, rotated at the position `lut_row` encodes."""
    x = x.reshape(-1).contiguous()
    out = torch.empty_like(x, dtype=torch.float32)
    half = dh // 2
    _rope_kernel[(n_heads,)](x, lut_row.contiguous(), out, half, BLOCK=_pow2(half))
    return out.reshape(1, n_heads * dh)


# `pos` moves every token and `n_fan` per layer: neither may specialize.
@triton.jit(do_not_specialize=["pos", "n_fan"])
def _qkv_post_kernel(
    QKV,
    QN,
    KN,
    ROW,
    QOUT,
    SLAB,
    FAN,
    n_fan,
    pos,
    eps,
    layer_stride,
    region_stride,
    NQ: tl.constexpr,
    HALF: tl.constexpr,
    LANE_SHIFT: tl.constexpr,
    REGION_W: tl.constexpr,
    DH_A: tl.constexpr,
    N_CU: tl.constexpr,
    HAS_KV: tl.constexpr,
):
    """Everything between the QKV projection and attention, for one token.

    Program `h < NQ` is query head `h`: RMSNorm with `QN`, then half-split
    RoPE against the position's table row -- the same arithmetic, in the same
    order, as `_rmsnorm_kernel` followed by `_rope_kernel`. Program `NQ` is
    the single KV head: `k` gets the same with `KN`, `v` the weightless norm,
    and both are rounded to bf16 and written to row `pos` of every slab in
    `FAN`, at both attention CUs' copies and the padded-head lane map of
    `kv_layout` -- what `Gemma4GpuDecode._append_kv` did with ~9 launches.
    """
    h = tl.program_id(0)
    o = tl.arange(0, HALF)
    cos = tl.load(ROW + o).to(tl.float32)
    sin = tl.load(ROW + HALF + o).to(tl.float32)
    if h < NQ:
        base = h * 2 * HALF
        x1 = tl.load(QKV + base + o).to(tl.float32)
        x2 = tl.load(QKV + base + HALF + o).to(tl.float32)
        inv = 1.0 / tl.sqrt((tl.sum(x1 * x1) + tl.sum(x2 * x2)) / (2 * HALF) + eps)
        y1 = x1 * inv * tl.load(QN + o).to(tl.float32)
        y2 = x2 * inv * tl.load(QN + HALF + o).to(tl.float32)
        tl.store(QOUT + base + o, y1 * cos - y2 * sin)
        tl.store(QOUT + base + HALF + o, y1 * sin + y2 * cos)
    elif HAS_KV:
        kb = NQ * 2 * HALF
        x1 = tl.load(QKV + kb + o).to(tl.float32)
        x2 = tl.load(QKV + kb + HALF + o).to(tl.float32)
        inv = 1.0 / tl.sqrt((tl.sum(x1 * x1) + tl.sum(x2 * x2)) / (2 * HALF) + eps)
        y1 = x1 * inv * tl.load(KN + o).to(tl.float32)
        y2 = x2 * inv * tl.load(KN + HALF + o).to(tl.float32)
        k1 = (y1 * cos - y2 * sin).to(tl.bfloat16)
        k2 = (y1 * sin + y2 * cos).to(tl.bfloat16)
        vb = kb + 2 * HALF
        v1 = tl.load(QKV + vb + o).to(tl.float32)
        v2 = tl.load(QKV + vb + HALF + o).to(tl.float32)
        inv = 1.0 / tl.sqrt((tl.sum(v1 * v1) + tl.sum(v2 * v2)) / (2 * HALF) + eps)
        v1 = (v1 * inv).to(tl.bfloat16)
        v2 = (v2 * inv).to(tl.bfloat16)
        for f in range(0, n_fan):
            row = SLAB + tl.load(FAN + f).to(tl.int64) * layer_stride + pos * REGION_W
            for cu in tl.static_range(N_CU):
                lo = row + cu * DH_A + o
                hi = lo + HALF + LANE_SHIFT
                tl.store(lo, k1)
                tl.store(hi, k2)
                tl.store(lo + region_stride, v1)
                tl.store(hi + region_stride, v2)


def qkv_post(qkv, q_norm, k_norm, lut_row, n_q, dh, eps, kv=None):
    """Head norms, RoPE and the KV-cache append, in one launch.

    `qkv` is the projection's `[1, (n_q + 2) * dh]` output, or `[1, n_q * dh]`
    on a KV-shared layer, which has no k or v. Returns the rotated q,
    `[1, n_q * dh]` f32.

    `kv`, where the layer owns its cache, is `(slab, fan, pos, region_stride)`:
    the `[n_layers, layer_elems]` bf16 slab, a device int32 tensor of the
    layers whose slabs take this row (`kv_layout`'s fan-out), the row, and
    the K-to-V region offset in elements.
    """
    from kv_layout import DH_A, N_ATTN_CU, REGION_W

    qkv = qkv.reshape(-1).contiguous()
    q = torch.empty(n_q * dh, dtype=torch.float32, device=qkv.device)
    if kv is not None:
        slab, fan, pos, region_stride = kv
        if slab.stride(1) != 1:
            raise ValueError("the KV slab's rows must be contiguous")
        # The kernel writes row `pos` by raw offset, so a row past the region
        # would land in the V region or the next layer rather than fault.
        attn_maxl = region_stride // REGION_W
        if not 0 <= pos < attn_maxl:
            raise ValueError(f"KV row {pos} is outside the slab's {attn_maxl} rows")
        args = (slab, fan, fan.numel(), pos, eps, slab.stride(0), region_stride)
    else:
        args = (q, q, 0, 0, eps, 0, 0)
    _qkv_post_kernel[(n_q + (kv is not None),)](
        qkv,
        q_norm.contiguous(),
        k_norm.contiguous() if k_norm is not None else q_norm,
        lut_row.contiguous(),
        q,
        *args,
        NQ=n_q,
        HALF=dh // 2,
        LANE_SHIFT=DH_A // 2 - dh // 2,
        REGION_W=REGION_W,
        DH_A=DH_A,
        N_CU=N_ATTN_CU,
        HAS_KV=kv is not None,
    )
    return q.reshape(1, n_q * dh)


# ---------------------------------------------------------------------------
# GeGLU
# ---------------------------------------------------------------------------
@triton.jit
def _geglu_kernel(G, U, Y, n, C2, KC, BLOCK: tl.constexpr):
    """gelu_tanh(gate) * up, elementwise."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0).to(tl.float32)
    z = C2 * (g + KC * g * g * g)
    tl.store(Y + offs, g * tl.sigmoid(z) * u, mask=mask)


def geglu(gate, up, block=1024):
    """gelu_tanh(gate) * up over matching flat tensors -> f32."""
    gate = gate.reshape(-1).contiguous()
    up = up.reshape(-1).contiguous()
    n = gate.numel()
    out = torch.empty(n, dtype=torch.float32, device=gate.device)
    _geglu_kernel[(triton.cdiv(n, block),)](
        gate, up, out, n, _GELU_2C, _GELU_K, BLOCK=block
    )
    return out.reshape(1, n)


# ---------------------------------------------------------------------------
# Logit softcap
# ---------------------------------------------------------------------------
@triton.jit
def _softcap_kernel(X, Y, n, cap, BLOCK: tl.constexpr):
    """`cap * tanh(x / cap)`, elementwise.

    Spelled through sigmoid for the same reason `_geglu_kernel` is:
    `tanh(z) = 2*sigmoid(2z) - 1` is exact, and `tl.math.tanh` is not available
    in this Triton.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    z = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32) / cap
    tl.store(Y + offs, cap * (2.0 * tl.sigmoid(2.0 * z) - 1.0), mask=mask)


def logit_softcap(x, cap, block=1024):
    """Gemma's final logit softcap over the vocabulary row."""
    x = x.reshape(-1).contiguous()
    n = x.numel()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    _softcap_kernel[(triton.cdiv(n, block),)](x, out, n, float(cap), BLOCK=block)
    return out


# ---------------------------------------------------------------------------
# Attention, one query row against a cache
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["S"])
def _attn_decode_kernel(
    Q,
    KC,
    VC,
    OUT,
    S,
    dh,
    scale,
    stride_s,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LANE_SHIFT: tl.constexpr,
    HALF: tl.constexpr,
):
    """softmax(q . K^T * scale) . V for one token, one program per query head.

    No causal mask: every cached key is at or before this position by
    construction, and the window is applied by the caller as a lower bound on
    the slice. MQA -- every query head reads the one kv head's cache.

    `BLOCK_S` is a fixed tile and `S` is a runtime argument, so one compile
    serves every context length. `S` is also exempt from Triton's integer
    specialization, which otherwise compiles a second variant the first time
    `S % 16 == 0` -- a 1.2 s stall at the 16th position of every generation. A tile derived from `S` would instead track
    the position and recompile on nearly every token. mlir-air's llama-1b
    decode makes the same choice: one xclbin with a compile-time attention
    loop serving every L in [1, ATTN_MAXL], rather than a template per
    window.

    The running max and sum are the online-softmax pair, so the scores for the
    whole context never exist at once.

    `stride_s` and the lane map are what let this read the cache WHERE IT LIES
    rather than a repacked copy. In the shared layout a row is `REGION_W` wide
    and a narrow head is scattered as `[real_lo | zeros | real_hi | zeros]`
    (see `kv_layout`), so dimension `d` of the head is at

        d < HALF ? d : d + LANE_SHIFT

    with `LANE_SHIFT = DH_A//2 - dh//2`, which is zero for a full-width head --
    so the same expression is the identity there and there is no second case to
    keep in step. The real lanes are two contiguous 128-element runs, so the
    loads stay coalesced; nothing extra is fetched.
    """
    h = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < dh
    lane = tl.where(offs_d < HALF, offs_d, offs_d + LANE_SHIFT)
    q = tl.load(Q + h * dh + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m_i = float("-inf")
    l_i = 0.0

    for j0 in range(0, S, BLOCK_S):
        offs_s = j0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        k = tl.load(
            KC + offs_s[:, None] * stride_s + lane[None, :],
            mask=mask_s[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        s = tl.sum(q[None, :] * k, axis=1) * scale
        s = tl.where(mask_s, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha

        v = tl.load(
            VC + offs_s[:, None] * stride_s + lane[None, :],
            mask=mask_s[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    tl.store(OUT + h * dh + offs_d, acc / l_i, mask=mask_d)


def attn_decode(q, kc, vc, n_heads, dh, scale, block_s=64, lane_shift=0):
    """q: [1, n_heads*dh] against kc/vc: [S, *] -> [1, n_heads*dh].

    `kc`/`vc` are rows of the cache, NOT necessarily `[S, dh]` and NOT required
    to be contiguous: the row stride is read off the tensor and `lane_shift`
    places the second half of a padded head (see `_attn_decode_kernel`). A
    caller holding a dense `[S, dh]` cache passes `lane_shift=0` and gets the
    previous behaviour exactly.

    The tensors are deliberately NOT forced contiguous. They used to be, which
    was free while every caller held a dense cache and would now silently
    repack the shared cache on every layer of every token -- reintroducing the
    copy this signature exists to avoid.

    `block_s` is fixed, so the only constexprs that vary are `BLOCK_D` and the
    lane map, and `dh` takes two values on this model -- two compiles per run.
    """
    S = kc.shape[0]
    if kc.stride(0) != vc.stride(0):
        raise ValueError(
            f"K and V rows must be equally spaced, got {kc.stride(0)} and "
            f"{vc.stride(0)}: one stride is passed to the kernel for both"
        )
    q = q.reshape(-1).contiguous()
    out = torch.empty(n_heads * dh, dtype=torch.float32, device=q.device)
    _attn_decode_kernel[(n_heads,)](
        q,
        kc,
        vc,
        out,
        S,
        dh,
        scale,
        kc.stride(0),
        BLOCK_S=block_s,
        BLOCK_D=_pow2(dh),
        LANE_SHIFT=lane_shift,
        HALF=dh // 2,
    )
    return out.reshape(1, n_heads * dh)


# ===========================================================================
# Prefill shapes: N rows rather than one
# ===========================================================================
# Everything above assumes a single activation row, which is decode. These are
# the same operators at N > 1, and they are separate kernels rather than the
# same ones with a loop because the shape changes what is worth doing: a GEMV's
# K-reduction becomes a tiled `tl.dot`, and attention stops fitting its scores
# in registers and needs an online softmax.


@triton.jit
def _matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_bk,
    stride_bn,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """`[M, K] @ [K, N] -> [M, N]`, bf16 in, f32 out.

    f32 out because that is what `LlamaPrefill._matmul` returns and what the
    next operator reads; qwen2_5's `matmul_kernel_gpu` stores bf16 because its
    chain wants bf16 next.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(
            A + offs_m[:, None] * stride_am + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)
    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def matmul(x, w, block_m=64, block_n=64, block_k=64):
    """x: [M, K] -> [M, N] against w: [K, N]. f32 out.

    Falls through to `gemv` at M == 1: `tl.dot` needs a minimum M on AMD and
    would pad the row out to it.
    """
    if x.shape[0] == 1:
        return gemv(x, w)
    M, K = x.shape
    _, N = w.shape
    x = x.to(torch.bfloat16).contiguous()
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    _matmul_kernel[(triton.cdiv(M, block_m), triton.cdiv(N, block_n))](
        x,
        w,
        out,
        M,
        N,
        K,
        x.stride(0),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


@triton.jit
def _rope_batch_kernel(
    X,
    LUT,
    Y,
    stride_x,
    stride_l,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
    TBLOCK: tl.constexpr,
):
    """Half-split rotary over N rows, one program per (row, head).

    `2*HALF` of each head's `DH` lanes are rotated and the rest copied through.
    Equal when the whole head rotates; Phi-4-mini's partial_rotary_factor=0.75
    leaves a tail. The pairing is (i, i+HALF) *within the rotated slice*, so it
    never reaches across into the tail.
    """
    row = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    cos = tl.load(LUT + row * stride_l + offs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(LUT + row * stride_l + HALF + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    base = row * stride_x + h * DH
    x1 = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X + base + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + base + offs, x1 * cos - x2 * sin, mask=mask)
    tl.store(Y + base + HALF + offs, x1 * sin + x2 * cos, mask=mask)
    if TBLOCK > 0:
        toffs = 2 * HALF + tl.arange(0, TBLOCK)
        tmask = toffs < DH
        tail = tl.load(X + base + toffs, mask=tmask, other=0.0).to(tl.float32)
        tl.store(Y + base + toffs, tail, mask=tmask)


def rope_batch(x, lut, n_heads, dh, rot=None):
    """x: [N, n_heads*dh], lut: [N, rot] -> [N, n_heads*dh], f32.

    `rot` is how many of each head's lanes rotate, defaulting to all of them.
    The LUT's width is what says how wide the rotation is, so it is checked
    against `rot` rather than trusted: a LUT sized for a different rotation
    would reinterpret the head boundary instead of failing.
    """
    rot = dh if rot is None else rot
    if rot % 2 or not 0 < rot <= dh:
        raise ValueError(f"rot must be even and in (0, {dh}], got {rot}")
    if lut.shape[-1] != rot:
        raise ValueError(f"rope lut is {lut.shape[-1]} wide, need {rot} (cos|sin)")
    N = x.shape[0]
    x = x.contiguous()
    lut = lut.contiguous()
    out = torch.empty_like(x, dtype=torch.float32)
    _rope_batch_kernel[(N, n_heads)](
        x,
        lut,
        out,
        x.stride(0),
        lut.stride(0),
        DH=dh,
        HALF=rot // 2,
        BLOCK=_pow2(rot // 2),
        TBLOCK=_pow2(dh - rot) if dh > rot else 0,
    )
    return out


@triton.jit
def _attn_prefill_kernel(
    Q,
    K,
    V,
    OUT,
    N,
    dh,
    scale,
    rep,
    window,
    stride_q,
    stride_k,
    HAS_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Causal GQA with an optional sliding window, flash-style.

    One program per (query head, query block), streaming the key blocks with an
    online softmax so the `[n_q, N, N]` score matrix the torch body materializes
    never exists -- that matrix is where a long prompt spends its prefill.

    The mask is the torch body's, restated per element: `j > i` is the causal
    half, and with a window `i - j >= window` is the other. `j == i` survives
    both, so no row is fully masked.
    """
    h = tl.program_id(0)
    pid_m = tl.program_id(1)
    kvh = h // rep  # GQA: several query heads share one kv head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < N
    mask_d = offs_d < dh

    q = tl.load(
        Q + offs_m[:, None] * stride_q + h * dh + offs_d[None, :],
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for j0 in range(0, N, BLOCK_N):
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        k = tl.load(
            K + offs_n[:, None] * stride_k + kvh * dh + offs_d[None, :],
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        # `tl.dot`, not a broadcast multiply-reduce: the latter materializes a
        # [BLOCK_M, BLOCK_N, BLOCK_D] intermediate, which at Gemma4's 512-wide
        # heads is 72 KB against this part's 64 KB of LDS.
        s = tl.dot(q, tl.trans(k)) * scale

        keep = mask_n[None, :] & (offs_n[None, :] <= offs_m[:, None])
        if HAS_WINDOW:
            keep = keep & (offs_m[:, None] - offs_n[None, :] < window)
        s = tl.where(keep, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # A key block entirely outside the sliding window leaves every score
        # masked, so `m_new` is -inf and `exp(-inf - -inf)` is NaN, which then
        # poisons `l_i` and `acc` for the rest of the row. Substituting any
        # finite value in the *exponent* fixes it without a branch: both
        # `exp(m_i - 0)` and `exp(s - 0)` are then `exp(-inf) = 0`, so the
        # block contributes nothing and the running state is untouched.
        # `m_i` keeps the real -inf, so the first block that does have keys
        # still sets the maximum correctly.
        #
        # Reachable on this model: the sliding layers use a 512-token window,
        # and a prompt past roughly `window + BLOCK_N` NaNs on four layers in
        # five. It was not caught because the tests stopped at N=163.
        m_exp = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_exp)
        p = tl.exp(s - m_exp[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            V + offs_n[:, None] * stride_k + kvh * dh + offs_d[None, :],
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(
        OUT + offs_m[:, None] * stride_q + h * dh + offs_d[None, :],
        out,
        mask=mask_m[:, None] & mask_d[None, :],
    )


def _attn_blocks(dh):
    """Query/key tiles that fit 64 KB of LDS at this head width.

    The tiles are bounded by `dh`, not chosen for throughput: q, k, v and the
    accumulator are each `tile x dh` f32, so a 512-wide head -- which Gemma4's
    full-attention layers have, four times what a typical flash-attention
    kernel is tuned for -- leaves room for a 16-row tile and no more.
    """
    if dh >= 512:
        return 16, 16
    if dh >= 256:
        return 16, 32
    return 32, 64


def attn_prefill(
    q, k, v, n_q, n_kv, dh, window=None, scale=None, block_m=None, block_n=None
):
    """q: [N, n_q*dh], k/v: [N, n_kv*dh] -> [N, n_q*dh]. Causal, GQA, windowed."""
    N = q.shape[0]
    if block_m is None or block_n is None:
        bm, bn = _attn_blocks(dh)
        block_m = block_m or bm
        block_n = block_n or bn
    scale = dh**-0.5 if scale is None else scale
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty((N, n_q * dh), dtype=torch.float32, device=q.device)
    _attn_prefill_kernel[(n_q, triton.cdiv(N, block_m))](
        q,
        k,
        v,
        out,
        N,
        dh,
        scale,
        n_q // n_kv,
        window if window is not None else 0,
        q.stride(0),
        k.stride(0),
        HAS_WINDOW=window is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=_pow2(dh),
    )
    return out
