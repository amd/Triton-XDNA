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
    stride_bk,
    stride_bn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """`[1, K] @ [K, N] -> [1, N]`, f32 accumulation.

    A reduction over K rather than `tl.dot`: with one activation row there is
    no M to tile, and a dot would pad M to the hardware's minimum and throw the
    work away.

    Both of B's strides are taken rather than assuming the N axis is packed.
    The LM head is reached as `lm_head.T`, a transposed view whose *K* axis is
    the contiguous one; assuming otherwise reads the wrong elements and still
    produces plausible logits -- the decode ran and emitted fluent nonsense.
    """
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(A + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        b = tl.load(
            B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(a[:, None] * b, axis=0)
    tl.store(C + offs_n, acc, mask=mask_n)


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


def gemv(x, w, block_n=None, block_k=None):
    """x: [1, K] -> [1, N] against w: [K, N]. f32 out, as `_mm` returns.

    `w` may be a transposed view -- the LM head is -- so its strides are read
    rather than assumed, and it is deliberately NOT forced contiguous: that
    would copy a 262144x1536 weight on every token.
    """
    K, N = w.shape
    if block_n is None or block_k is None:
        bn, bk = _gemv_blocks(K, N)
        block_n = block_n or bn
        block_k = block_k or bk
    # bf16 in, f32 accumulation -- `LlamaPrefill._matmul` rounds the activation
    # before the product and the NPU GEMM does the same, so feeding f32 here
    # would make the GPU decode compute from different inputs than the path it
    # is meant to reproduce, and drift a token at a time.
    x = x.reshape(-1).to(torch.bfloat16).contiguous()
    out = torch.empty(N, dtype=torch.float32, device=x.device)
    _gemv_kernel[(triton.cdiv(N, block_n),)](
        x,
        w,
        out,
        K,
        N,
        w.stride(0),
        w.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out.reshape(1, N)


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
    N,
    eps,
    scale,
    BLOCK_N: tl.constexpr,
):
    """`(residual + rmsnorm(x) * weight) * scale`, one row, one launch.

    The Gemma4 block closes each of its three sublayers this way, so fusing it
    turns nine launches per layer into three. `scale` is the per-layer
    `out_scale` on the sublayer that carries one and 1.0 on the others -- a
    scalar the kernel multiplies by anyway, so it costs nothing to always take
    it and saves a separate pass over the row where it is not 1.
    """
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x) / N + eps)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + offs, (r + x * inv * w) * scale, mask=mask)


def rmsnorm_residual(x, weight, eps, residual, scale=1.0):
    """`(residual + rmsnorm(x) * weight) * scale` over one row, f32."""
    x = x.reshape(-1).contiguous()
    N = x.numel()
    out = torch.empty(N, dtype=torch.float32, device=x.device)
    _rmsnorm_residual_kernel[(1,)](
        x,
        weight.contiguous(),
        residual.reshape(-1).contiguous(),
        out,
        N,
        eps,
        scale,
        BLOCK_N=_pow2(N),
    )
    return out.reshape(1, N)


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
@triton.jit
def _attn_decode_kernel(
    Q,
    KC,
    VC,
    OUT,
    S,
    dh,
    scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """softmax(q . K^T * scale) . V for one token, one program per query head.

    No causal mask: every cached key is at or before this position by
    construction, and the window is applied by the caller as a lower bound on
    the slice. MQA -- every query head reads the one kv head's cache.

    `BLOCK_S` is a fixed tile and `S` is a runtime argument, so one compile
    serves every context length. A tile derived from `S` would instead track
    the position and recompile on nearly every token. mlir-air's llama-1b
    decode makes the same choice: one xclbin with a compile-time attention
    loop serving every L in [1, ATTN_MAXL], rather than a template per
    window.

    The running max and sum are the online-softmax pair, so the scores for the
    whole context never exist at once.
    """
    h = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < dh
    q = tl.load(Q + h * dh + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m_i = float("-inf")
    l_i = 0.0

    for j0 in range(0, S, BLOCK_S):
        offs_s = j0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        k = tl.load(
            KC + offs_s[:, None] * dh + offs_d[None, :],
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
            VC + offs_s[:, None] * dh + offs_d[None, :],
            mask=mask_s[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    tl.store(OUT + h * dh + offs_d, acc / l_i, mask=mask_d)


def attn_decode(q, kc, vc, n_heads, dh, scale, block_s=64):
    """q: [1, n_heads*dh] against kc/vc: [S, dh] -> [1, n_heads*dh].

    `block_s` is fixed, so the only constexpr that varies is `BLOCK_D`, and
    `dh` takes two values on this model -- two compiles for the whole run.
    """
    S = kc.shape[0]
    q = q.reshape(-1).contiguous()
    out = torch.empty(n_heads * dh, dtype=torch.float32, device=q.device)
    _attn_decode_kernel[(n_heads,)](
        q,
        kc.contiguous(),
        vc.contiguous(),
        out,
        S,
        dh,
        scale,
        BLOCK_S=block_s,
        BLOCK_D=_pow2(dh),
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
    X, LUT, Y, n_heads, half, stride_x, stride_l, BLOCK: tl.constexpr
):
    """Half-split rotary over N rows, one program per (row, head)."""
    row = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = offs < half
    cos = tl.load(LUT + row * stride_l + offs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(LUT + row * stride_l + half + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    base = row * stride_x + h * 2 * half
    x1 = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X + base + half + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + base + offs, x1 * cos - x2 * sin, mask=mask)
    tl.store(Y + base + half + offs, x1 * sin + x2 * cos, mask=mask)


def rope_batch(x, lut, n_heads, dh):
    """x: [N, n_heads*dh], lut: [N, dh] -> [N, n_heads*dh], f32."""
    N = x.shape[0]
    x = x.contiguous()
    lut = lut.contiguous()
    out = torch.empty_like(x, dtype=torch.float32)
    half = dh // 2
    _rope_batch_kernel[(N, n_heads)](
        x, lut, out, n_heads, half, x.stride(0), lut.stride(0), BLOCK=_pow2(half)
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
