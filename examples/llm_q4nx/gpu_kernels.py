# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton kernels for the iGPU half of a hybrid run.

`kernels.py` is this directory's NPU kernel set; this is its GPU counterpart,
and it exists for the reason `README.md` states outright -- *"PyTorch is used by
the examples as a CPU reference."* torch is the oracle these are checked
against, not the way an example computes on a device. `examples/gpt2` and
`examples/qwen2_5` have had `*_kernel_gpu` for exactly this since they grew a
GPU path; `llm_q4nx` had no GPU path at all until now, which is why it had none
of these.

Written for **decode**, where the shapes are not the prefill's
------------------------------------------------------------
Every kernel here assumes one row of activations. That is not a simplification,
it is the whole design point: at N=1 every projection is a GEMV, `tl.dot` has
nothing to contract over on AMD, and the cost is dominated by *launch count*
rather than arithmetic. Gemma4-E2B issues ~281 projections per token -- 8 per
layer across 35 layers, plus the head -- so what matters is that each is one
kernel rather than a torch op with its own dispatch, allocation and type
promotion.

So `_gemv_kernel` reduces over K instead of calling `tl.dot`, and the
elementwise kernels are shaped to be fused into as few launches as the forward
allows. A prefill kernel set would look nothing like this; use `kernels.py`.

Bucketed context length
-----------------------
`attn_decode` is compiled per `BLOCK_S`, which would otherwise track the
position and recompile on every token. It is rounded up to a power of two and
masked, so a 2048-token session pays at most eleven compiles instead of two
thousand -- the same bargain `kernels.ROW_TILE` makes on the NPU side, for the
same reason.
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

    **Enter it once around a whole generation, not per step.** Re-entering is
    free -- it no-ops when the GPU backend is already active -- but *switching*
    is not: `set_active` drops Triton's compiled-kernel cache, so a scope
    entered per token recompiles the shapes that token uses. Measured over 64
    tokens, per-step scoping spent 7547 ms recompiling against 883 ms when the
    scope is held across the loop, which is the difference between 9.30 s and
    2.74 s of wall clock for the same output.
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
    x = x.reshape(-1).contiguous()
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
    """
    h = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, BLOCK_S)
    mask_d = offs_d < dh
    mask_s = offs_s < S

    q = tl.load(Q + h * dh + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    k = tl.load(
        KC + offs_s[:, None] * dh + offs_d[None, :],
        mask=mask_s[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(q[None, :] * k, axis=1) * scale
    # Masked lanes to -inf so they leave the max alone and exp to zero.
    scores = tl.where(mask_s, scores, float("-inf"))
    p = tl.exp(scores - tl.max(scores, axis=0))
    p = p / tl.sum(p, axis=0)

    v = tl.load(
        VC + offs_s[:, None] * dh + offs_d[None, :],
        mask=mask_s[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(OUT + h * dh + offs_d, tl.sum(p[:, None] * v, axis=0), mask=mask_d)


def attn_decode(q, kc, vc, n_heads, dh, scale):
    """q: [1, n_heads*dh] against kc/vc: [S, dh] -> [1, n_heads*dh].

    `BLOCK_S` is bucketed to a power of two; see the module docstring.
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
        BLOCK_S=_pow2(S),
        BLOCK_D=_pow2(dh),
    )
    return out.reshape(1, n_heads * dh)
