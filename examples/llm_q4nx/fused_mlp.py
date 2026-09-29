# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The Q4NX gated MLP as one multi-launch ELF, with the weights staged once.

Why this exists
---------------
Under XRT the generated launcher `memcpy`s every pointer argument into a BO on
*every* launch, and nothing in that path inspects the argument for pages it
already has (`_resident_copy`/`shared_empty` in `kernels.py` are real, but both
return their input unchanged unless the runtime is HSA). A Gemma4 prefill
therefore re-stages ~5 GiB of padded weights per call, which is why its NPU
prefill measured 26.9 s against the CPU backend's 1.67 s -- and why the two
barely move with prompt length. `gate_up` and `down` alone are 79-81% of those
bytes.

`NPUChain` already answers this: `static_indices` stages an operand host->device
on the *first* call only, and `bo_key` gives each layer its own BO set. So the
weights land once per process instead of once per token, and the MLP's three
dispatches -- each costing ~24 ms of fixed overhead regardless of size, as
`gemma4_prefill.load_weights` notes -- collapse into one.

Lineage
-------
This is `examples/qwen2_5/model.py`'s `_FusedMLP`, narrowed. Qwen's MLP is the
same *diamond* -- `gate` and `up` both read the same input, merge through a
gated activation, then `down` -- and its chain is the working reference for
that shape on this backend. Read it alongside this file.

Two deliberate differences:

* **Activation.** `gelu_tanh(gate) * up` where Qwen has `silu(gate) * up`. The
  op set is identical (mul, add, sigmoid), so it reuses Qwen's
  `transform_swiglu_f32in_aie2p.mlir` **unchanged** -- the f32-in/bf16-out
  promote sequence this needs is the one that script already selects. Verified
  lowering and numerics before this file existed.
* **Scope.** Qwen folds the residual adds and both RMSNorms into the same
  dispatch. This chain stops at `down`, leaving the norms and the residual on
  the caller. The weights are the win; folding the norms is a clean follow-on
  that Qwen shows how to do (gamma row-scaled into `Bg`/`Bu` so the in-chain
  norm stays bare and reuses the standalone script).

Why the gate/up split is free here
----------------------------------
`load_weights` concatenates `gate` and `up` into one `[D, 2*inter]` tensor so
the two projections issue as a single GEMM, which is right when every GEMM is
its own dispatch. Inside a chain it is not: the whole chain is one dispatch
whatever it contains, so splitting the weight back into two costs no dispatch
and no bytes, and it keeps the merge kernel reading two contiguous buffers.

That matters because the alternative does not work. A chain forbids host work
between ops, so a single `[M, 2*inter]` GEMM output would need a merge kernel
reading both column halves at a row stride; every spelling of that failed --
2D under the elementwise script leaves the output in L2, 2D under the swiglu
script hangs the device, and a 1D linear-index form (`row = offs // INTER`)
fails in aircc.

The contraction, and why it is split
------------------------------------
`_mm_kernel` contracts the whole of K in one tile, which asks two things of K
at once: `tl.arange` wants a power of two, and `rows * K` has to stay under
Triton's tensor cap. `down` contracts the FFN width, so both used to bind on it
-- 12288 padded to 16384, and then a row tile of 256 where the rest of the
chain runs 512, which is half the array.

Cutting the contraction into equal power-of-two pieces answers both without
padding anything, whenever the width has a large enough power of two in it:
12288 is 3 x 4096 and 6144 is 3 x 2048. Each piece is its own GEMM over a slice
of `H`'s columns and `Bd`'s rows, and the pieces are added back. `_plan_down`
picks the piece, and falls back to padding the width where no such factor
exists.

`K_pad` is the same constraint on `gate`/`up`, and stays: D is 1536, which has
no factor above 512, so splitting it would cost more ops than the padding
costs arithmetic.

Tiling
------
`kernels` sets the tile sizes and explains how they become herd shapes. What
matters here is that the chain's row tile is a choice per call, not a constant:
see `ROW_TIERS` there and `run` below.
"""

from __future__ import annotations

import math
import os
import re

import numpy as np
import torch
import triton
import triton.language as tl

# The tile width, the row tiers and Triton's tensor cap come from `kernels`.
# The chain runs the same GEMM kernel on the same schedule as every standalone
# projection, so a second opinion here would be a second opinion about the
# hardware.
from kernels import (
    FUSED_BLOCK,
    MAX_BLOCK_NUMEL,
    MAX_TILE_NUMEL,
    WIDE_M,
    _GELU_2C,
    _GELU_K,
    _npu_driver,
    col_tier,
    elem_block,
    fused_epilogue_script,
    matmul_script,
    row_tier,
    script,
    unaliased_stride,
)

#: Qwen's, unmodified. See the module docstring.
MERGE_SCRIPT = "qwen2_5/transform_swiglu_f32in_aie2p.mlir"
#: ...and the one for when the gate GEMM already applied the activation, so the
#: merge is a bare multiply over a bf16 gate and an f32 up. Same script but for
#: the promote sequence, which pads each operand with its own element type.
MERGE_SCRIPT_BF16_GATE = "llm_q4nx/transform_mul_bf16f32in_aie2p.mlir"
#: The f32 variant: these operands are produced on-device by the GEMMs before
#: it and never touch the host, so they are still f32 rather than bf16.
#: It splits the range over four of npu2's eight columns. Widening it to eight
#: was tried and is slower at this size -- the extra L2 split costs more than
#: the idle columns do.
ADD_SCRIPT = "gpt2/transform_add_f32_aie2p.mlir"

#: How many pieces `down`'s contraction may be cut into before padding the FFN
#: width is the better deal. Each piece is a GEMM op and all but the first also
#: an add, so the chain grows by two ops per piece -- cheap next to the
#: arithmetic a padded width would add, but not without limit.
MAX_DOWN_SPLIT = 4

#: The K a `down` capture is taken at. Any legal one will do -- the extent is
#: restated afterwards -- and this is small enough that `block_m * K` clears
#: Triton's tensor cap at every row tier.
_DOWN_CAPTURE_K = 1024


def _host(w):
    """The numpy view of a weight, whether it is a shared buffer or an array."""
    return w.numpy() if hasattr(w, "numpy") else w


def _bo(w):
    """The XRT buffer object to bind, or None when the weight is not shared."""
    return getattr(w, "bo", None)


def _pow2(n):
    return 1 << (n - 1).bit_length()


def _plan_down(inter):
    """`(HID, chunk)` -- the width the merge writes, and one `down` K tile.

    Both are now the FFN width itself. `down` contracts that width, and the
    two rules that used to forbid saying so -- `tl.arange` wants a power of
    two, `rows * K` must clear Triton's tensor cap -- are the FRONTEND's, and
    `down` no longer goes through it: its op is given to the chain as MLIR.
    So no padding, no split into power-of-two pieces, and none of the adds
    that folded them back.

    Kept as a function because the split is what the buffer layout and
    `_down_slots` are written against, and one width per model is still worth
    naming.
    """
    return inter, inter


def _exact_k_ttshared(chain, rows, k, n, block_m, block_n, a_stride, script,
                      k_capture=None, kernel=None, tail=(), c_dtype=None):
    """`_mm_kernel`'s ttsharedir, restated to contract exactly `k`.

    Two frontend rules stop a kernel from asking for the reduction a model
    actually has: `tl.arange` wants a power of two, and `block_m * K` has to
    stay under Triton's tensor cap. `linalg.matmul` has neither, and the
    schedules tile K by `l2_k`, so the capture only has to be LEGAL -- its K
    need not be the one wanted, above or below. `k_capture` names a legal one
    where the real K is too wide to ask for (`down` contracts 12288, and
    512x16384 is twice the cap).

    Both row strides are kernel constexprs, so a capture taken at the real
    stride has every offset right and the K extent is the only edit.

    Measured on the wide gate at M=1024, N=12288: 8.76 -> 6.54 ms, TFLOP/s
    unchanged at 5.9 -- the same efficiency over 25% fewer bytes.
    """
    kp = k_capture if k_capture is not None else _pow2(k)
    tA = torch.zeros((rows, a_stride), dtype=torch.bfloat16)
    tB = torch.zeros((kp, n), dtype=torch.bfloat16)
    tC = torch.zeros((rows, n), dtype=c_dtype or torch.float32)
    grid = (rows // block_m, n // block_n)
    src = chain._capture_ttshared(
        kernel or _mm_kernel, grid,
        (tA, tB, tC, rows, n, kp, a_stride, 1, n, 1, n, 1) + tuple(tail),
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": kp},
    )
    if isinstance(src, bytes):
        src = src.decode()
    if k == kp:
        return src
    for pat, rep in (
        (rf"sizes: \[{block_m}, {kp}\]", f"sizes: [{block_m}, {k}]"),
        (rf"sizes: \[{kp}, {block_n}\]", f"sizes: [{k}, {block_n}]"),
        (rf"\b{block_m}x{kp}xbf16\b", f"{block_m}x{k}xbf16"),
        (rf"\b{kp}x{block_n}xbf16\b", f"{k}x{block_n}xbf16"),
    ):
        src, hits = re.subn(pat, rep, src)
        if not hits:
            raise RuntimeError(f"exact-K rewrite matched nothing: {pat}")
    return src


def _row_tile(rows, k, block_n):
    """Rows one `_mm_kernel` program may take when it contracts `k` in one tile.

    The chain's own `rows` where Triton's tensor cap and the array allow it,
    halved until they do. The op then covers all of `rows` in
    `rows // _row_tile(...)` grid steps, so every op still spans the same rows
    -- only its herd gets smaller.
    """
    cap = min(MAX_TILE_NUMEL // k, MAX_BLOCK_NUMEL // block_n)
    return min(rows, 1 << cap.bit_length() - 1)


def _as_torch_bf16(a):
    """ml_dtypes bfloat16 numpy -> torch bfloat16, reinterpreted not converted.

    `llama_prefill._t`'s bf16 branch, inlined rather than imported: that module
    pulls in an example's `config`, and this one is a library parameterized on
    dimensions -- like `kernels.py`, it imports nothing example-local. Going via
    float32 instead would double every weight in transit for a result that ends
    up bf16 again.
    """
    return torch.from_numpy(np.ascontiguousarray(a).view(np.uint16)).view(
        torch.bfloat16
    )


@triton.jit
def _mm_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sam: tl.constexpr,
    sak: tl.constexpr,
    sbk: tl.constexpr,
    sbn: tl.constexpr,
    scm: tl.constexpr,
    scn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    aoff: tl.constexpr = 0,
    boff: tl.constexpr = 0,
):
    """bf16 x bf16 -> f32, one tile per program. Qwen's `_mm_kernel`.

    Spelled with explicit strides rather than reusing `kernels._matmul_kernel`
    because a chain op's operands are whole padded buffers, so the row stride
    and the logical width come apart.

    `aoff`/`boff` start the contraction part-way into both operands, which is
    what lets several of these share one pair of buffers and each take a slice
    of K. A chain's `arg_map` names a whole buffer and nothing else, so the
    slice has to be the kernel's business; constexpr, so each slice is its own
    compiled program with no address arithmetic at run time.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a = tl.load(A + aoff + offs_m[:, None] * sam + offs_k[None, :] * sak)
    b = tl.load(B + boff + offs_k[:, None] * sbk + offs_n[None, :] * sbn)
    tl.store(C + offs_m[:, None] * scm + offs_n[None, :] * scn, tl.dot(a, b))


@triton.jit
def _mm_gelu_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sam: tl.constexpr,
    sak: tl.constexpr,
    sbk: tl.constexpr,
    sbn: tl.constexpr,
    scm: tl.constexpr,
    scn: tl.constexpr,
    C2: tl.constexpr,
    KC: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """`_mm_kernel` with `gelu_tanh` applied, writing bf16.

    The activation has to be in the same module as the matmul for a transform
    to fuse it, so it is written here rather than left to the merge. Lowered
    with `kernels.fused_epilogue_script`, it runs on the 32 cores the schedule
    already builds to copy the accumulator out, and `Cg` never exists in DDR as
    f32 -- which is both the traffic saved and, since the drain is the only
    consumer, why the output is bf16.

    Same activation as `_geglu_f32in`, minus the `* up` that has no operand
    here; see its docstring for why `C2` is twice sqrt(2/pi).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a = tl.load(A + offs_m[:, None] * sam + offs_k[None, :] * sak)
    b = tl.load(B + offs_k[:, None] * sbk + offs_n[None, :] * sbn)
    g = tl.dot(a, b)
    z = C2 * (g + KC * g * g * g)
    tl.store(
        C + offs_m[:, None] * scm + offs_n[None, :] * scn,
        (g * tl.sigmoid(z)).to(tl.bfloat16),
    )


@triton.jit
def _mul_bf16_f32in(
    G,
    U,
    Y,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """`Y = G * U` where the gate arrives bf16 and already activated.

    What is left of the merge once `_mm_gelu_kernel` has done the activation in
    the gate GEMM's drain. Reading `G` as bf16 rather than f32 is half of this
    pass's DDR traffic on that side.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    g = tl.load(G + offsets[:]).to(tl.float32)
    tl.store(Y + offsets[:], (g * tl.load(U + offsets[:])).to(tl.bfloat16))


@triton.jit
def _add_f32(A, B, C, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """C = A + B over three f32 buffers. Qwen's `_add_kernel`."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(C + offsets[:], tl.load(A + offsets[:]) + tl.load(B + offsets[:]))


@triton.jit
def _geglu_f32in(
    G,
    U,
    Y,
    C2: tl.constexpr,
    K: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """gelu_tanh(gate) * up over two on-device f32 buffers -> bf16.

    Qwen's `_swiglu_f32in` with `kernels._geglu_kernel`'s activation. That
    activation is exact, not an approximation. mlir-air's decode runs gelu_tanh
    in `glu.cc`, so a fast-GELU here would make this prefill and that decode
    disagree about the model.

    Written with tanh and not with the `x * sigmoid(2z)` it is equal to,
    because AIE2P has exactly two vector transcendentals -- `exp2` and `tanh`,
    in `aie2p_nlf_vector.h` -- and `inv` is in the scalar header. A sigmoid's
    reciprocal therefore costs one call per lane: the divide form's inner loop
    comes out with 17 `@llvm.aie2p.inv` against 2 `exp2`, where this one has 2
    `@llvm.aie2p.tanh` and no scalar call at all. Measured at 1024x12288,
    7.18 ms against 9.44, and closer to the reference besides (1.12e-02 mean
    relative against 1.76e-02) because the hardware tanh beats an exp and a
    reciprocal composed.

    `C2` stays twice sqrt(2/pi) so the constant means the same thing on both
    paths; the halving is here.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    g = tl.load(G + offsets[:])
    u = tl.load(U + offsets[:])
    y = (C2 * 0.5) * (g + K * g * g * g)
    t = tl.extra.cuda.libdevice.tanh(y)
    tl.store(Y + offsets[:], (0.5 * g * (1.0 + t) * u).to(tl.bfloat16))


class FusedMLP:
    """`down(gelu_tanh(gate(A)) * up(A))` for one FFN width, as one dispatch.

    One instance per distinct `inter`; Gemma4 needs two (6144 below layer 15,
    12288 at and above). Chains are per (width, row tier), never per layer: a
    chain owns an `hw_context`, and 35 of those exhausts the NPU
    (`DRM_IOCTL_AMDXDNA_CREATE_HWCTX`). Per-layer weights ride the shared chain
    as separate BO sets, selected by `bo_key`, which is the same arrangement
    Qwen and gpt2 use and what upstream mlir-air's llama does.

    Combined-arg layout, which `run` indexes with::

        0 A    bf16 (rows, K_pad)   caller's activation, per dispatch
        1 Bg   bf16 (K_pad, HID)    static
        2 Cg    f32 (rows, HID)     intermediate
        3 Bu   bf16 (K_pad, HID)    static
        4 Cu    f32 (rows, HID)     intermediate
        5 H    bf16 (rows * HID)    intermediate
        6 Bd   bf16 (HID, D_pad)    static
        7 OUT   f32 (rows, D_pad)   output
        8..     f32 (rows, D_pad)   the split contraction's partials and sums

    More than one op, so the single-op chain corruption defect documented on
    `NPUChain` does not apply.

    `rows` is one of `kernels.ROW_TIERS`, chosen per call and built on first
    use. A
    dispatch computes the whole tile whether or not the prompt fills it, so a
    six-token prefill through the array-filling tile does 512 rows of work for
    six; the small tier is what it falls back to.
    """

    def __init__(self, d_model, inter):
        self.D = d_model
        self.H = inter
        # `gate` and `up` contract D itself, not the power of two above it.
        # The frontend cannot spell that (`tl.arange`), so their op arrives as
        # MLIR -- see `_exact_k_ttshared`. `K_pad` stays as the fallback for a
        # D that IS a power of two, where the two are the same number anyway.
        self.K_pad = _pow2(d_model)
        self.K_exact = d_model
        # The activation's row stride, held off a power of two so A's rows do
        # not alias in DRAM -- `kernels.unaliased_stride`. At an unpadded
        # D=1536 it is already not one, so it returns D unchanged.
        self.K_stride = unaliased_stride(self.K_exact)
        self.HID, self.down_chunk = _plan_down(inter)
        # The column tile each GEMM runs at, and `down`'s N. A multiple of its
        # own tile or the grid truncates and silently drops output columns;
        # Gemma4's D=1536 already is, but derive it rather than rely on that.
        self.gu_n, self.dn_n = col_tier(self.HID), col_tier(d_model)
        self.D_pad = math.ceil(d_model / self.dn_n) * self.dn_n
        self._chains = {}  # rows -> NPUChain
        #: rows -> whether that chain's gate GEMM carries its own
        #: activation, which decides `Cg`'s element type.
        self._gate_fused = {}
        self._weights = {}  # layer_idx -> (Bg, Bu, Bd)

    # ---- weights ----
    def prep_weights(self, gate_up, down):
        """Split the concatenated `gate_up`, pad both halves and `down`.

        `gate_up` is `[D, 2*inter]` as `load_weights` concatenated it, so the
        halves are a plain column split -- done once here, never per dispatch.
        `gate`/`up` contract D exactly, so their weights have D rows and there
        is no padding to keep zero. `down` still pads its N to `D_pad`.
        """
        from ml_dtypes import bfloat16

        D, H, K_pad, HID, D_pad = (
            self.D,
            self.H,
            self.K_pad,
            self.HID,
            self.D_pad,
        )
        gu = gate_up.to(torch.float32).cpu().numpy()
        dn = down.to(torch.float32).cpu().numpy()
        Bg, Bu, Bd = (
            self._alloc(self.K_exact, HID),
            self._alloc(self.K_exact, HID),
            self._alloc(HID, D_pad),
        )
        _host(Bg)[:D, :H] = gu[:, :H].astype(bfloat16)
        _host(Bu)[:D, :H] = gu[:, H:].astype(bfloat16)
        _host(Bd)[:H, :D] = dn.astype(bfloat16)
        return Bg, Bu, Bd

    def _alloc(self, rows, cols):
        """A zeroed `[rows, cols]` bf16 weight, shared with the iGPU if possible.

        Shared, because this is the only copy that needs to exist. The NPU
        dispatches on it by naming its BO; a GPU decode reads the same pages
        through `.torch()`. Allocating it privately here is what forced a
        second, device-resident copy to be built for the decode -- and, since
        the originals are dropped once this exists, a third to be reconstructed
        on the way there.

        Falls back to a plain array where the interop is unavailable, which
        costs the sharing and nothing else: the contents and the layout are
        identical either way.
        """
        try:
            from triton.backends.amd_triton_npu import shared

            return shared.zeros(
                rows, cols, dtype=torch.bfloat16, device="xrt:0", share="hip:0"
            )
        except Exception as e:  # noqa: BLE001 -- see the docstring
            if os.environ.get("AMD_TRITON_NPU_DEBUG"):
                print(f"[fused_mlp] weights stay private: {e}", flush=True)
            from ml_dtypes import bfloat16 as _bf16

            return np.zeros((rows, cols), dtype=_bf16)

    def add_layer(self, layer_idx, gate_up, down):
        """Pad and keep this layer's weights. Call once, at load time."""
        self._weights[layer_idx] = self.prep_weights(gate_up, down)

    def logical_weights(self, layer_idx):
        """This layer's `gate_up` and `down` unpadded, on the host.

        For `--compare-cpu`, whose reference instance runs the torch path and
        needs the shapes `load_weights` held. Exact rather than close:
        `prep_weights` only split and zero-padded, both forms are bf16, and the
        reinterpret does not convert.

        A decode wants `device_weights` instead -- this one allocates, and
        undoes padding the GPU does not mind.
        """
        Bg, Bu, Bd = (_host(b) for b in self._weights[layer_idx])
        D, H = self.D, self.H
        gate_up = torch.cat([_as_torch_bf16(Bg[:D, :H]), _as_torch_bf16(Bu[:D, :H])], 1)
        return gate_up.contiguous(), _as_torch_bf16(Bd[:H, :D]).contiguous()

    def device_weights(self, layer_idx):
        """This layer's padded weights as iGPU tensors over the same pages.

        The point of allocating them shared: a GPU decode reads exactly what
        the NPU dispatches on, so the model exists once. They come back as
        `shapes()` describes, which the caller has to account for: `K_pad` rows
        always, and an FFN width that is `inter` unless it had to be padded. A
        GEMV over the padding is bit-identical to one without it, since the
        weights are zero there -- provided the activation is zero out to
        `K_pad`.

        Raises where the weights are not shared; there is no device view to
        give, and silently handing back a host array would be worse.
        """
        w = self._weights[layer_idx]
        if any(not hasattr(b, "torch") for b in w):
            raise RuntimeError(
                "this layer's weights are not in shared pages, so they have no "
                "iGPU view; the decode has to copy them instead"
            )
        return tuple(b.torch() for b in w)

    def shapes(self):
        """`(K, HID, D_pad)` -- what `device_weights` is shaped to.

        The first is `K_exact`: `gate`/`up` contract D itself now, so their
        weights have D rows and not the power of two above it.
        """
        return self.K_exact, self.HID, self.D_pad

    # ---- chain ----
    #: Where the fixed operands sit in the combined-arg list. The split
    #: contraction's partials and its running sums follow, so those indices are
    #: derived rather than named -- see `_down_slots`.
    A_I, BG_I, CG_I, BU_I, CU_I, H_I, BD_I, OUT_I = range(8)

    def _down_slots(self):
        """`(partials, sums)` -- the buffer indices the split `down` needs.

        One partial per piece of the contraction, then the running sums that
        fold them: the last fold writes `OUT_I`, so there are two fewer sums
        than partials. An unsplit `down` writes `OUT_I` directly and needs
        neither.
        """
        n = self.HID // self.down_chunk
        if n == 1:
            return (), ()
        base = self.OUT_I + 1
        return tuple(range(base, base + n)), tuple(range(base + n, base + 2 * n - 2))

    def _get_chain(self, M):
        if M in self._chains:
            return self._chains[M]
        from triton.backends.amd_triton_npu.multilaunch import NPUChain

        K_exact, HID, D_pad = self.K_exact, self.HID, self.D_pad
        chunk = self.down_chunk
        n_down = HID // chunk
        partials, sums = self._down_slots()
        gu_n, dn_n = self.gu_n, self.dn_n
        gu_m, dn_m = _row_tile(M, K_exact, gu_n), _row_tile(M, _DOWN_CAPTURE_K, dn_n)
        # Per op, not per chain: the two can land on different blocks -- `down`
        # contracts a wider K and writes a narrower N -- and a schedule only
        # places on the block it was generated for.
        gu_script = matmul_script(gu_m, gu_n)
        dn_script = matmul_script(dn_m, dn_n)
        # The gate alone may take the drain-fused schedule, which is generated
        # for one block and one only. It runs at a narrower column tile than
        # `up` does, so the two stop being the same GEMM; that is the price of
        # the block the drain fits in.
        gate_m, gate_n = FUSED_BLOCK
        # Off by default, and measured worse: with an exact K the plain gate
        # is cheap enough that the 256x512 block the drain's L1 budget forces
        # costs more than the f32 `Cg` write it saves. FFN 1544 -> 1587 ms at
        # P=2040. The capability is kept because it no longer carries a stale
        # limitation -- the fused gate can state an unpadded K like every other
        # op -- not because it is worth turning on.
        gate_script = (
            fused_epilogue_script(gate_m, gate_n)
            if M % gate_m == 0 and HID % gate_n == 0
            else None
        )

        # Shape-representative placeholders; only shapes and dtypes drive the
        # warmup lowering, never the values.
        tA = torch.zeros((M, self.K_stride), dtype=torch.bfloat16)
        tBg = torch.zeros((K_exact, HID), dtype=torch.bfloat16)
        tC = torch.zeros((M, HID), dtype=torch.float32)
        tCf = torch.zeros(M * HID, dtype=torch.float32)
        # The gate's output, where the drain-fused schedule writes it: bf16,
        # because the drain is its only consumer.
        tCg = torch.zeros((M, HID), dtype=torch.bfloat16)
        tCgf = torch.zeros(M * HID, dtype=torch.bfloat16)
        tH = torch.zeros(M * HID, dtype=torch.bfloat16)
        tHm = torch.zeros((M, HID), dtype=torch.bfloat16)
        tBd = torch.zeros((HID, D_pad), dtype=torch.bfloat16)
        tOut = torch.zeros((M, D_pad), dtype=torch.float32)
        tOutf = torch.zeros(M * D_pad, dtype=torch.float32)

        chain = NPUChain(f"q4nx_mlp_{self.H}_m{M}")
        if gate_script is not None:
            # gate, with its activation in the GEMM's own drain herd
            gate_src = _exact_k_ttshared(
                chain, M, K_exact, HID, gate_m, gate_n, self.K_stride,
                gate_script, kernel=_mm_gelu_kernel,
                tail=(_GELU_2C, _GELU_K), c_dtype=torch.bfloat16,
            )
            chain.add(
                gate_src,
                grid=(M // gate_m, HID // gate_n),
                arg_map={0: self.A_I, 1: self.BG_I, 2: self.CG_I},
                args=(),
                transform_script=gate_script,
            )
            gemms = ((self.BU_I, self.CU_I),)
        else:
            gemms = ((self.BG_I, self.CG_I), (self.BU_I, self.CU_I))
        gu_src = _exact_k_ttshared(
            chain, M, self.K_exact, HID, gu_m, gu_n, self.K_stride, gu_script
        )
        for src, dst in gemms:
            # gate, then up -- the same GEMM against the same activation, so
            # they differ only in which weight they read and where they land,
            # and one module serves both.
            chain.add(
                gu_src,
                grid=(M // gu_m, HID // gu_n),
                arg_map={0: self.A_I, 1: src, 2: dst},
                args=(),
                transform_script=gu_script,
            )
        # merge: H = gelu_tanh(Cg) * Cu, or just the multiply where the gate
        # GEMM's drain already did the activation.
        # gate + up + H. The gate is bf16 where its GEMM's drain applied the
        # activation and f32 where the merge still has to.
        merge_block = elem_block(M * HID, (2 if gate_script is not None else 4) + 4 + 2)
        chain.add(
            _mul_bf16_f32in if gate_script is not None else _geglu_f32in,
            grid=((M * HID) // merge_block,),
            arg_map={0: self.CG_I, 1: self.CU_I, 2: self.H_I},
            args=(
                (tCgf, tCf, tH, M * HID)
                if gate_script is not None
                else (tCf, tCf, tH, _GELU_2C, _GELU_K, M * HID)
            ),
            constexprs={"BLOCK_SIZE": merge_block},
            transform_script=script(
                MERGE_SCRIPT_BF16_GATE if gate_script is not None else MERGE_SCRIPT
            ),
        )
        # down, contracting the FFN width in ONE op. It used to go in
        # `n_down` power-of-two pieces with adds folding them back, because a
        # kernel cannot ask for a K of 12288 -- not a power of two, and
        # `dn_m * 16384` is twice Triton's tensor cap. Stating it in the
        # module costs neither: the pieces were 3 launches and 2 full-width
        # f32 adds against this one, 54 MiB of output traffic against 6.
        dn_src = _exact_k_ttshared(
            chain, M, HID, D_pad, dn_m, dn_n, HID, dn_script,
            k_capture=_DOWN_CAPTURE_K,
        )
        chain.add(
            dn_src,
            grid=(M // dn_m, D_pad // dn_n),
            arg_map={0: self.H_I, 1: self.BD_I, 2: self.OUT_I},
            args=(),
            transform_script=dn_script,
        )
        self._chains[M] = chain
        self._gate_fused[M] = gate_script is not None
        return chain

    # ---- dispatch ----
    def run(self, layer_idx, h):
        """`down(gelu_tanh(gate(h)) * up(h))` for one layer. h: [N, D] -> [N, D].

        Returns f32, matching what `_matmul` returns on the unfused path.

        Rows are tiled at one of `kernels.ROW_TIERS` and dispatched per tile, so
        a handful of ELFs serve every prompt length -- the same bargain
        `triton_matmul` makes, and the reason no tier follows the prompt.
        """
        from ml_dtypes import bfloat16

        D, K_pad, HID, D_pad = self.D, self.K_pad, self.HID, self.D_pad
        Bg, Bu, Bd = self._weights[layer_idx]
        static = {self.BG_I: Bg, self.BU_I: Bu, self.BD_I: Bd}
        bound = {i: _bo(w) for i, w in static.items() if _bo(w) is not None} or None
        partials, sums = self._down_slots()

        h2d = h.reshape(-1, D).to(torch.float32).cpu().numpy()
        N = h2d.shape[0]
        out = np.empty((N, D), dtype=np.float32)
        M = row_tier(N)

        # Scoped per call, not held: `kernels.launch` scopes the driver the same
        # way, and this runs inside a prefill that also issues torch ops. The
        # chain's warmup compilation needs it too, so the scope covers the
        # build, not just the dispatch.
        with _npu_driver():
            chain = self._get_chain(M)
            for m0 in range(0, N, M):
                rows = min(M, N - m0)
                # Zeroed, not empty: the gate/up GEMMs contract over all K_pad
                # columns, so the padding past D has to be 0 rather than
                # whatever the last tile left there.
                A = np.zeros((M, self.K_stride), dtype=bfloat16)
                A[:rows, :D] = h2d[m0 : m0 + rows].astype(bfloat16)
                # Chain intermediates and the output are fully written by their
                # producing kernel, so np.empty avoids a per-call memset.
                args = [None] * (self.OUT_I + 1 + len(partials) + len(sums))
                args[self.A_I] = A
                args[self.BG_I] = _host(Bg)
                args[self.BU_I] = _host(Bu)
                args[self.BD_I] = _host(Bd)
                # The gate's output is bf16 where its drain herd applied the
                # activation, and f32 where the merge still has to.
                args[self.CG_I] = np.empty(
                    (M, HID), dtype=bfloat16 if self._gate_fused[M] else np.float32
                )
                args[self.CU_I] = np.empty((M, HID), dtype=np.float32)
                args[self.H_I] = np.empty(M * HID, dtype=bfloat16)
                for i in (self.OUT_I, *partials, *sums):
                    args[i] = np.empty((M, D_pad), dtype=np.float32)
                # The weights are bound where they are shared, so the chain
                # dispatches on the pages they already occupy instead of
                # staging a BO of its own. `static_indices` still names them:
                # bound or not, they carry no new host data per call.
                got = chain.run(
                    args,
                    bo_key=f"q4nx_mlp_{self.H}_L{layer_idx}",
                    static_indices=set(static),
                    intermediate_indices={self.CG_I, self.CU_I, self.H_I}
                    | set(partials)
                    | set(sums),
                    output_indices={self.OUT_I},
                    bound_buffers=bound,
                )
                out[m0 : m0 + rows] = got[self.OUT_I].astype(np.float32)[:rows, :D]

        return torch.from_numpy(out).reshape(*h.shape[:-1], D)

    def close(self):
        for chain in self._chains.values():
            chain.close()
        self._chains.clear()
        self._weights.clear()
