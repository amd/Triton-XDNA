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

import numpy as np
import torch
import triton
import triton.language as tl

# The tile width, the row tiers and Triton's tensor cap come from `kernels`.
# The chain runs the same GEMM kernel on the same schedule as every standalone
# projection, so a second opinion here would be a second opinion about the
# hardware.
from kernels import (
    BLOCK_N,
    MAX_TILE_NUMEL,
    WIDE_M,
    _GELU_2C,
    _GELU_K,
    _npu_driver,
    row_tier,
    script,
)

#: Elementwise tile for the merge, as every other flat kernel here uses.
MERGE_BLOCK = 1024

MATMUL_SCRIPT = "gpt2/transform_matmul_aie2p.mlir"
#: Qwen's, unmodified. See the module docstring.
MERGE_SCRIPT = "qwen2_5/transform_swiglu_f32in_aie2p.mlir"
#: The f32 variant: these operands are produced on-device by the GEMMs before
#: it and never touch the host, so they are still f32 rather than bf16.
#: It splits the range over four of npu2's eight columns. Widening it to eight
#: was tried and is slower at this size -- the extra L2 split costs more than
#: the idle columns do.
ADD_SCRIPT = "gpt2/transform_add_f32_aie2p.mlir"

#: Elementwise tile for the fold, as `MERGE_BLOCK` is for the merge.
ADD_BLOCK = 1024

#: How many pieces `down`'s contraction may be cut into before padding the FFN
#: width is the better deal. Each piece is a GEMM op and all but the first also
#: an add, so the chain grows by two ops per piece -- cheap next to the
#: arithmetic a padded width would add, but not without limit.
MAX_DOWN_SPLIT = 4


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

    `down` contracts the FFN width, and `_mm_kernel` does a whole contraction
    in one tile: `tl.arange` wants a power of two and `rows * K` has to stay
    under Triton's cap. Contracting in equal power-of-two pieces satisfies both
    WITHOUT padding anything, whenever the width has a large enough power of
    two in it -- 12288 is 3 x 4096 and 6144 is 3 x 2048, so Gemma4 pays no
    padding at either width and every piece still gets a full-width herd.

    A width with no such factor falls back to padding it to a power of two, as
    this did throughout before: correct at any width, and the pieces are then
    trivially powers of two as well.
    """
    cap = 1 << (MAX_TILE_NUMEL // WIDE_M).bit_length() - 1
    chunk = inter & -inter  # the largest power of two dividing it
    if min(chunk, cap) * MAX_DOWN_SPLIT >= inter:
        return inter, min(chunk, cap)
    return _pow2(inter), min(_pow2(inter), cap)


def _row_tile(rows, k):
    """Rows one `_mm_kernel` program may take when it contracts `k` in one tile.

    The chain's own `rows` where Triton's tensor cap allows it, halved until it
    does. The op then covers all of `rows` in `rows // _row_tile(rows, k)` grid
    steps, so every op still spans the same rows -- only its herd gets smaller.
    """
    return min(rows, 1 << (MAX_TILE_NUMEL // k).bit_length() - 1)


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
    activation is exact, not an approximation: `tanh(z) = 2*sigmoid(2z) - 1`
    turns gelu_pytorch_tanh into `x * sigmoid(2z)`, which is why `C2` is twice
    sqrt(2/pi). mlir-air's decode runs gelu_tanh in `glu.cc`, so a fast-GELU
    here would make this prefill and that decode disagree about the model.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    g = tl.load(G + offsets[:])
    u = tl.load(U + offsets[:])
    z = C2 * (g + K * g * g * g)
    tl.store(Y + offsets[:], (g * tl.sigmoid(z) * u).to(tl.bfloat16))


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
        self.K_pad = _pow2(d_model)
        self.HID, self.down_chunk = _plan_down(inter)
        # `down`'s N. A multiple of BLOCK_N or the grid truncates and silently
        # drops output columns; Gemma4's D=1536 already is, but derive it rather
        # than rely on that.
        self.D_pad = math.ceil(d_model / BLOCK_N) * BLOCK_N
        self._chains = {}  # rows -> NPUChain
        self._weights = {}  # layer_idx -> (Bg, Bu, Bd)

    # ---- weights ----
    def prep_weights(self, gate_up, down):
        """Split the concatenated `gate_up`, pad both halves and `down`.

        `gate_up` is `[D, 2*inter]` as `load_weights` concatenated it, so the
        halves are a plain column split -- done once here, never per dispatch.
        The rows past `D` stay zero, which is what makes `K_pad` harmless: the
        gate/up GEMMs contract them against the zeros `run` leaves in `A`.
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
            self._alloc(K_pad, HID),
            self._alloc(K_pad, HID),
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
        """`(K_pad, HID, D_pad)` -- what `device_weights` is shaped to."""
        return self.K_pad, self.HID, self.D_pad

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

        K_pad, HID, D_pad = self.K_pad, self.HID, self.D_pad
        chunk = self.down_chunk
        n_down = HID // chunk
        partials, sums = self._down_slots()
        gu_m, dn_m = _row_tile(M, K_pad), _row_tile(M, chunk)

        # Shape-representative placeholders; only shapes and dtypes drive the
        # warmup lowering, never the values.
        tA = torch.zeros((M, K_pad), dtype=torch.bfloat16)
        tBg = torch.zeros((K_pad, HID), dtype=torch.bfloat16)
        tC = torch.zeros((M, HID), dtype=torch.float32)
        tCf = torch.zeros(M * HID, dtype=torch.float32)
        tH = torch.zeros(M * HID, dtype=torch.bfloat16)
        tHm = torch.zeros((M, HID), dtype=torch.bfloat16)
        tBd = torch.zeros((HID, D_pad), dtype=torch.bfloat16)
        tOut = torch.zeros((M, D_pad), dtype=torch.float32)
        tOutf = torch.zeros(M * D_pad, dtype=torch.float32)

        mm_script = script(MATMUL_SCRIPT)
        chain = NPUChain(f"q4nx_mlp_{self.H}_m{M}")
        for src, dst in ((self.BG_I, self.CG_I), (self.BU_I, self.CU_I)):
            # gate, then up -- the same GEMM against the same activation, so
            # they differ only in which weight they read and where they land.
            chain.add(
                _mm_kernel,
                grid=(M // gu_m, HID // BLOCK_N),
                arg_map={0: self.A_I, 1: src, 2: dst},
                args=(tA, tBg, tC, M, HID, K_pad, K_pad, 1, HID, 1, HID, 1),
                constexprs={
                    "BLOCK_SIZE_M": gu_m,
                    "BLOCK_SIZE_N": BLOCK_N,
                    "BLOCK_SIZE_K": K_pad,
                },
                transform_script=mm_script,
            )
        # merge: H = gelu_tanh(Cg) * Cu
        chain.add(
            _geglu_f32in,
            grid=((M * HID) // MERGE_BLOCK,),
            arg_map={0: self.CG_I, 1: self.CU_I, 2: self.H_I},
            args=(tCf, tCf, tH, _GELU_2C, _GELU_K, M * HID),
            constexprs={"BLOCK_SIZE": MERGE_BLOCK},
            transform_script=script(MERGE_SCRIPT),
        )
        # down, in `n_down` pieces of the contraction. Each takes its own slice
        # of H's columns and of Bd's rows through the kernel's constant
        # offsets, because a chain's arg_map names whole buffers.
        for i in range(n_down):
            chain.add(
                _mm_kernel,
                grid=(M // dn_m, D_pad // BLOCK_N),
                arg_map={
                    0: self.H_I,
                    1: self.BD_I,
                    2: partials[i] if partials else self.OUT_I,
                },
                args=(tHm, tBd, tOut, M, D_pad, chunk, HID, 1, D_pad, 1, D_pad, 1),
                constexprs={
                    "BLOCK_SIZE_M": dn_m,
                    "BLOCK_SIZE_N": BLOCK_N,
                    "BLOCK_SIZE_K": chunk,
                    "aoff": i * chunk,
                    "boff": i * chunk * D_pad,
                },
                transform_script=mm_script,
            )
        # and fold the pieces back together.
        acc = partials[0] if partials else None
        for i in range(1, n_down):
            dst = self.OUT_I if i == n_down - 1 else sums[i - 1]
            chain.add(
                _add_f32,
                grid=((M * D_pad) // ADD_BLOCK,),
                arg_map={0: acc, 1: partials[i], 2: dst},
                args=(tOutf, tOutf, tOutf, M * D_pad),
                constexprs={"BLOCK_SIZE": ADD_BLOCK},
                transform_script=script(ADD_SCRIPT),
            )
            acc = dst
        self._chains[M] = chain
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
                A = np.zeros((M, K_pad), dtype=bfloat16)
                A[:rows, :D] = h2d[m0 : m0 + rows].astype(bfloat16)
                # Chain intermediates and the output are fully written by their
                # producing kernel, so np.empty avoids a per-call memset.
                args = [None] * (self.OUT_I + 1 + len(partials) + len(sums))
                args[self.A_I] = A
                args[self.BG_I] = _host(Bg)
                args[self.BU_I] = _host(Bu)
                args[self.BD_I] = _host(Bd)
                args[self.CG_I] = np.empty((M, HID), dtype=np.float32)
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
