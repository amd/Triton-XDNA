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

Padding
-------
`down` contracts over `inter`, and `tl.arange` needs a power of two, so the
merge output is `HID_pad = next_pow2(inter)` wide -- 8192 for 6144, 16384 for
12288. The gate/up GEMMs write that full width; their weights are zero past
`inter`, so the tail columns are zeros the merge maps to zeros and `down`
contracts against the matching zero rows of `Bd`. Nothing has to trim anything.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import triton
import triton.language as tl

from kernels import _GELU_2C, _GELU_K, _npu_driver, script

#: Row tile. `ROW_TILE` in kernels.py, repeated rather than imported so the
#: reason travels with it: a kernel is compiled per (grid, constexprs), so a
#: chain whose M tracked the prompt length would rebuild the whole ELF on the
#: first longer prompt -- measured at 231 s for the step from 128 to 256 rows.
#: Fixed M, and `run` loops over row tiles instead.
BLOCK_M = 128
BLOCK_N = 256

#: Elementwise tile for the merge, as every other flat kernel here uses.
MERGE_BLOCK = 1024

MATMUL_SCRIPT = "gpt2/transform_matmul_aie2p.mlir"
#: Qwen's, unmodified. See the module docstring.
MERGE_SCRIPT = "qwen2_5/transform_swiglu_f32in_aie2p.mlir"


def _pow2(n):
    return 1 << (n - 1).bit_length()


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
):
    """bf16 x bf16 -> f32, one tile per program. Qwen's `_mm_kernel`.

    Spelled with explicit strides rather than reusing `kernels._matmul_kernel`
    because a chain op's operands are whole padded buffers, so the row stride
    and the logical width come apart.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a = tl.load(A + offs_m[:, None] * sam + offs_k[None, :] * sak)
    b = tl.load(B + offs_k[:, None] * sbk + offs_n[None, :] * sbn)
    tl.store(C + offs_m[:, None] * scm + offs_n[None, :] * scn, tl.dot(a, b))


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
    12288 at and above). One chain each -- and one chain per *width*, not per
    layer: a chain owns an `hw_context`, and 35 of those exhausts the NPU
    (`DRM_IOCTL_AMDXDNA_CREATE_HWCTX`). Per-layer weights ride the shared chain
    as separate BO sets, selected by `bo_key`, which is the same arrangement
    Qwen and gpt2 use and what upstream mlir-air's llama does.

    Combined-arg layout, which `run` indexes with::

        0 A    bf16 (BLOCK_M, K_pad)      caller's activation, per dispatch
        1 Bg   bf16 (K_pad,  HID_pad)     static
        2 Cg    f32 (BLOCK_M, HID_pad)    intermediate
        3 Bu   bf16 (K_pad,  HID_pad)     static
        4 Cu    f32 (BLOCK_M, HID_pad)    intermediate
        5 H    bf16 (BLOCK_M * HID_pad)   intermediate
        6 Bd   bf16 (HID_pad, D_pad)      static
        7 OUT   f32 (BLOCK_M, D_pad)      output

    Four ops, so the single-op chain corruption defect documented on `NPUChain`
    does not apply.
    """

    def __init__(self, d_model, inter):
        self.D = d_model
        self.H = inter
        self.K_pad = _pow2(d_model)
        self.HID_pad = _pow2(inter)
        # `down`'s N. A multiple of BLOCK_N or the grid truncates and silently
        # drops output columns; Gemma4's D=1536 already is, but derive it rather
        # than rely on that.
        self.D_pad = math.ceil(d_model / BLOCK_N) * BLOCK_N
        self._chain = None
        self._weights = {}  # layer_idx -> (Bg, Bu, Bd)

    # ---- weights ----
    def prep_weights(self, gate_up, down):
        """Split the concatenated `gate_up`, pad both halves and `down`.

        `gate_up` is `[D, 2*inter]` as `load_weights` concatenated it, so the
        halves are a plain column split -- done once here, never per dispatch.
        Everything past the logical extent stays zero: that is what makes the
        `HID_pad` tail harmless end to end (see the module docstring).
        """
        from ml_dtypes import bfloat16

        D, H, K_pad, HID_pad, D_pad = (
            self.D,
            self.H,
            self.K_pad,
            self.HID_pad,
            self.D_pad,
        )
        gu = gate_up.to(torch.float32).cpu().numpy()
        dn = down.to(torch.float32).cpu().numpy()
        Bg = np.zeros((K_pad, HID_pad), dtype=bfloat16)
        Bg[:D, :H] = gu[:, :H].astype(bfloat16)
        Bu = np.zeros((K_pad, HID_pad), dtype=bfloat16)
        Bu[:D, :H] = gu[:, H:].astype(bfloat16)
        Bd = np.zeros((HID_pad, D_pad), dtype=bfloat16)
        Bd[:H, :D] = dn.astype(bfloat16)
        return Bg, Bu, Bd

    def add_layer(self, layer_idx, gate_up, down):
        """Pad and keep this layer's weights. Call once, at load time."""
        self._weights[layer_idx] = self.prep_weights(gate_up, down)

    # ---- chain ----
    def _get_chain(self):
        if self._chain is not None:
            return self._chain
        from triton.backends.amd_triton_npu.multilaunch import NPUChain

        K_pad, HID_pad, D_pad = self.K_pad, self.HID_pad, self.D_pad
        M = BLOCK_M

        # Shape-representative placeholders; only shapes and dtypes drive the
        # warmup lowering, never the values.
        tA = torch.zeros((M, K_pad), dtype=torch.bfloat16)
        tBg = torch.zeros((K_pad, HID_pad), dtype=torch.bfloat16)
        tCg = torch.zeros((M, HID_pad), dtype=torch.float32)
        tBu = torch.zeros((K_pad, HID_pad), dtype=torch.bfloat16)
        tCu = torch.zeros((M, HID_pad), dtype=torch.float32)
        tCgf = torch.zeros(M * HID_pad, dtype=torch.float32)
        tCuf = torch.zeros(M * HID_pad, dtype=torch.float32)
        tH = torch.zeros(M * HID_pad, dtype=torch.bfloat16)
        tHm = torch.zeros((M, HID_pad), dtype=torch.bfloat16)
        tBd = torch.zeros((HID_pad, D_pad), dtype=torch.bfloat16)
        tOut = torch.zeros((M, D_pad), dtype=torch.float32)

        mm_script = script(MATMUL_SCRIPT)
        chain = NPUChain(f"q4nx_mlp_{self.H}")
        # op0 gate: Cg(2) = A(0) @ Bg(1)
        chain.add(
            _mm_kernel,
            grid=(M // BLOCK_M, HID_pad // BLOCK_N),
            arg_map={0: 0, 1: 1, 2: 2},
            args=(tA, tBg, tCg, M, HID_pad, K_pad, K_pad, 1, HID_pad, 1, HID_pad, 1),
            constexprs={
                "BLOCK_SIZE_M": BLOCK_M,
                "BLOCK_SIZE_N": BLOCK_N,
                "BLOCK_SIZE_K": K_pad,
            },
            transform_script=mm_script,
        )
        # op1 up: Cu(4) = A(0) @ Bu(3) -- shares the input buffer with op0.
        chain.add(
            _mm_kernel,
            grid=(M // BLOCK_M, HID_pad // BLOCK_N),
            arg_map={0: 0, 1: 3, 2: 4},
            args=(tA, tBu, tCu, M, HID_pad, K_pad, K_pad, 1, HID_pad, 1, HID_pad, 1),
            constexprs={
                "BLOCK_SIZE_M": BLOCK_M,
                "BLOCK_SIZE_N": BLOCK_N,
                "BLOCK_SIZE_K": K_pad,
            },
            transform_script=mm_script,
        )
        # op2 merge: H(5) = gelu_tanh(Cg(2)) * Cu(4)
        chain.add(
            _geglu_f32in,
            grid=((M * HID_pad) // MERGE_BLOCK,),
            arg_map={0: 2, 1: 4, 2: 5},
            args=(tCgf, tCuf, tH, _GELU_2C, _GELU_K, M * HID_pad),
            constexprs={"BLOCK_SIZE": MERGE_BLOCK},
            transform_script=script(MERGE_SCRIPT),
        )
        # op3 down: OUT(7) = H(5) @ Bd(6)
        chain.add(
            _mm_kernel,
            grid=(M // BLOCK_M, D_pad // BLOCK_N),
            arg_map={0: 5, 1: 6, 2: 7},
            args=(tHm, tBd, tOut, M, D_pad, HID_pad, HID_pad, 1, D_pad, 1, D_pad, 1),
            constexprs={
                "BLOCK_SIZE_M": BLOCK_M,
                "BLOCK_SIZE_N": BLOCK_N,
                "BLOCK_SIZE_K": HID_pad,
            },
            transform_script=mm_script,
        )
        self._chain = chain
        return chain

    # ---- dispatch ----
    def run(self, layer_idx, h):
        """`down(gelu_tanh(gate(h)) * up(h))` for one layer. h: [N, D] -> [N, D].

        Returns f32, matching what `_matmul` returns on the unfused path.

        Rows are tiled at `BLOCK_M` and dispatched per tile, so one ELF serves
        every prompt length -- the same bargain `triton_matmul` makes, and the
        reason `BLOCK_M` is a constant here.
        """
        from ml_dtypes import bfloat16

        D, K_pad, HID_pad, D_pad = self.D, self.K_pad, self.HID_pad, self.D_pad
        Bg, Bu, Bd = self._weights[layer_idx]

        h2d = h.reshape(-1, D).to(torch.float32).cpu().numpy()
        N = h2d.shape[0]
        out = np.empty((N, D), dtype=np.float32)

        # Scoped per call, not held: `kernels.launch` scopes the driver the same
        # way, and this runs inside a prefill that also issues torch ops. The
        # chain's warmup compilation needs it too, so the scope covers the
        # build, not just the dispatch.
        with _npu_driver():
            chain = self._get_chain()
            for m0 in range(0, N, BLOCK_M):
                rows = min(BLOCK_M, N - m0)
                # Zeroed, not empty: the gate/up GEMMs contract over all K_pad
                # columns, so the padding past D has to be 0 rather than
                # whatever the last tile left there.
                A = np.zeros((BLOCK_M, K_pad), dtype=bfloat16)
                A[:rows, :D] = h2d[m0 : m0 + rows].astype(bfloat16)
                # Chain intermediates and the output are fully written by their
                # producing kernel, so np.empty avoids a per-call memset.
                Cg = np.empty((BLOCK_M, HID_pad), dtype=np.float32)
                Cu = np.empty((BLOCK_M, HID_pad), dtype=np.float32)
                H = np.empty(BLOCK_M * HID_pad, dtype=bfloat16)
                OUT = np.empty((BLOCK_M, D_pad), dtype=np.float32)
                got = chain.run(
                    [A, Bg, Cg, Bu, Cu, H, Bd, OUT],
                    bo_key=f"q4nx_mlp_{self.H}_L{layer_idx}",
                    static_indices={1, 3, 6},
                    intermediate_indices={2, 4, 5},
                    output_indices={7},
                )
                out[m0 : m0 + rows] = got[7].astype(np.float32)[:rows, :D]

        return torch.from_numpy(out).reshape(*h.shape[:-1], D)

    def close(self):
        if self._chain is not None:
            self._chain.close()
            self._chain = None
        self._weights.clear()
