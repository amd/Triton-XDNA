# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4's FFN with gate, up and the GELU merge as one GEMM.

`FusedMLP` runs four ops per row tile: the gate and up GEMMs, each writing an
f32 plane, a merge that reads both planes back to write `H`, and `down`. Here
gate and up are one GEMM over their weights interleaved column by column
(gate column j, then up column j). Its epilogue cores split each accumulator
pair and write `gelu_tanh(g) * u` as bf16 `H`, so the two f32 planes and the
merge op go away.

The schedule is `transform_matmul_gate_up_gelu_aie2p.mlir`: the 64x128 matmul
schedule with an epilogue herd, a K step of 32 and no ping-pong on the K loop,
which leaves L1 room for the epilogue's tile next to the accumulator.

`run`, `add_layer` and the weight accessors keep `FusedMLP`'s contracts. The
weights are `Bgu [K, 2*HID]` (interleaved) and `Bd [HID, D_pad]`, so
`device_weights` returns two tensors, not three.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

import fused_mlp as fm
from kernels import (
    DEEP_L2_K,
    L2_K,
    _GELU_2C,
    _GELU_K,
    _npu_driver,
    matmul_script,
    narrow_k,
    row_tier,
    script,
)

GU_SCRIPT = "llm_q4nx/transform_matmul_gate_up_gelu_aie2p.mlir"
#: The fused op's column tile: four 128-column core tiles.
GU_N = 512
#: Rows one program takes at most: 64-row core tiles over the array's 8 columns.
GU_MAX_M = 64 * 8


@triton.jit
def _mm_gate_up_gelu(
    A,
    B,
    H,
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
    C2: tl.constexpr,
    KG: tl.constexpr,
):
    """H = gelu_tanh(A @ Bg) * (A @ Bu) for one tile; B interleaves gate/up columns."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a = tl.load(A + offs_m[:, None] * sam + offs_k[None, :] * sak)
    b = tl.load(B + offs_k[:, None] * sbk + offs_n[None, :] * sbn)
    g, u = tl.split(tl.reshape(tl.dot(a, b), (BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)))
    z = C2 * (g + KG * g * g * g)
    h = (g * tl.sigmoid(z) * u).to(tl.bfloat16)
    offs_h = pid_n * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
    tl.store(H + offs_m[:, None] * scm + offs_h[None, :] * scn, h)


class FusedMLPGU(fm.FusedMLP):
    """`down(gelu_tanh(gate(A)) * up(A))` as two ops: the fused gate|up|GELU and `down`.

    Combined-arg layout::

        0 A    bf16 (rows, K_stride)  caller's activation, per dispatch
        1 Bgu  bf16 (K_exact, 2*HID)  static, gate and up columns interleaved
        2 H    bf16 (rows * HID)      intermediate
        3 Bd   bf16 (HID, D_pad)      static
        4 OUT   f32 (rows, D_pad)     output
    """

    A_I, BGU_I, H_I, BD_I, OUT_I = range(5)

    def __init__(self, d_model, inter):
        super().__init__(d_model, inter)
        if (2 * self.HID) % GU_N:
            raise ValueError(
                f"FFN width {self.HID} is not a multiple of the fused op's "
                f"{GU_N // 2}-column output tile"
            )

    # ---- weights ----
    def prep_weights(self, gate_up, down):
        """`gate_up` [D, 2*inter] (gate columns, then up) -> interleaved `Bgu`."""
        from ml_dtypes import bfloat16

        D, H, HID, D_pad = self.D, self.H, self.HID, self.D_pad
        gu = gate_up.to(torch.float32).cpu().numpy()
        dn = down.to(torch.float32).cpu().numpy()
        Bgu, Bd = self._alloc(self.K_exact, 2 * HID), self._alloc(HID, D_pad)
        b = fm._host(Bgu).reshape(self.K_exact, HID, 2)
        b[:D, :H, 0] = gu[:, :H].astype(bfloat16)
        b[:D, :H, 1] = gu[:, H:].astype(bfloat16)
        fm._host(Bd)[:H, :D] = dn.astype(bfloat16)
        return Bgu, Bd

    def logical_weights(self, layer_idx):
        """`gate_up` and `down` as `load_weights` held them, de-interleaved."""
        Bgu, Bd = (fm._host(b) for b in self._weights[layer_idx])
        D, H, HID = self.D, self.H, self.HID
        b = Bgu.reshape(self.K_exact, HID, 2)
        gate = fm._as_torch_bf16(np.ascontiguousarray(b[:D, :H, 0]))
        up = fm._as_torch_bf16(np.ascontiguousarray(b[:D, :H, 1]))
        gate_up = torch.cat([gate, up], 1)
        return gate_up.contiguous(), fm._as_torch_bf16(Bd[:H, :D]).contiguous()

    # ---- chain ----
    def _get_chain(self, M):
        if M in self._chains:
            return self._chains[M]
        from triton.backends.amd_triton_npu.multilaunch import NPUChain

        K, HID, D_pad = self.K_exact, self.HID, self.D_pad
        N2 = 2 * HID
        gu_m = min(fm._row_tile(M, K, GU_N), GU_MAX_M)
        dn_n = self.dn_n
        dn_m = fm._row_tile(M, fm._DOWN_CAPTURE_K, dn_n)
        dn_script = matmul_script(
            dn_m, dn_n, l2_k=DEEP_L2_K if HID % DEEP_L2_K == 0 else L2_K
        )
        chain = NPUChain(f"q4nx_mlpgu_{self.H}_m{M}")
        # Captured at the power of two above K, which `tl.arange` needs, and
        # restated to contract K itself.
        kp = fm._pow2(K)
        src = chain._capture_ttshared(
            _mm_gate_up_gelu,
            (M // gu_m, N2 // GU_N),
            (
                torch.zeros((M, self.K_stride), dtype=torch.bfloat16),
                torch.zeros((kp, N2), dtype=torch.bfloat16),
                torch.zeros((M, HID), dtype=torch.bfloat16),
                M,
                N2,
                kp,
                self.K_stride,
                1,
                N2,
                1,
                HID,
                1,
            ),
            {
                "BLOCK_SIZE_M": gu_m,
                "BLOCK_SIZE_N": GU_N,
                "BLOCK_SIZE_K": kp,
                "C2": _GELU_2C,
                "KG": _GELU_K,
            },
        )
        if isinstance(src, bytes):
            src = src.decode()
        if kp != K:
            src = narrow_k(src, K, kp, gu_m, GU_N)
        chain.add(
            src,
            grid=(M // gu_m, N2 // GU_N),
            arg_map={0: self.A_I, 1: self.BGU_I, 2: self.H_I},
            args=(),
            transform_script=script(GU_SCRIPT),
        )
        dn_src = fm._exact_k_ttshared(
            chain,
            M,
            HID,
            D_pad,
            dn_m,
            dn_n,
            HID,
            dn_script,
            k_capture=fm._DOWN_CAPTURE_K,
        )
        chain.add(
            dn_src,
            grid=(M // dn_m, D_pad // dn_n),
            arg_map={0: self.H_I, 1: self.BD_I, 2: self.OUT_I},
            args=(),
            transform_script=dn_script,
        )
        self._chains[M] = chain
        return chain

    # ---- dispatch ----
    def run(self, layer_idx, h):
        """`down(gelu_tanh(gate(h)) * up(h))` for one layer. h: [N, D] -> [N, D] f32."""
        from ml_dtypes import bfloat16

        D, HID, D_pad = self.D, self.HID, self.D_pad
        Bgu, Bd = self._weights[layer_idx]
        static = {self.BGU_I: Bgu, self.BD_I: Bd}
        bound = {i: fm._bo(w) for i, w in static.items() if fm._bo(w) is not None}

        h2d = h.reshape(-1, D)
        N = h2d.shape[0]
        out = np.empty((N, D), dtype=np.float32)
        M = row_tier(N)
        A_pg, OUT_pg = self._io_pages(M)

        with _npu_driver():
            chain = self._get_chain(M)
            for m0 in range(0, N, M):
                rows = min(M, N - m0)
                if A_pg is not None:
                    A_pg.torch()[:rows, :D] = h2d[m0 : m0 + rows].to(torch.bfloat16)
                    A = A_pg.numpy()
                else:
                    A = np.zeros((M, self.K_stride), dtype=bfloat16)
                    A[:rows, :D] = (
                        h2d[m0 : m0 + rows].to(torch.float32).cpu().numpy()
                    ).astype(bfloat16)
                args = [None] * (self.OUT_I + 1)
                args[self.A_I] = A
                args[self.BGU_I] = fm._host(Bgu)
                args[self.BD_I] = fm._host(Bd)
                args[self.H_I] = np.empty(M * HID, dtype=bfloat16)
                args[self.OUT_I] = (
                    OUT_pg.numpy()
                    if OUT_pg is not None
                    else np.empty((M, D_pad), dtype=np.float32)
                )
                io_bound = dict(bound)
                if A_pg is not None:
                    io_bound[self.A_I] = A_pg.bo
                if OUT_pg is not None:
                    io_bound[self.OUT_I] = OUT_pg.bo
                got = chain.run(
                    args,
                    bo_key=f"q4nx_mlpgu_{self.H}_L{layer_idx}",
                    static_indices=set(static),
                    intermediate_indices={self.H_I, self.OUT_I},
                    output_indices={self.OUT_I},
                    bound_buffers=io_bound or None,
                )
                res = OUT_pg.numpy() if OUT_pg is not None else got[self.OUT_I]
                out[m0 : m0 + rows] = np.asarray(res, np.float32)[:rows, :D]

        return torch.from_numpy(out).reshape(*h.shape[:-1], D)
