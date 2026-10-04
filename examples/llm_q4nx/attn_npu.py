# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Prefill attention with both GEMMs on the NPU and the softmax on the host.

Why split it this way
---------------------
On the host, two thirds of a Gemma4 attention layer is its two matrix
products (Q.K^T and P.V), and those are what this stack already lowers well.
The softmax is the part it does not: the elementwise schedules give each core
one row per launch iteration, so a [4096, 1024] block takes ~40 ms on the
array against ~0.7 ms on the host. So the GEMMs go to the NPU and the masked
softmax stays on the host, with every operand in a shared page bound to its
chain -- the NPU reads Q, K and V and writes the scores where the host already
has them, and the host writes the probabilities where the NPU reads them. No
staging copy moves in either direction.

Shape of the work
-----------------
Multi-query only (one KV head): every query head attends the same K and V, so
the heads are stacked into one operand and each GEMM covers all of them.
Queries go in blocks of `QB`; a block sees

* sliding layers: keys [q0 - W, q0 + QB) -- one width for every block, so the
  blocks share a compiled GEMM;
* full layers: keys [0, q0 + QB) -- causal, so block b needs (b + 1) * QB
  keys and nothing past them.

Every block's GEMM is an op in ONE chain per product, so a layer is two
dispatches whatever the prompt length. Keys outside a block's window, and
padded keys past the prompt or before position 0, are removed by an additive
mask the host applies before the softmax.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import fused_mlp
import kernels as kn

#: Query rows per block, per head. A sliding block sees QB + W keys of which
#: each row uses W, so a smaller block would waste less -- but 256 measured
#: slower than 512 (12.3 vs 11.8 ms a layer): the GEMMs shrink with it, and
#: they are what this spends its time on.
QB = 512


def _page(shape, dtype):
    t = kn.shared_empty(shape, dtype)
    t.zero_()
    return t


class _Chain:
    """`n` independent GEMMs C_i = A_i @ B_i as one dispatch, on bound pages.

    `shapes` is a list of (M, K, N) per op; ops of one shape share a module.
    """

    def __init__(self, name, shapes):
        from triton.backends.amd_triton_npu.multilaunch import NPUChain

        self.n = len(shapes)
        self.chain = NPUChain(name)
        self.a, self.b, self.c = [], [], []
        srcs = {}
        for i, (M, K, N) in enumerate(shapes):
            stride = kn.unaliased_stride(K)
            if (M, K, N) not in srcs:
                bn = 512 if N % 512 == 0 else 256
                bm = fused_mlp._row_tile(M, K, bn)
                sc = kn.matmul_script(bm, bn)
                src = fused_mlp._exact_k_ttshared(
                    self.chain, M, K, N, bm, bn, stride, sc
                )
                srcs[(M, K, N)] = (src, sc, (M // bm, N // bn))
            src, sc, grid = srcs[(M, K, N)]
            self.chain.add(
                src,
                grid=grid,
                arg_map={0: i, 1: self.n + i, 2: 2 * self.n + i},
                args=(),
                transform_script=sc,
            )
            self.a.append(_page((M, stride), torch.bfloat16))
            self.b.append(_page((K, N), torch.bfloat16))
            self.c.append(_page((M, N), torch.float32))
        pad = 3 * self.n
        self.chain.add(
            kn._stage_pad_kernel,
            grid=(1,),
            arg_map={0: pad, 1: pad, 2: pad},
            args=tuple(torch.zeros(1024) for _ in range(3)),
            constexprs={"BLOCK": 1024},
            transform_script=kn.script(kn._PAD_SCRIPT),
        )
        self.bufs = [*self.a, *self.b, *self.c]
        self.io = {i: kn.shared_bo(t) for i, t in enumerate(self.bufs)}
        self.pad = np.zeros(1024, np.float32)

    def close(self):
        self.chain.close()

    def run(self):
        n = self.n
        self.chain.run(
            [*(kn._np(t) for t in self.bufs), self.pad],
            bo_key="attn",
            static_indices=set(),
            intermediate_indices={3 * n},
            output_indices=set(range(2 * n, 3 * n)),
            bound_buffers=self.io,
        )


#: Plans (block counts) one instance keeps. Each holds two chains and every
#: chain an NPU hardware context, and the device runs out of those (STATUS: ~35)
#: -- so a process serving many prompt lengths evicts the least recent.
MAX_PLANS = 2


class NPUAttention:
    """Causal multi-query attention for one (head dim, window), any length."""

    def __init__(self, n_q, dh, window=None):
        self.n_q, self.dh, self.window = n_q, dh, window
        self._plans = {}  # number of query blocks -> (qk, pv, key spans)
        self._masks = (None, None)  # (prompt length, additive mask per block)

    def _spans(self, nblk):
        """(first key position, key count) for each query block."""
        if self.window:
            return [(b * QB - self.window, QB + self.window) for b in range(nblk)]
        return [(0, (b + 1) * QB) for b in range(nblk)]

    def _plan(self, nblk):
        plan = self._plans.pop(nblk, None)
        if plan is not None:
            self._plans[nblk] = plan  # most recent last
        else:
            while len(self._plans) >= MAX_PLANS:
                old = self._plans.pop(next(iter(self._plans)))
                old[0].close()
                old[1].close()
            rows, dh = self.n_q * QB, self.dh
            spans = self._spans(nblk)
            tag = f"{dh}_{self.window or 0}_{nblk}"
            with kn._npu_driver():
                qk = _Chain(f"attn_qk_{tag}", [(rows, dh, nc) for _, nc in spans])
                pv = _Chain(f"attn_pv_{tag}", [(rows, nc, dh) for _, nc in spans])
            plan = self._plans[nblk] = (qk, pv, spans)
        return plan

    def _block_masks(self, N, spans):
        """The additive mask per block, for the most recent prompt length only."""
        if self._masks[0] == N:
            return self._masks[1]
        qi = torch.arange(QB)[:, None]
        masks = []
        for b, (lo, nc) in enumerate(spans):
            qpos = b * QB + qi
            kpos = lo + torch.arange(nc)[None, :]
            keep = (kpos <= qpos) & (kpos >= 0) & (kpos < N)
            if self.window:
                keep &= qpos - kpos < self.window
            masks.append(torch.zeros(QB, nc).masked_fill(~keep, float("-inf")))
        self._masks = (N, masks)
        return masks

    def __call__(self, q, k, v, scale=None):
        """q [N, n_q*dh], k/v [N, dh] -> [N, n_q*dh], float32."""
        N = q.shape[0]
        n_q, dh = self.n_q, self.dh
        nblk = math.ceil(N / QB)
        qk, pv, spans = self._plan(nblk)
        masks = self._block_masks(N, spans)
        if scale is not None and scale != 1.0:
            q = q * scale
        qh = q.reshape(N, n_q, dh)
        for b, (lo, nc) in enumerate(spans):
            q0, q1 = b * QB, min((b + 1) * QB, N)
            Q = qk.a[b][:, :dh].view(n_q, QB, dh)
            Q[:, : q1 - q0] = qh[q0:q1].transpose(0, 1)
            Q[:, q1 - q0 :] = 0
            k0, k1 = max(lo, 0), min(lo + nc, N)
            # Keys outside [0, N) stay zero; the mask is what removes them,
            # since a zero key still scores 0 rather than -inf.
            if k0 > lo or k1 < lo + nc:
                qk.b[b].zero_()
                pv.b[b].zero_()
            qk.b[b][:, k0 - lo : k1 - lo] = k[k0:k1].T
            pv.b[b][k0 - lo : k1 - lo] = v[k0:k1]
        with kn._npu_driver():
            qk.run()
        for b, (lo, nc) in enumerate(spans):
            S = qk.c[b].view(n_q, QB, nc)
            S.add_(masks[b])
            pv.a[b][:, :nc].view(n_q, QB, nc).copy_(torch.softmax(S, -1))
        with kn._npu_driver():
            pv.run()
        out = torch.empty(N, n_q, dh)
        for b in range(nblk):
            q0, q1 = b * QB, min((b + 1) * QB, N)
            out[q0:q1] = pv.c[b].view(n_q, QB, dh)[:, : q1 - q0].transpose(0, 1)
        return out.reshape(N, n_q * dh)
