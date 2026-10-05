# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4-E2B prefill on mlir-air's fused one-device NPU prefill.

mlir-air's `fused_prefill` runs every op of a 128-token chunk on one configured
device, with the int4 weights dequantized on the cores. This class drives it
in place of `Gemma4Prefill`'s own forward and keeps everything a decoder reads
from a prefiller: the shared KV slab, its fan-out to KV-shared layers, the
context length and the first token.

The K/V handoff stays zero-copy. The fused prefill computes each chunk's roped
K and normed V on the host and hands them to `_kv_append`, which writes its own
attention records. The override here also writes those rows into the
`kv_layout` slab, the same way `Gemma4Prefill._store_kv` does, so both decoders
read the slab in place: the NPU decode through its BO, the iGPU decode through
its HIP alias. The fused prefill's host copy of K/V, kept only for `kv_stack()`,
is dropped.
"""

import numpy as np
import torch

import airsrc
import kv_layout
from config import head_dim
from gemma4_prefill import Gemma4Prefill, _bf16_np


def _fused_prefill_class():
    airsrc.add_air_paths("gemma4_e2b_q4nx")
    from air_examples.llms.gemma4_e2b_q4nx.fused_prefill import runtime

    return runtime.FusedPrefill


class Gemma4FusedPrefill(Gemma4Prefill):
    """`Gemma4Prefill` whose `prefill` runs mlir-air's fused NPU prefill.

    `build_dir` holds the fused prefill's artifacts (`build.py OUT`). The host
    weights are still loaded: the decoders built from this object read the
    embedding, norms and head from it.
    """

    def __init__(self, build_dir, *a, **kw):
        kw.setdefault("backend", "cpu")
        super().__init__(*a, **kw)
        owner = self
        base = _fused_prefill_class()

        class _Fused(base):
            def _kv_append(self, L, r0, k, v):
                super()._kv_append(L, r0, k, v)
                self.kv.pop(L, None)
                owner._store_kv_rows(L, r0, k, v)

        self._fused = _Fused(build_dir, max_len=min(self.kv_attn_maxl, self.max_seq))

    def load_weights(self, model=None):
        super().load_weights(model)
        airsrc.add_air_paths("gemma4_e2b_q4nx")
        import gemma4_e2b_q4nx_weights as gw

        self._fused.load_weights(gw.Q4nxModel(model or self.model))

    def make_npu_resident(self):
        """Nothing to convert: the projections run in the fused prefill.

        The host weights here only give the decoders the embedding, norms and
        head, so padding them into NPU buffers would be work nothing reads.
        """

    def _store_kv_rows(self, layer_idx, r0, k, v):
        """Rows [r0, r0+t) of an owning layer, into every slab that reads it."""
        t = k.shape[0]
        dh = head_dim(layer_idx)
        pair = (
            (kv_layout.K_REGION, _bf16_np(torch.from_numpy(np.asarray(k, np.float32)))),
            (kv_layout.V_REGION, _bf16_np(torch.from_numpy(np.asarray(v, np.float32)))),
        )
        for L in self._kv_fanout[layer_idx]:
            for region, src in pair:
                kv_layout.scatter_rows(
                    self._region(L, region)[r0 : r0 + t], src, dh, region
                )
        if layer_idx in self._kv_src:
            kf, vf = self._kv_src[layer_idx]
            kf[r0 : r0 + t] = k
            vf[r0 : r0 + t] = v

    def prefill(self, ids):
        ids = [int(t) for t in np.asarray(ids).reshape(-1)]
        N = len(ids)
        # Both bound the rows written below; checked before the slab is touched.
        limit = min(self.kv_attn_maxl, self.max_seq)
        if N > limit:
            raise ValueError(
                f"a {N}-token prompt exceeds the {limit} rows this prefiller "
                f"holds (ATTN_MAXL={self.kv_attn_maxl}, max_seq={self.max_seq})"
            )
        self._zero_slab()
        logits = torch.from_numpy(np.asarray(self._fused.prefill(ids), np.float32))
        self._ids = ids
        self.current_context_length = N
        self.last_first_token = int(logits.argmax())
        return logits.reshape(-1)

    def suspend(self):
        """Release the fused prefill's hardware contexts, keeping its BOs."""
        self._fused.suspend()
