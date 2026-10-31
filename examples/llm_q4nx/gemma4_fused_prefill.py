# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4 prefill on mlir-air's fused one-device NPU prefill.

mlir-air's `fused_prefill` runs every op of a 128-token chunk on one configured
device, with the int4 weights dequantized on the cores. This class drives it
in place of `Gemma4Prefill`'s own forward and keeps everything a decoder reads
from a prefiller: the shared KV slab, the context length and the first
token.

The K/V handoff stays zero-copy. The fused prefill computes each chunk's roped
K and normed V on the host and hands them to `_kv_append`, which writes its own
attention records. The override here also writes those rows into the
`kv_layout` slab, the same way `Gemma4Prefill._store_kv` does, so both decoders
read the slab in place: the NPU decode through its BO, the iGPU decode through
its HIP alias. The fused prefill's host copy of K/V, kept only for `kv_stack()`,
is dropped.
"""

import importlib
import json
from pathlib import Path

import numpy as np
import torch

import airsrc
import kv_layout
from config import AIR_PACKAGES, head_dim
from dense_fused_prefill import engine_on_launch_runtime, run_fused
from gemma4_prefill import Gemma4Prefill, _bf16_np

#: The model's mlir-air package. E4B's runs E2B's code at its own dims, so
#: everything is reached through it rather than through E2B's directly.
PACKAGE = AIR_PACKAGES[0]


def _weights():
    airsrc.add_air_paths(PACKAGE)
    return importlib.import_module(f"{PACKAGE}_weights")


def _fused_prefill_class():
    _weights()
    runtime = importlib.import_module(
        f"air_examples.llms.{PACKAGE}.fused_prefill.runtime"
    )

    engine_on_launch_runtime()
    return runtime.FusedPrefill


class Gemma4FusedPrefill(Gemma4Prefill):
    """`Gemma4Prefill` whose `prefill` runs mlir-air's fused NPU prefill.

    `build_dir` holds the fused prefill's artifacts (`build.py OUT`). The host
    weights are still loaded: the decoders built from this object read the
    embedding, norms and head from it.
    """

    ENGINE = "air-fused"

    def __init__(self, build_dir, *a, **kw):
        kw.setdefault("backend", "cpu")
        super().__init__(*a, **kw)
        owner = self
        base = _fused_prefill_class()

        class _Fused(base):
            def _kv_append(self, L, r0, k, v):
                super()._kv_append(L, r0, k, v)
                self.kv.pop(L, None)
                # k, v are [T, n_kv, dh].
                t = k.shape[0]
                owner._store_kv_rows(L, r0, k.reshape(t, -1), v.reshape(t, -1))

        # A build with more than one KV head is made for a fixed prompt length,
        # recorded in its manifest; one without takes any.
        man = json.loads((Path(build_dir) / "manifest.json").read_text())
        self._rows = min(self.kv_attn_maxl, self.max_seq, man.get("max_len") or 1 << 30)
        self._fused = _Fused(build_dir, max_len=self._rows)

    def load_weights(self, model=None):
        super().load_weights(model)
        self._fused.load_weights(_weights().Q4nxModel(model or self.model))

    def make_npu_resident(self):
        """Nothing to convert: the projections run in the fused prefill.

        The host weights here only give the decoders the embedding, norms and
        head, so padding them into NPU buffers would be work nothing reads.
        """

    def _store_kv_rows(self, layer_idx, r0, k, v):
        """Rows [r0, r0+t) of an owning layer, into its slab."""
        t = k.shape[0]
        dh = head_dim(layer_idx)
        for region, src in (
            (kv_layout.K_REGION, _bf16_np(torch.from_numpy(np.asarray(k, np.float32)))),
            (kv_layout.V_REGION, _bf16_np(torch.from_numpy(np.asarray(v, np.float32)))),
        ):
            kv_layout.scatter_rows(
                self._region(layer_idx, region)[r0 : r0 + t], src, dh, region
            )
        if layer_idx in self._kv_src:
            kf, vf = self._kv_src[layer_idx]
            kf[r0 : r0 + t] = k
            vf[r0 : r0 + t] = v

    def prefill(self, ids):
        ids = [int(t) for t in np.asarray(ids).reshape(-1)]
        N = len(ids)
        # These bound the rows written below; checked before the slab is touched.
        if N > self._rows:
            raise ValueError(
                f"a {N}-token prompt exceeds the {self._rows} rows this prefiller "
                f"holds (ATTN_MAXL={self.kv_attn_maxl}, max_seq={self.max_seq}, "
                f"and the fused prefill build's own limit)"
            )
        self._zero_slab()
        out = run_fused(self._fused, ids)
        logits = torch.from_numpy(np.asarray(out, np.float32))
        self._ids = ids
        self.current_context_length = N
        self.last_first_token = int(logits.argmax())
        return logits.reshape(-1)

    def suspend(self):
        """Release the fused prefill's hardware contexts, keeping its BOs."""
        self._fused.suspend()


class Gemma4TritonFusedPrefill(Gemma4FusedPrefill):
    """`Gemma4FusedPrefill` on the devices `gemma4_triton_fused` builds,
    run by `gemma4_npu_prefill`."""

    ENGINE = "triton-fused"

    def __init__(self, build_dir, *a, **kw):
        from gemma4_npu_prefill import NpuPrefill

        kw.setdefault("backend", "cpu")
        Gemma4Prefill.__init__(self, *a, **kw)
        self._rows = min(self.kv_attn_maxl, self.max_seq)
        self._fused = NpuPrefill(
            build_dir, max_len=self._rows, kv_store=self._store_kv_rows
        )

    def load_weights(self, model=None):
        Gemma4Prefill.load_weights(self, model)
        self._fused.load_weights(model or self.model)
