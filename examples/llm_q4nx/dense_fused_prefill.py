# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The dense Q4NX models' prefill on mlir-air's fused one-device NPU prefill.

mlir-air's `shared/fused_prefill` runs Llama, Qwen3, Phi-4 and Gemma3 from one
model table: every op of a prompt chunk on one configured device, with the
int4 weights dequantized on the cores. `fused_prefill_class(base, ...)` makes a
model's own prefill class (`LlamaPrefill` or a subclass) run its `prefill` on
that instead, and keeps everything a decoder reads from a prefiller.

The handoff is the base class's. The fused prefill keeps each layer's roped K
and raw V, the same contract as `LlamaPrefill.save_kv_npz`; `prefill` passes
them to `_kv_store`, so the host cache, `kv_view`, `kv_stack`, the npz and a
bound decoder sink all see them exactly as they see the Triton prefill's.
"""

import os
import weakref

import numpy as np
import torch

import airsrc


def _dense():
    airsrc.register_air_examples()
    from air_examples.llms.shared.fused_prefill import dense, models

    engine_on_launch_runtime()
    return dense, models


def engine_on_launch_runtime():
    """Run mlir-air's fused prefill engine on this process's launch runtime.

    The engine is written against pyxrt. Under `AMD_TRITON_NPU_RUNTIME=hsa` its
    `xrt` is swapped for `hsa_xrt`, the same calls on the HSA runtime.
    """
    if os.environ.get("AMD_TRITON_NPU_RUNTIME") != "hsa":
        return
    from air_examples.llms.shared.fused_prefill import engine
    from triton.backends.amd_triton_npu import hsa_xrt

    engine.xrt = hsa_xrt


#: Fused prefills that may hold their hardware contexts between prompts.
_holding = weakref.WeakSet()


def release_fused_prefills():
    """Release the hardware contexts of every fused prefill between prompts.

    Each reopens them at its next prefill. Called before an NPU decoder opens
    its own, and by the Triton backend when the device refuses it one. True if
    any context was released.
    """
    held = [f for f in _holding if not f.suspended]
    _holding.clear()
    for f in held:
        f.suspend()
    return bool(held)


def run_fused(fused, ids):
    """`fused.prefill(ids)`, leaving its hardware contexts open.

    Reopening them reconfigures the array, so they stay open until something
    else needs the room and calls `release_fused_prefills`. The device grants
    a limited number, and Triton prefill chains in the same process may hold
    the rest; chains reopen on their next run, so they are closed to make
    room.
    """
    from triton.backends.amd_triton_npu import driver
    from triton.backends.amd_triton_npu.multilaunch import NPUChain

    if release_fused_prefills not in driver.context_releasers:
        driver.context_releasers.append(release_fused_prefills)
    try:
        try:
            return fused.prefill(ids)
        except RuntimeError as e:
            if "HWCTX" not in str(e) or not NPUChain._open:
                raise
            while NPUChain._close_stalest():
                pass
            # A refused resume can leave some of its own contexts open.
            for c in fused.contexts():
                c.close()
            return fused.prefill(ids)
    finally:
        _holding.add(fused)


class _FusedPrefill:
    """Mixed in ahead of a `LlamaPrefill` class; see `fused_prefill_class`."""

    #: The model's key in mlir-air's `shared/fused_prefill/models.py`.
    AIR_MODEL = None
    ENGINE = "air-fused"

    def __init__(self, build_dir, *a, **kw):
        kw.setdefault("backend", "cpu")
        super().__init__(*a, **kw)
        dense, models = _dense()
        self._desc = models.MODELS[self.AIR_MODEL]
        if self._desc.layers != self.n_layers:
            raise ValueError(
                f"the fused prefill runs all {self._desc.layers} layers of "
                f"{self.AIR_MODEL}; n_layers={self.n_layers} was asked for"
            )
        self._fused = dense.DensePrefill(build_dir, self._desc)

    def load_weights(self, model=None):
        """The fused prefill's weights only.

        The base class's host weights would serve nothing: the decoders load
        their own, and the projections run in the fused prefill. Skipping
        them saves a bf16 copy of every matrix.
        """
        from config import MODEL_DEFAULT

        self._fused.load_weights(model or self.model or MODEL_DEFAULT)

    def make_npu_resident(self):
        """Nothing to convert: no host weights are loaded."""

    def prefill(self, ids):
        ids = [int(t) for t in np.asarray(ids).reshape(-1)]
        N = len(ids)
        limit = min(self._fused.max_len, self.max_seq)
        if N > limit:
            raise ValueError(
                f"a {N}-token prompt exceeds the {limit} rows this prefiller "
                f"holds (fused max_len={self._fused.max_len}, "
                f"max_seq={self.max_seq})"
            )
        logits = np.asarray(run_fused(self._fused, ids), np.float32)
        for L in range(self.n_layers):
            k, v = self._fused.kv_view(L)
            self._kv_store(L, torch.from_numpy(k), torch.from_numpy(v), N)
        self.current_context_length = N
        return torch.from_numpy(logits)

    def clear_context(self):
        super().clear_context()
        self._fused.clear_context()

    def suspend(self):
        """Release the fused prefill's hardware context, keeping its BOs."""
        self._fused.suspend()


def fused_prefill_class(base, air_model):
    """`base` (a `LlamaPrefill` class) with its prefill on mlir-air's fused one.

    The result is constructed as `cls(build_dir, **base_kwargs)`.
    """
    return type(
        f"Fused{base.__name__}",
        (_FusedPrefill, base),
        {"AIR_MODEL": air_model, "__doc__": _FusedPrefill.__doc__},
    )
