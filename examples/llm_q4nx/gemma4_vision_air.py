# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""mlir-air's NPU vision tower (`llms/gemma4_e2b_vision`) on FastFlowLM's weights.

mlir-air's tower runs its GEMMs and the attention as AIR ELFs, built per image
shape, and the elementwise work between them on the host (`hostops.c`). It
serves as the reference for this repository's NPU tower.

mlir-air reads the HF bf16 checkpoint; here it reads `vision_weight.q4nx`
(`gemma4_vision_q4nx`), which holds the same bf16 tensors, so `AirWeights` only
adapts names to mlir-air's `VisionWeights` interface.

E2B and E4B share the tower. mlir-air's modules read `GEMMA4_VARIANT`, which
the E4B config sets; the projection width comes from the weights.
"""

import sys

import numpy as np
import torch
from ml_dtypes import bfloat16

import airsrc
import gemma4_vision_q4nx as vw

#: mlir-air's norm names -> this loader's.
_NORMS = {
    "input_ln": "norm1",
    "post_attn_ln": "post_attn_norm",
    "pre_ffn_ln": "norm2",
    "post_ffn_ln": "ffn_post_norm",
    "q_norm": "q_norm",
    "k_norm": "k_norm",
}


def _air_vision():
    """mlir-air's vision modules, importable. Returns the encoder module."""
    import os

    from triton.backends.amd_triton_npu.driver import find_peano_root

    # mlir-air compiles its AIE kernels with the Peano it finds here.
    if not os.environ.get("PEANO_INSTALL_DIR") and find_peano_root():
        os.environ["PEANO_INSTALL_DIR"] = find_peano_root()
    airsrc.add_air_paths("gemma4_e2b_q4nx")
    d = str(airsrc.air_llms_root() / "gemma4_e2b_vision")
    if d not in sys.path:
        sys.path.insert(0, d)
    import gemma4_vision_encoder

    return gemma4_vision_encoder


class AirWeights:
    """mlir-air's `VisionWeights` interface over `gemma4_vision_q4nx`'s dict."""

    def __init__(self, w):
        self._w = w
        self.out_dim = w["embedding_proj"].shape[1]
        self.path = None

    def norm(self, layer, which):
        return np.asarray(self._w["layers"][layer][_NORMS[which]], np.float32)

    def wT(self, layer, which):
        """[in, out] bf16, as the loader already returns it."""
        return np.ascontiguousarray(self._w["layers"][layer][which], bfloat16)

    def clamp(self, layer, which):
        in_lo, in_hi, out_lo, out_hi = self._w["layers"][layer]["clip"][which]
        return (in_lo, in_hi), (out_lo, out_hi)

    def patch_proj_T(self):
        return np.ascontiguousarray(self._w["patch_embd"], bfloat16)

    def position_table(self):
        return np.asarray(self._w["position_table"], np.float32)

    def embed_proj_T(self):
        return np.ascontiguousarray(self._w["embedding_proj"], bfloat16)


class AirVisionTower:
    """`encode(pixels, pos_ids)` -> `[n_soft, out_hidden]` f32, on mlir-air's
    NPU tower. The ELFs for an image shape are compiled on first use and
    cached under `cache_dir` (mlir-air's default: `G4_VISION_BUILD_DIR`)."""

    def __init__(self, weights=None, model=None, ops=None, verbose=False, encoder=None):
        """`ops` are the encoder's device ops: mlir-air's `NpuOps` by default,
        or `gemma4_vision_device.TritonOps`. `encoder` makes the encoder class
        from mlir-air's encoder module; its own `VisionEncoder` by default."""
        from triton.backends.amd_triton_npu import driver

        enc = _air_vision()
        w = weights if weights is not None else vw.load_vision(model)
        self.fingerprint = w.get("fingerprint")
        self.ops = ops or enc.NpuOps(verbose=verbose)
        cls = encoder(enc) if encoder is not None else enc.VisionEncoder
        self.enc = cls(self.ops, weights=AirWeights(w))
        # The tower holds a hardware context per ELF, and the prefill and
        # decode need theirs. Registered so a refused launch can reclaim them.
        driver.context_releasers.append(self.release)

    def encode(self, pixels, pos_ids):
        soft = self.enc.encode(np.asarray(pixels), np.asarray(pos_ids))
        return torch.from_numpy(np.ascontiguousarray(soft, np.float32))

    def release(self):
        """Close the tower's hardware contexts, keeping its buffers, so the
        weights are not uploaded again. The next encode reopens them. True if
        any were open."""
        return _release(self.ops)

    def timings(self):
        """Per-op device seconds from the last encodes, mlir-air's own split."""
        return self.ops.timings()


def _release(ops):
    """Close `ops`' hardware contexts, keeping its buffers: an `NpuOps`'
    kernel cache, or whatever else it holds (`ops.release`)."""
    if hasattr(ops, "release"):
        return ops.release()
    cache = ops.cache
    loaded = list(cache._loaded.items())
    for name, backend in loaded:
        backend.unload()
        type(cache)._contexts.pop((cache._uid, name), None)
    cache._loaded.clear()
    return bool(loaded)
