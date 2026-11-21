# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""What tells the models apart, as air-FLM's `fused_decode` entries and its
models' packages do. The selected model is FD_SUPERKERNEL_MODEL's, which a
model's decoder sets before it imports `geom` or `wave`."""

import importlib
import os

MODELS = {
    "qwen3-4b": dict(
        weights="qwen3_4b_q4nx_weights",
        packages=("qwen3_4b_q4nx", "qwen3_8b_q4nx", "qwen3_4b"),
        lm_rounds=30,  # VOCAB_CHUNK_I2
        kv_per_cu=2,
        norms=2,
        act="silu",
        glu_half=256,
        window=False,
        lm_head="tied",  # the embedding, requantized to Q4
    ),
    "gemma3-4b": dict(
        weights="gemma3_4b_q4nx_weights",
        packages=("gemma3_4b_q4nx",),
        lm_rounds=5,
        kv_per_cu=1,
        norms=4,  # Gemma's sandwich
        act="gelu",
        glu_half=512,
        window=True,  # SLIDING_WINDOW on the layers that are not global
        lm_head="q4",  # the bundle's own
    ),
}

NAME = os.environ.get("FD_SUPERKERNEL_MODEL", "qwen3-4b")
S = MODELS[NAME]
W = importlib.import_module(S["weights"])  # the model's weights module
