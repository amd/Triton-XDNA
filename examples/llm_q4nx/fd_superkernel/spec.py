# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""What tells the models apart, as air-FLM's `fused_decode` entries and its
models' packages do. The selected model is FD_SUPERKERNEL_MODEL's, which a
model's decoder sets before it imports `geom` or `wave`."""

import importlib
import os

import airsrc


def _llama_lut(n):
    """Llama-3.2's RoPE LUT for n positions, [cos | sin] a position, with
    the llama3 frequency scaling (mlir-air's generate_rope_lut)."""
    from llama32_1b_weights import LlamaConfig, generate_rope_lut

    return generate_rope_lut(LlamaConfig(), n)


MODELS = {
    "llama-3.2-1b": dict(
        weights="llama32_1b_q4nx_weights",
        packages=("llama32_1b_q4nx", "llama32_1b"),
        default_source="FastFlowLM/Llama-3.2-1B-NPU2",
        layers=16,
        vocab=128256,
        eps=1e-6,
        # air-FLM pairs its proj rows (PAIR_ROWS=2); its non-paired build
        # (DECODE_PAIR_ROWS=1), which this one follows, runs as fast
        lm_rounds=36,
        kv_per_cu=2,
        norms=2,
        act="silu",
        glu_half=512,
        window=False,
        rope="lut",  # one LUT every layer shares
        rope_lut=_llama_lut,
        lm_head="tied",
    ),
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
    "qwen2.5-7b": dict(
        weights="qwen25_7b_q4nx_weights",
        packages=("qwen25_7b_q4nx", "qwen25_3b"),
        lm_rounds=7,
        kv_per_cu=1,
        q_seg=8,  # 7 q heads a kv head, padded to 8 (GQA_SEG)
        norms=2,
        act="silu",
        glu_half=256,
        window=False,
        rope="bias",  # q/k/v biases behind the cos/sin, no qk-norm
        lm_head="requant",  # its own, quantized on load
        source="requant",  # an HF checkpoint quantized on load
        default_source="Qwen/Qwen2.5-7B-Instruct",
        layers_per_w=7,  # DECODE_WGROUP: a W buffer stays under 4 GiB
    ),
    "qwen2.5-3b": dict(
        weights="qwen25_3b_q4_weights",
        packages=("qwen25_3b_q4", "qwen25_3b"),
        lm_rounds=12,
        kv_per_cu=1,
        cus=2,  # 2 kv heads: the CUs of one column
        q_seg=8,
        norms=2,
        act="silu",
        glu_half=512,
        window=False,
        rope="bias",
        lm_head="q4_0",
        source="q4_0",  # FastFlowLM's bundle, Q4_0
        default_source="FastFlowLM/Qwen2.5-3B-Instruct-NPU2",
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
airsrc.add_air_paths(*S["packages"])
W = importlib.import_module(S["weights"])  # the model's weights module
