# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""What tells the models apart, as air-FLM's `fused_decode` entries and its
models' packages do. The model is the one a model's entry module selected
(`select`) before anything imports this."""

import importlib
import types

import airsrc


def _lut(package, config="LlamaConfig"):
    """The model's RoPE LUT for n positions, [cos | sin] a position, from
    mlir-air's generate_rope_lut in `package` on its `config`: Llama-3.2's
    with the llama3 frequency scaling, Phi-4-mini's LongRoPE."""

    def lut(n):
        m = importlib.import_module(package)
        return m.generate_rope_lut(getattr(m, config)(), n)

    return lut


def _dims(package, config="LlamaConfig"):
    """The dims a weights module without them leaves to `package`'s config."""

    def dims():
        c = getattr(importlib.import_module(package), config)()
        return dict(
            D=c.emb_dim,
            DH=c.head_dim,
            N_Q_HEADS=c.n_heads,
            N_KV_HEADS=c.n_kv_heads,
            DQ=c.n_heads * c.head_dim,
            DK=c.n_kv_heads * c.head_dim,
            INTER=c.hidden_dim,
        )

    return dims


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
        rope_lut=_lut("llama32_1b_weights"),
        lm_head="tied",
    ),
    "llama-3.2-3b": dict(
        weights="llama32_3b_q4nx_weights",
        packages=("llama32_3b_q4nx", "llama32_3b", "llama32_1b_q4nx"),
        dims=_dims("llama32_3b_weights"),
        default_source="FastFlowLM/Llama-3.2-3B-NPU2",
        layers=28,
        vocab=128256,
        eps=1e-6,
        lm_rounds=18,  # VOCAB_CHUNK_I2 9, paired
        kv_per_cu=2,
        norms=2,
        act="silu",
        glu_half=512,
        window=False,
        rope="lut",
        rope_lut=_lut("llama32_3b_weights"),
        lm_head="tied",
    ),
    "phi4-mini": dict(
        weights="phi4_mini_q4nx_weights",
        packages=("phi4_mini_q4nx", "llama32_1b_q4nx", "llama32_3b"),
        dims=_dims("phi4_mini_q4nx_weights", "phi4_mini_config"),
        default_source="FastFlowLM/Phi4-mini-Instruct-NPU2",
        layers=32,
        vocab=200064,
        eps=1e-6,
        lm_rounds=36,  # VOCAB_CHUNK_I2 18, paired
        kv_per_cu=2,
        norms=2,
        act="silu",
        glu_half=512,
        window=False,
        rope="lut",
        rope_dim=96,  # partial rotary: the first 96 of a head's 128
        rope_lut=_lut("phi4_mini_q4nx_weights", "phi4_mini_config"),
        lm_head="tied",
    ),
    "qwen3-8b": dict(
        weights="qwen3_8b_q4nx_weights",
        packages=("qwen3_8b_q4nx", "qwen3_4b"),
        default_source="FastFlowLM/Qwen3-8B-NPU2",
        lm_rounds=16,  # VOCAB_CHUNK_I2 8, paired
        kv_per_cu=2,
        norms=2,
        act="silu",
        glu_half=512,
        window=False,
        lm_head="q4",
        layers_per_w=9,
    ),
    "llama-3.1-8b": dict(
        weights="llama31_8b_q4nx_weights",
        packages=("llama31_8b_q4nx", "llama32_3b", "llama32_1b_q4nx"),
        dims=_dims("llama31_8b_q4nx_weights", "llama31_8b_config"),
        default_source="FastFlowLM/Llama-3.1-8B-NPU2",
        layers=32,
        vocab=128256,
        eps=1e-6,
        lm_rounds=32,  # VOCAB_CHUNK_I2 16, paired
        kv_per_cu=2,
        norms=2,
        act="silu",
        glu_half=512,
        window=False,
        rope="lut",
        rope_lut=_lut("llama31_8b_q4nx_weights", "llama31_8b_config"),
        lm_head="q4",
        layers_per_w=8,
    ),
    "qwen3-4b": dict(
        default_source="FastFlowLM/Qwen3-4B-NPU2",
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
        default_source="FastFlowLM/Gemma3-4B-NPU2",
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

from . import _model as NAME  # noqa: E402

if NAME is None:
    raise RuntimeError(
        "no model selected: import a model's module (fd_superkernel.qwen3_4b, "
        "...) or call fd_superkernel.select first"
    )
S = MODELS[NAME]
airsrc.add_air_paths(*S["packages"])
W = importlib.import_module(S["weights"])  # the model's weights module
if "dims" in S:
    W = types.SimpleNamespace(**{**vars(W), **S["dims"]()})
