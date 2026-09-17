# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Llama-3.2-1B dims, the q4nx weight loader, and the RoPE table.

The dims and the loader are mlir-air's -- there is no reason to re-derive
either, and a second copy of the RoPE frequency scaling is exactly the drift
worth avoiding. This module only locates mlir-air's LLM
packages on sys.path and re-exports what the prefill needs.
"""

import os
import sys

#: The harness every Q4NX example shares.
_SHARED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_q4nx"
)

# Llama-3.2-1B. Mirrors mlir-air's llama32_1b_q4nx_weights; kept here so the
# kernels can be read without chasing an import.
D = 2048  # model dim
DQ = 2048  # q proj out  (32 heads * 64)
DK = 512  # k proj out  (8 kv heads * 64)
DV = 512
DH = 64  # head dim
N_Q_HEADS = 32
N_KV_HEADS = 8
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 4
INTER = 8192  # mlp intermediate
N_LAYERS = 16
VOCAB = 128256

# The deployed fused decode's epsilon (fused_decode/models/all_models.h:54).
# HF's config says 1e-5; matching the decode matters more than matching HF,
# and at D=2048 the two differ far below bf16 resolution.
RMS_EPS = 1e-6

# The Paris gate, from llama32_1b_q4nx_prefill.py.
PROMPT = [128000, 791, 6864, 315, 9822, 374]  # "The capital of France is"
EXPECT_FIRST = 12366  # " Paris"
BOS = 128000  # <|begin_of_text|>, for the session warmup

MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Llama-3.2-1B-NPU2")


if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import airsrc  # noqa: E402

#: mlir-air llms packages this model needs on sys.path: its own q4nx package
#: (the weight reader) and the base llama32_1b package (LlamaConfig and the
#: RoPE table).
AIR_PACKAGES = ("llama32_1b_q4nx", "llama32_1b")


def _add_air_paths():
    airsrc.add_air_paths(*AIR_PACKAGES)


def load_q4nx(model=None):
    """Host-dequantized Llama-3.2-1B weights from a model.q4nx bundle.

    Dequantizing on the host is deliberate: prefill is compute-bound, so W4A16
    would cut memory traffic that is not the bottleneck and add unpack work to
    every GEMM tile.

    Returns a dict with
        layers: list of 16 dicts, each with
            attn_norm, ffn_norm : float32 [D]
            q, k, v, o, gate, up, down : bfloat16 [K, out]  (y = x @ W)
        embed, final_norm, lm_head : float32
    """
    _add_air_paths()
    from llama32_1b_q4nx_weights import Q4nxModel

    qm = Q4nxModel(model or MODEL_DEFAULT)
    layers = []
    for k in range(N_LAYERS):
        w = qm.layer_weights(k)
        attn_norm, ffn_norm = qm.layer_rms(k)
        layers.append(dict(attn_norm=attn_norm, ffn_norm=ffn_norm, **w))
    embed, final_norm, lm_head = qm.embed_norm_lmhead()
    return dict(
        layers=layers,
        embed=embed,
        final_norm=final_norm,
        lm_head=lm_head,
        fingerprint=qm.fingerprint(),
    )


def rope_lut(seq_len, dtype=None):
    """[seq_len, 64] = [cos_0..cos_31, sin_0..sin_31] per position.

    mlir-air's generate_rope_lut, unmodified. It carries the llama3 frequency
    scaling (factor 32, old_ctx 8192) that the fused decode also applies; a
    reimplementation that drops it produces K rotated on the wrong frequencies
    and long prompts silently degrade.
    """
    _add_air_paths()
    from llama32_1b_weights import LlamaConfig, generate_rope_lut

    if dtype is None:
        from ml_dtypes import bfloat16

        dtype = bfloat16
    return generate_rope_lut(LlamaConfig(n_layers=N_LAYERS), seq_len, dtype)
