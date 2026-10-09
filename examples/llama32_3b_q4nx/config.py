# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Llama-3.2-3B dims, the q4nx weight loader, and the RoPE table.

The dims and the loader are mlir-air's -- there is no reason to re-derive
either, and a second copy of the RoPE frequency scaling is exactly the drift
worth avoiding. This module only locates mlir-air's LLM packages on sys.path
and re-exports what the prefill needs.

Same shape as the 1B's config, and deliberately so: the forward
(`../llm_q4nx/llama_prefill.py`) is shared, and it resolves every constant from
whichever `config` is first on sys.path. What differs is below, and only below.
"""

import os
import sys

#: The harness every Q4NX example shares.
_SHARED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_q4nx"
)

#: This example's model, and the key into llm_q4nx/registry.py. The shared
#: Llama forward asserts against it (llm_q4nx/llama_prefill.py::bound_model).
MODEL_NAME = "llama-3.2-3b"

# Llama-3.2-3B. Mirrors mlir-air's llama32_3b_weights.LlamaConfig; kept here so
# the kernels can be read without chasing an import.
D = 3072  # model dim
DQ = 3072  # q proj out  (24 heads * 128)
DK = 1024  # k proj out  (8 kv heads * 128)
DV = 1024
DH = 128  # head dim -- twice the 1B's
N_Q_HEADS = 24
N_KV_HEADS = 8
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 3, against the 1B's 4
INTER = 8192  # mlp intermediate -- same as the 1B despite the wider model
N_LAYERS = 28
VOCAB = 128256

# The deployed fused decode's epsilon (fused_decode/models/all_models.h:56).
# It is a global default there rather than a per-model value, and llama3.2-3b.h
# does not override it, so this matches the 1B.
RMS_EPS = 1e-6

# The Paris gate, from llama32_3b_q4nx_prefill.py -- the same prompt and the
# same expected token as the 1B, which makes a wrong-weights mixup invisible to
# the gate alone. The fingerprint check in load_q4nx is what catches that.
PROMPT = [128000, 791, 6864, 315, 9822, 374]  # "The capital of France is"
EXPECT_FIRST = 12366  # " Paris"
BOS = 128000  # <|begin_of_text|>, for the session warmup

MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Llama-3.2-3B-NPU2")


if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import airsrc  # noqa: E402

#: mlir-air llms packages this model needs on sys.path: its own q4nx package
#: (`_proj_dims`), the base llama32_3b package (LlamaConfig and the RoPE table),
#: and -- less obviously -- the 1B's q4nx package, because `Q4nxModel` is the
#: bundle reader for every Q4NX model and lives there. mlir-air's own
#: `llama32_3b_q4nx_weights` reaches it the same way, by putting the 1B's
#: directory on sys.path at import.
AIR_PACKAGES = ("llama32_3b_q4nx", "llama32_3b", "llama32_1b_q4nx")


def _add_air_paths():
    airsrc.add_air_paths(*AIR_PACKAGES)


def load_q4nx(model=None):
    """Host-dequantized Llama-3.2-3B weights from a model.q4nx bundle.

    Dequantizing on the host is deliberate: prefill is compute-bound, so W4A16
    would cut memory traffic that is not the bottleneck and add unpack work to
    every GEMM tile.

    Reads through mlir-air's `Q4nxModel` directly, as the 1B does, rather than
    through its `load_q4nx_weights` wrapper: that wrapper returns a
    `LlamaWeights` of bf16 arrays, and the forward wants the 1B's dict of
    float32 norms and bf16 projections. Going through it would mean packing a
    dataclass only to unpack it, and two dtype round trips.

    One thing the 3B needs that the 1B does not: its bundle ships I8-packed
    headers, which do not encode each projection's logical [out, K], so the
    dims have to be supplied. `_proj_dims` derives them from the config, and is
    imported rather than rewritten -- getting it wrong transposes a projection,
    which loads and then produces fluent nonsense.

    Returns a dict with
        layers: list of 28 dicts, each with
            attn_norm, ffn_norm : float32 [D]
            q, k, v, o, gate, up, down : bfloat16 [K, out]  (y = x @ W)
        embed, final_norm, lm_head : float32
    """
    _add_air_paths()
    from llama32_1b_q4nx_weights import Q4nxModel
    from llama32_3b_q4nx_weights import _proj_dims
    from llama32_3b_weights import LlamaConfig

    cfg = LlamaConfig()
    if cfg.n_layers != N_LAYERS or cfg.emb_dim != D:
        raise RuntimeError(
            f"mlir-air's Llama-3.2-3B config moved ({cfg.n_layers} layers of "
            f"{cfg.emb_dim}); this example says {N_LAYERS} of {D}"
        )
    dims = _proj_dims(cfg)

    qm = Q4nxModel(model or MODEL_DEFAULT)
    layers = []
    for k in range(N_LAYERS):
        w = qm.layer_weights(k, dims)
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
    """[seq_len, DH] = [cos_0..cos_63, sin_0..sin_63] per position.

    mlir-air's generate_rope_lut, unmodified. It carries the llama3 frequency
    scaling (factor 32, old_ctx 8192) that the fused decode also applies; a
    reimplementation that drops it produces K rotated on the wrong frequencies
    and long prompts silently degrade.

    The 3B's table is twice the 1B's width, because its heads are: 128 wide, so
    64 cos and 64 sin.
    """
    _add_air_paths()
    from llama32_3b_weights import LlamaConfig, generate_rope_lut

    if dtype is None:
        from ml_dtypes import bfloat16

        dtype = bfloat16
    return generate_rope_lut(LlamaConfig(n_layers=N_LAYERS), seq_len, dtype)
