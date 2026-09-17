# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Llama-3.1-8B dims, the q4nx weight loader, and the RoPE table.

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
MODEL_NAME = "llama-3.1-8b"

# Llama-3.1-8B. Mirrors mlir-air's llama32_3b_weights.LlamaConfig; kept here so
# the kernels can be read without chasing an import.
D = 4096  # model dim
DQ = 4096  # q proj out  (32 heads * 128)
DK = 1024  # k proj out  (8 kv heads * 128)
DV = 1024
DH = 128  # head dim
N_Q_HEADS = 32
N_KV_HEADS = 8
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 4
INTER = 14336  # mlp intermediate -- not a power of two, unlike the smaller two
N_LAYERS = 32
VOCAB = 128256

# The deployed fused decode's epsilon (fused_decode/models/all_models.h:56).
# It is a global default there rather than a per-model value, and llama3.2-3b.h
# does not override it, so this matches the 1B.
RMS_EPS = 1e-6

# The Paris gate, from llama31_8b_q4nx_prefill.py -- and NOT the bare "The
# capital of France is" the 1B and 3B use. On Llama-3.1-8B that phrasing ranks
# " a" (264) fractionally above " Paris" (12366), and the HF bf16 reference
# agrees, so gating on it would fail a perfectly correct build. "...is called"
# makes it decisive: " Paris" leads by 3.4 logits.
PROMPT = [128000, 791, 6864, 3363, 315, 9822, 374, 2663]
EXPECT_FIRST = 12366  # " Paris"
BOS = 128000  # <|begin_of_text|>, for the session warmup

MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Llama-3.1-8B-NPU2")


if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import airsrc  # noqa: E402

#: mlir-air llms packages this model needs on sys.path: its own q4nx package
#: (`_proj_dims`), the base llama32_3b package (LlamaConfig and the RoPE table),
#: and -- less obviously -- the 1B's q4nx package, because `Q4nxModel` is the
#: bundle reader for every Q4NX model and lives there. mlir-air's own
#: `llama31_8b_q4nx_weights` reaches it the same way, by putting the 1B's
#: directory on sys.path at import.
AIR_PACKAGES = ("llama31_8b_q4nx", "llama32_3b", "llama32_1b_q4nx")


def _add_air_paths():
    airsrc.add_air_paths(*AIR_PACKAGES)


def load_q4nx(model=None):
    """Host-dequantized Llama-3.1-8B weights from a model.q4nx bundle.

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
    from llama31_8b_q4nx_weights import _proj_dims, llama31_8b_config

    cfg = llama31_8b_config()
    if cfg.n_layers != N_LAYERS or cfg.emb_dim != D:
        raise RuntimeError(
            f"mlir-air's Llama-3.1-8B config moved ({cfg.n_layers} layers of "
            f"{cfg.emb_dim}); this example says {N_LAYERS} of {D}"
        )
    dims = _proj_dims(cfg)

    qm = Q4nxModel(model or MODEL_DEFAULT)

    # Llama-3.1-8B does NOT tie its LM head to the embedding, unlike the 1B and
    # 3B, whose `embed_norm_lmhead()` returns the tied full-precision embedding
    # as the head. The bundle carries a real quantized `lm_head.weight`, and
    # using the embedding instead still produces fluent text off a plausible
    # logit vector -- it cost a gate failure at 57618 to find.
    #
    # Order matters here, and only for this model. Dequantizing the untied LM
    # head peaks at ~11 GiB inside `dequant` to produce a 2 GiB result, so
    # doing it after the layers -- as the 1B and 3B loaders do, where the head
    # is tied and free -- pays that transient on top of 13 GiB of resident
    # layer weights. Taking it first pays it against an empty heap and drops
    # the load's peak by about that much.
    embed = qm.bf16("model.embed_tokens.weight")
    final_norm = qm.bf16("model.norm.weight")
    lm_head = qm.dequant("lm_head.weight", VOCAB, D)

    layers = []
    for k in range(N_LAYERS):
        w = qm.layer_weights(k, dims)
        attn_norm, ffn_norm = qm.layer_rms(k)
        layers.append(dict(attn_norm=attn_norm, ffn_norm=ffn_norm, **w))
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
    from llama31_8b_q4nx_weights import generate_rope_lut, llama31_8b_config

    if dtype is None:
        from ml_dtypes import bfloat16

        dtype = bfloat16
    return generate_rope_lut(llama31_8b_config(), seq_len, dtype)
