# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Phi-4-mini dims, the q4nx weight loader, and the LongRoPE table.

The dims and the loader are mlir-air's -- there is no reason to re-derive
either, and a second copy of the LongRoPE frequency construction is exactly the
drift worth avoiding. This module only locates mlir-air's LLM packages on
sys.path and re-exports what the prefill needs.

Same shape as the other configs here. What differs is Phi-4's, and only
Phi-4's: the rotation covers 96 of each head's 128 lanes, and its frequencies
come from a factor table in the bundle rather than from a closed form.
"""

import os
import sys

#: The harness every Q4NX example shares.
_SHARED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_q4nx"
)

#: This example's model, and the key into llm_q4nx/registry.py. The forward
#: asserts against it (llm_q4nx/llama_prefill.py::bound_model).
MODEL_NAME = "phi4-mini"

# Phi-4-mini-instruct. Mirrors mlir-air's phi4_mini_q4nx_weights; kept here so
# the kernels can be read without chasing an import.
D = 3072  # model dim
DH = 128  # head dim
N_Q_HEADS = 24
N_KV_HEADS = 8
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 3 -- the narrowest GQA ratio here
DQ = N_Q_HEADS * DH  # 3072 -- equal to D, so o_proj is square
DK = N_KV_HEADS * DH  # 1024
DV = DK
INTER = 8192  # mlp intermediate
N_LAYERS = 32
VOCAB = 200064  # the largest here

#: How many of each head's lanes the rotation covers:
#: `partial_rotary_factor=0.75` against a 128-wide head. The trailing 32 are
#: copied through, which is what the decode's `rope_partial` does. Every other
#: model here rotates the whole head, so this is the constant that makes
#: `phi4_prefill.Phi4Prefill` necessary.
ROPE_DIM = 96

assert ROPE_DIM < DH, "Phi-4-mini rotates only part of each head"

# The deployed fused decode's epsilon. `fused_decode/models/phi4-4b.h` does not
# override `RMS_NORM_EPS`, so this model takes the 1e-6 default in
# all_models.h, like every other model here. Note mlir-air's own numpy prefill
# uses 1e-5 for its FINAL norm -- HF's `rms_norm_eps` -- so the two disagree
# there; 1e-6 is what the device computes and what the KV handoff has to match.
RMS_EPS = 1e-6

#: RoPE base, before the LongRoPE factor table divides into it. `rope_lut`
#: takes the whole construction from mlir-air; the value is restated for the
#: reader.
ROPE_THETA = 10000.0

# The Paris gate, from phi4_mini_q4nx_prefill.py. Its own tokenizer, so neither
# the ids nor the expected token match any other model's.
PROMPT = [976, 9029, 328, 10128, 382]  # "The capital of France is"
EXPECT_FIRST = 12650  # " Paris"

#: Only `make_session_class`'s warmup turn uses this. Phi-4 emits no BOS of its
#: own for a raw prompt; this is <|endoftext|> standing in for one.
BOS = 199999

#: What `--greedy` must generate from PROMPT, first token included.
#:
#: Unlike every other model here, this is NOT copied from a `PARIS_GREEDY`
#: constant -- Phi-4-mini's mlir-air driver carries none. It was produced by
#: running **mlir-air's own prefill feeding the same fused decode**
#: (`generate_stream(..., prefiller=LlamaQ4nxPrefill(...), min_prefill=1)`),
#: which is the path this example substitutes itself into. No Triton code runs
#: in that reference. This example reproduces it exactly, all ten tokens.
#:
#: Worth recording, because it cost a diagnosis: mlir-air's **replay** path
#: (`prefiller=None`, which computes the prompt's K/V with the decode kernel
#: token by token instead of prefilling) continues differently from here --
#: `[..., 1617, 10358, 5678, 11]` against this `[..., 1617, 37118, 88691,
#: 1299]`, the same first seven tokens and then a split. So the two paths do
#: not agree for this model, whatever their driver's docstring says, and the
#: replay path is the wrong thing to gate against. The prefill path is the
#: right reference precisely because it is the one we replace.
EXPECT_IDS = [12650, 11, 1118, 382, 5542, 395, 1617, 37118, 88691, 1299]

MODEL_DEFAULT = os.environ.get(
    "Q4NX_MODEL_SOURCE", "FastFlowLM/Phi4-mini-Instruct-NPU2"
)


if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import airsrc  # noqa: E402

#: mlir-air llms packages this model needs on sys.path: its own q4nx package,
#: the 1B's (whose `Q4nxModel` is the bundle reader for every Q4NX model), and
#: `llama32_3b` (the `LlamaConfig` the dims are read back against). mlir-air's
#: own weights module reaches all three the same way, by inserting them at
#: import.
AIR_PACKAGES = ("phi4_mini_q4nx", "llama32_1b_q4nx", "llama32_3b")


def _add_air_paths():
    airsrc.add_air_paths(*AIR_PACKAGES)


def load_q4nx(model=None):
    """Host-dequantized Phi-4-mini weights from a model.q4nx bundle.

    Dequantizing on the host is deliberate: prefill is compute-bound, so W4A16
    would cut memory traffic that is not the bottleneck and add unpack work to
    every GEMM tile.

    Read through mlir-air's `Q4nxModel` directly, as the 8B's loader does,
    rather than through its `load_q4nx_weights` wrapper: that wrapper returns a
    `LlamaWeights` of bf16 arrays and the forward wants a dict of float32 norms
    and bf16 projections, so going through it would mean packing a dataclass
    only to unpack it.

    Two Phi-4 facts this relies on:

    * **The bundle is already split.** HF's Phi3ForCausalLM fuses
      `self_attn.qkv_proj` and `mlp.gate_up_proj`, and so does the GGUF the
      converter reads -- but the q4nx bundle it writes is split into q/k/v and
      gate/up, so there is nothing to unfuse here. `load_weights` re-fuses them
      its own way, to issue one GEMM per group.
    * **The LM head is tied.** `tie_word_embeddings=true` and the HF checkpoint
      carries no `lm_head.weight`. The bundle does carry a separate quantized
      one; it is a lossier copy of the embedding and using it would be wrong.
      Returning `embed` for both is what makes `load_weights` alias them rather
      than hold two copies.

    Returns a dict with
        layers: list of 32 dicts, each with
            attn_norm, ffn_norm : float32 [D]
            q, k, v, o, gate, up, down : bfloat16 [K, out]  (y = x @ W)
        embed, final_norm, lm_head : float32   (lm_head IS embed -- tied)
    """
    _add_air_paths()
    from llama32_1b_q4nx_weights import Q4nxModel
    from phi4_mini_q4nx_weights import _proj_dims, phi4_mini_config

    cfg = phi4_mini_config()
    if cfg.n_layers != N_LAYERS or cfg.emb_dim != D or cfg.vocab_size != VOCAB:
        raise RuntimeError(
            f"mlir-air's Phi-4-mini config moved ({cfg.n_layers} layers of "
            f"{cfg.emb_dim}, vocab {cfg.vocab_size}); this example says "
            f"{N_LAYERS} of {D}, vocab {VOCAB}"
        )
    dims = _proj_dims(cfg)

    qm = Q4nxModel(model or MODEL_DEFAULT)
    embed = qm.bf16("model.embed_tokens.weight")
    final_norm = qm.bf16("model.norm.weight")

    layers = []
    for k in range(N_LAYERS):
        w = qm.layer_weights(k, dims)
        attn_norm, ffn_norm = qm.layer_rms(k)
        layers.append(dict(attn_norm=attn_norm, ffn_norm=ffn_norm, **w))
    return dict(
        layers=layers,
        embed=embed,
        final_norm=final_norm,
        lm_head=embed,  # tied
        fingerprint=qm.fingerprint(),
    )


def rope_lut(seq_len, dtype=None):
    """[seq_len, ROPE_DIM] = [cos_0..cos_47, sin_0..sin_47] per position.

    ROPE_DIM wide, not DH: the trailing 32 lanes of each head are not rotated,
    so there are no cos/sin entries for them. `Phi4Prefill._rope` checks the
    width rather than inferring the head size from it, because inferring is
    exactly what the inherited Llama `_rope` does and what would silently
    mis-split the heads.

    mlir-air's `generate_rope_lut`, unmodified. It carries three things a
    reimplementation would have to keep in step: the LongRoPE factor table read
    out of the bundle, the choice of the short table at or below
    `original_max_position_embeddings` and the long one past it, and the
    `sqrt(1 + ln(max/orig)/ln(orig))` scaling of both cos and sin. That last is
    a deliberate divergence from FastFlowLM on mlir-air's part -- they return
    1.0 -- so it is emphatically not something to re-derive here.
    """
    _add_air_paths()
    from phi4_mini_q4nx_weights import generate_rope_lut, phi4_mini_config

    if dtype is None:
        from ml_dtypes import bfloat16

        dtype = bfloat16
    return generate_rope_lut(phi4_mini_config(), seq_len, dtype)
