# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3-8B dims, the q4nx weight loader, and the RoPE table.

The dims and the loader are mlir-air's -- there is no reason to re-derive
either. This module only locates mlir-air's LLM packages on sys.path and
re-exports what the prefill needs.

Qwen3-4B's config at 4096, with two differences and only two: `INTER` is 12288
rather than 9728, and **the LM head is not tied**. The forward is shared
unchanged (`../llm_q4nx/qwen3_prefill.py`); everything Qwen3-specific about it
-- per-head q/k norms, single-theta RoPE -- is the same here.
"""

import os
import sys

#: The harness every Q4NX example shares.
_SHARED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_q4nx"
)

#: This example's model, and the key into llm_q4nx/registry.py. The forward
#: asserts against it (llm_q4nx/llama_prefill.py::bound_model).
MODEL_NAME = "qwen3-8b"

# Qwen3-8B. Mirrors mlir-air's qwen3_8b_q4nx_weights; kept here so the kernels
# can be read without chasing an import.
D = 4096  # model dim
DH = 128  # head dim
N_Q_HEADS = 32
N_KV_HEADS = 8
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 4
DQ = N_Q_HEADS * DH  # 4096 -- equal to D here, so o_proj is square
DK = N_KV_HEADS * DH  # 1024
DV = DK
INTER = 12288  # mlp intermediate
N_LAYERS = 36
VOCAB = 151936

#: Qwen3-4B's config asserts `DQ != D`, because on that model o_proj contracts
#: 4096 -> 2560 and a shape copied from a Llama config would load and then
#: produce nonsense. Here the two coincide again, which is a fact about the
#: model rather than a return to the Llama shape -- the same forward runs both,
#: and it never assumes they are equal.
assert DQ == D, "Qwen3-8B's q dim and model dim coincide; o_proj is square"

# The deployed fused decode's epsilon (fused_decode/models/all_models.h), also
# what mlir-air's Qwen3 reference forward uses.
RMS_EPS = 1e-6

#: RoPE base. Single theta, and no llama3 frequency scaling -- Qwen3 has
#: neither the scaling nor Gemma's second theta. `rope_lut` gets this from
#: mlir-air rather than from here; the value is restated for the reader.
ROPE_THETA = 1000000.0

# The Paris gate, from qwen3_8b_q4nx_inference.py -- the same prompt and the
# same expected token as Qwen3-4B's, which is not something to assume across
# models (Llama-3.1-8B needs a different phrasing from the 1B's) but is true
# here. The Qwen3 tokenizer emits no BOS, so the prompt starts at a real word.
PROMPT = [785, 6722, 315, 9625, 374]  # "The capital of France is"
EXPECT_FIRST = 12095  # " Paris"

#: Only `make_session_class`'s warmup turn uses this, and that path is reached
#: only by `--interactive`, which this model's driver does not support. Named
#: so the harness has something to warm with if it ever does: Qwen3 has no BOS,
#: so this is <|endoftext|> standing in for one.
BOS = 151643

#: What `--greedy` must generate from PROMPT, first token included --
#: mlir-air's own PARIS_GREEDY for this model, which it recorded on NPU2 with
#: real weights (" Paris. The capital of Italy is Rome. The"). It is the only
#: assertion that covers the decode at all; the first-token gate above is pure
#: prefill and cannot see the builder environment, -DMODEL_TYPE, GLU_SLICE or
#: the core stack -- and this model moves the last of those to 6144.
#:
#: Note it is NOT Qwen3-4B's continuation, which shares the first five tokens
#: and then says Germany/Berlin where this says Italy/Rome. Two models agreeing
#: on a gate's first token says nothing about the rest.
EXPECT_IDS = [12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576]

MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Qwen3-8B-NPU2")


if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import airsrc  # noqa: E402

#: mlir-air llms packages this model needs on sys.path: its own q4nx package,
#: which holds the reader, the codec and the reference forward, and `qwen3_4b`,
#: which holds the `LlamaConfig` the dims are taken from. mlir-air's own module
#: reaches the latter the same way, by inserting it at import.
AIR_PACKAGES = ("qwen3_8b_q4nx", "qwen3_4b")


def _add_air_paths():
    airsrc.add_air_paths(*AIR_PACKAGES)


def _fingerprint(path):
    """Short digest of the bundle's safetensors header.

    mlir-air's Llama reader exposes this as `Q4nxModel.fingerprint()`; its
    Qwen3 reader does not, so the same rule -- sha256 of the header bytes
    alone, never the multi-GB payload, first 8 hex digits -- is applied here.
    It changes whenever the Hub re-exports the bundle, which is what makes it
    worth printing next to a result.
    """
    import hashlib

    with open(path, "rb") as f:
        hlen = int.from_bytes(f.read(8), "little")
        return hashlib.sha256(f.read(hlen)).hexdigest()[:8]


def load_q4nx(model=None):
    """Host-dequantized Qwen3-8B weights from a model.q4nx bundle.

    Dequantizing on the host is deliberate: prefill is compute-bound, so W4A16
    would cut memory traffic that is not the bottleneck and add unpack work to
    every GEMM tile.

    Read through mlir-air's `Q4nxModel` for this model, which is the one
    Qwen3-4B's reader subclasses. **Qwen3-8B does not tie its LM head**: the
    bundle carries a bf16 `model.embed_tokens` for the input gather and a
    separate Q4NX-packed `lm_head.weight` for the logits. Using the embedding
    instead still produces fluent text off a plausible logit vector -- on
    Llama-3.1-8B, the other untied model here, that cost a gate failure at
    57618 to find.

    The head is taken *first*, before the layers, for the reason the 8B's
    loader records: dequantizing a 151936x4096 head peaks at several GiB inside
    `dequant`, and paying that against an empty heap rather than on top of the
    resident layer weights lowers the load's peak by about that much.

    Returns a dict with
        layers: list of 36 dicts, each with
            attn_norm, ffn_norm : float32 [D]
            q_norm, k_norm      : float32 [DH]   (Qwen3's per-head norms)
            q, k, v, o, gate, up, down : bfloat16 [K, out]  (y = x @ W)
        embed, final_norm, lm_head : float32   (lm_head is its own tensor)
    """
    _add_air_paths()
    import qwen3_8b_q4nx_weights as gw

    if (gw.NUM_LAYERS, gw.D, gw.INTER, gw.VOCAB) != (N_LAYERS, D, INTER, VOCAB):
        raise RuntimeError(
            f"mlir-air's Qwen3-8B dims moved ({gw.NUM_LAYERS} layers of {gw.D}, "
            f"inter {gw.INTER}, vocab {gw.VOCAB}); this example says "
            f"{N_LAYERS} of {D}, inter {INTER}, vocab {VOCAB}"
        )

    src = model or MODEL_DEFAULT
    qm = gw.Q4nxModel(src)
    embed, final_norm, lm_head = qm.embed_norm_lmhead()

    layers = []
    for k in range(N_LAYERS):
        w = qm.layer_weights(k)
        attn_norm, ffn_norm = qm.layer_rms(k)
        q_norm, k_norm = qm.layer_qk_norm(k)
        layers.append(
            dict(
                attn_norm=attn_norm,
                ffn_norm=ffn_norm,
                q_norm=q_norm,
                k_norm=k_norm,
                **w,
            )
        )
    return dict(
        layers=layers,
        embed=embed,
        final_norm=final_norm,
        lm_head=lm_head,
        fingerprint=_fingerprint(gw.resolve_q4nx_model(src)),
    )


def rope_lut(seq_len, dtype=None):
    """[seq_len, DH] = [cos_0..cos_63, sin_0..sin_63] per position.

    Built by calling mlir-air's own single-position generator once per
    position, rather than by vectorizing it here: that generator is what the
    fused decode rotates each new token with, including its bf16 rounding of
    cos and sin, and a reimplementation that agrees today is a thing that can
    stop agreeing silently. `seq_len` is a prompt length, so the loop is
    thousands of iterations at most.

    Single theta (1e6) and no llama3 frequency scaling -- the Llama configs'
    `generate_rope_lut` carries scaling that Qwen3 must not have.
    """
    import numpy as np

    _add_air_paths()
    from qwen3_8b_q4nx_weights import generate_rope_lut

    if dtype is None:
        from ml_dtypes import bfloat16

        dtype = bfloat16
    lut = np.stack([generate_rope_lut(p) for p in range(seq_len)])
    return lut.astype(dtype)
