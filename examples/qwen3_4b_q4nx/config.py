# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3-4B dims, the q4nx weight loader, and the RoPE table.

The dims and the loader are mlir-air's -- there is no reason to re-derive
either. This module only locates mlir-air's LLM packages on sys.path and
re-exports what the prefill needs.

Same shape as the Llama configs, and deliberately so. What differs is Qwen3's,
and only Qwen3's: per-head q/k norms in every layer, a single-theta RoPE table,
and a q dim that is not the model dim.
"""

import os
import sys

#: The harness every Q4NX example shares.
_SHARED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_q4nx"
)

#: This example's model, and the key into llm_q4nx/registry.py. The forward
#: asserts against it (llm_q4nx/llama_prefill.py::bound_model).
MODEL_NAME = "qwen3-4b"

# Qwen3-4B. Mirrors mlir-air's qwen3_4b_q4nx_weights; kept here so the kernels
# can be read without chasing an import.
D = 2560  # model dim
DH = 128  # head dim
N_Q_HEADS = 32
N_KV_HEADS = 8
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 4
DQ = N_Q_HEADS * DH  # 4096 -- NOT D. o_proj contracts 4096 -> 2560.
DK = N_KV_HEADS * DH  # 1024
DV = DK
INTER = 9728  # mlp intermediate
N_LAYERS = 36
VOCAB = 151936

#: Every Llama example here has DQ == D, so the two are interchangeable in a
#: way that happens to work. Qwen3-4B decouples them, which is the one place a
#: shape copied from a Llama config would load and then produce nonsense.
assert DQ != D, "Qwen3-4B has a decoupled q dim; o_proj contracts DQ -> D"

# The deployed fused decode's epsilon (fused_decode/models/all_models.h), also
# what mlir-air's Qwen3 reference forward uses.
RMS_EPS = 1e-6

#: RoPE base. Single theta, and no llama3 frequency scaling -- Qwen3 has
#: neither the scaling nor Gemma's second theta. `rope_lut` gets this from
#: mlir-air rather than from here; the value is restated for the reader.
ROPE_THETA = 1000000.0

# The Paris gate, from qwen3_4b_q4nx_inference.py. The Qwen3 tokenizer emits no
# BOS, so unlike the Llama examples the prompt starts at a real word.
PROMPT = [785, 6722, 315, 9625, 374]  # "The capital of France is"
EXPECT_FIRST = 12095  # " Paris"

#: Only `make_session_class`'s warmup turn uses this, and that path is reached
#: only by `--interactive`, which this model's driver does not support. Named
#: so the harness has something to warm with if it ever does: Qwen3 has no BOS,
#: so this is <|endoftext|> standing in for one.
BOS = 151643

#: What `--greedy` must generate from PROMPT, first token included --
#: mlir-air's own PARIS_GREEDY for this model, which it recorded on NPU2 with
#: real weights (" Paris. The capital of Germany is Berlin. The"). It is the
#: only assertion that covers the decode at all; the first-token gate above is
#: pure prefill and cannot see the builder environment, -DMODEL_TYPE,
#: GLU_SLICE or the core stack.
EXPECT_IDS = [12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576]

MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Qwen3-4B-NPU2")


if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import airsrc  # noqa: E402

#: mlir-air llms packages this model needs on sys.path. Its own q4nx package is
#: a thin re-parameterization: the reader, the codec and the reference forward
#: all live in `qwen3_8b_q4nx`, and `qwen3_4b` holds the `LlamaConfig` the
#: dims are taken from. mlir-air's own module reaches both the same way, by
#: inserting them at import.
AIR_PACKAGES = ("qwen3_4b_q4nx", "qwen3_8b_q4nx", "qwen3_4b")


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
    """Host-dequantized Qwen3-4B weights from a model.q4nx bundle.

    Dequantizing on the host is deliberate: prefill is compute-bound, so W4A16
    would cut memory traffic that is not the bottleneck and add unpack work to
    every GEMM tile.

    Read through mlir-air's `Q4nxModel` for this model -- which is Qwen3-8B's
    reader re-pointed at these dims, and which overrides `embed_norm_lmhead`
    because **Qwen3-4B ties its LM head** where the 8B does not. Asking the 8B
    reader for a separate `lm_head.weight` here raises a KeyError on a tensor
    the bundle does not carry.

    Returns a dict with
        layers: list of 36 dicts, each with
            attn_norm, ffn_norm : float32 [D]
            q_norm, k_norm      : float32 [DH]   (Qwen3's per-head norms)
            q, k, v, o, gate, up, down : bfloat16 [K, out]  (y = x @ W)
        embed, final_norm, lm_head : float32   (lm_head IS embed -- tied)
    """
    _add_air_paths()
    import qwen3_4b_q4nx_weights as gw

    if (gw.NUM_LAYERS, gw.D, gw.INTER, gw.VOCAB) != (N_LAYERS, D, INTER, VOCAB):
        raise RuntimeError(
            f"mlir-air's Qwen3-4B dims moved ({gw.NUM_LAYERS} layers of {gw.D}, "
            f"inter {gw.INTER}, vocab {gw.VOCAB}); this example says "
            f"{N_LAYERS} of {D}, inter {INTER}, vocab {VOCAB}"
        )

    src = model or MODEL_DEFAULT
    qm = gw.Q4nxModel(src)
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
    embed, final_norm, lm_head = qm.embed_norm_lmhead()
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
    from qwen3_4b_q4nx_weights import generate_rope_lut

    if dtype is None:
        from ml_dtypes import bfloat16

        dtype = bfloat16
    lut = np.stack([generate_rope_lut(p) for p in range(seq_len)])
    return lut.astype(dtype)
