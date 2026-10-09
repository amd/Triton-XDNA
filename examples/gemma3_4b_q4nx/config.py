# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma3-4B dims, the q4nx weight loader, and the two RoPE tables.

The dims and the loader are mlir-air's -- there is no reason to re-derive
either, and Gemma carries more conventions that are silent when wrong than any
other model here. This module only locates mlir-air's LLM packages on sys.path
and re-exports what the prefill needs.

Four of those conventions are already resolved in the bundle, and re-applying
any of them is the failure this file exists to prevent. They are documented at
`load_q4nx`.
"""

import os
import sys

#: The harness every Q4NX example shares.
_SHARED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_q4nx"
)

#: This example's model, and the key into llm_q4nx/registry.py. The forward
#: asserts against it (llm_q4nx/llama_prefill.py::bound_model).
MODEL_NAME = "gemma3-4b"

# Gemma3-4B (text). Mirrors mlir-air's gemma3_4b_q4nx_weights.
D = 2560  # model dim
DH = 256  # head dim -- twice the 3B's, four times the 1B's
N_Q_HEADS = 8
N_KV_HEADS = 4
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 2
DQ = N_Q_HEADS * DH  # 2048 -- NOT D. o_proj contracts 2048 -> 2560.
DK = N_KV_HEADS * DH  # 1024
DV = DK
INTER = 10240  # mlp intermediate
N_LAYERS = 34
VOCAB = 262208

#: Like Qwen3-4B and unlike every Llama here.
assert DQ != D, "Gemma3-4B has a decoupled q dim; o_proj contracts DQ -> D"

# The deployed fused decode's epsilon, and what mlir-air's reference forward
# uses. The norm weights are fed as-is at this eps -- see `load_q4nx`.
RMS_EPS = 1e-6

#: Gemma3's dual-theta RoPE. Five layers in six are "local" (theta 1e4, no
#: position scaling, attention limited to a 1024-token sliding window); every
#: sixth is "global" (theta 1e6 with positions divided by 8, unbounded
#: attention). Both halves of that pairing matter: a local layer given the
#: global table attends on the wrong frequencies, and a global layer given the
#: window truncates the context it exists to carry.
ROPE_LOCAL_THETA = 10000.0
ROPE_GLOBAL_THETA = 1000000.0
ROPE_GLOBAL_LINEAR_FACTOR = 8.0
SLIDING_WINDOW = 1024
SLIDING_PATTERN = 6


def is_global_layer(layer_idx):
    """True on the unbounded, theta-1e6 layers: every sixth, counting from 1.

    mlir-air's own predicate, restated rather than imported only because the
    forward reads it once per layer; `load_q4nx` checks the two agree.
    """
    return (layer_idx + 1) % SLIDING_PATTERN == 0


# The Paris gate, from gemma3_4b_q4nx_inference.py. Gemma's tokenizer does emit
# a BOS (2), so the prompt is six tokens for five words.
PROMPT = [2, 818, 5279, 529, 7001, 563]  # "<bos>The capital of France is"
EXPECT_FIRST = 9079  # " Paris"
BOS = 2

#: What `--greedy` must generate from PROMPT, first token included --
#: mlir-air's own PARIS_GREEDY for this model. The only assertion that covers
#: the decode at all; the first-token gate above is pure prefill.
EXPECT_IDS = [9079, 236761, 108, 50429, 563, 496, 4185, 3988, 573, 1610]

MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Gemma3-4B-NPU2")


if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import airsrc  # noqa: E402

#: mlir-air llms packages this model needs on sys.path. Its q4nx package is
#: self-contained -- unlike Qwen3-4B's, it does not re-parameterize a sibling.
AIR_PACKAGES = ("gemma3_4b_q4nx",)


def _add_air_paths():
    airsrc.add_air_paths(*AIR_PACKAGES)


def _fingerprint(path):
    """Short digest of the bundle's safetensors header -- header bytes only.

    The same rule as mlir-air's Llama reader exposes as
    `Q4nxModel.fingerprint()`; its Gemma reader does not have one.
    """
    import hashlib

    with open(path, "rb") as f:
        hlen = int.from_bytes(f.read(8), "little")
        return hashlib.sha256(f.read(hlen)).hexdigest()[:8]


def load_q4nx(model=None):
    """Host-dequantized Gemma3-4B weights from a model.q4nx bundle.

    **Four Gemma conventions are already applied in the bundle.** Each one is
    invisible if re-applied -- the model still runs and still produces fluent
    text -- so they are listed here rather than left to be rediscovered:

    * **The (1+w) norm fold.** Gemma's RMSNorm is `x * (1 + w)`, but the bundle
      passes through `gemma-3-4b-it-Q4_1.gguf` and llama.cpp's Gemma conversion
      already folded the +1 into every norm weight. The AIE kernel does a plain
      `norm * w`, and so does this forward. Adding +1 here would double-fold.
      (mlir-air's check: the stored norm mean is ~8, not ~0.)
    * **The embedding scale.** `model.embed_tokens` is already multiplied by
      sqrt(hidden_size), so the gather is as-is.
    * **The LM head is tied but stored separately, and unscaled.** `lm_head` is
      its own Q4NX tensor holding the *raw* matrix -- using the scaled embedding
      for the logits instead would scale them by 50.6.
    * **qk-norm weights carry the same fold**, and are fed as-is.

    Returns a dict with
        layers: list of 34 dicts, each with
            attn_norm, post_attn_norm : float32 [D]
            ffn_norm, post_ffn_norm   : float32 [D]
            q_norm, k_norm            : float32 [DH]
            q, k, v, o, gate, up, down : bfloat16 [K, out]  (y = x @ W)
        embed, final_norm, lm_head : float32
    """
    _add_air_paths()
    import gemma3_4b_q4nx_weights as gw

    if (gw.NUM_LAYERS, gw.D, gw.DH, gw.INTER, gw.VOCAB) != (
        N_LAYERS,
        D,
        DH,
        INTER,
        VOCAB,
    ):
        raise RuntimeError(
            f"mlir-air's Gemma3-4B dims moved ({gw.NUM_LAYERS} layers of {gw.D}, "
            f"head {gw.DH}, inter {gw.INTER}, vocab {gw.VOCAB}); this example "
            f"says {N_LAYERS} of {D}, head {DH}, inter {INTER}, vocab {VOCAB}"
        )
    # The sliding pattern decides which RoPE table and which mask every layer
    # gets, and it is the one constant here that is restated rather than
    # imported, so it is the one worth checking.
    if any(is_global_layer(L) != gw.is_global_layer(L) for L in range(N_LAYERS)):
        raise RuntimeError(
            "this example and mlir-air disagree about which Gemma3 layers are "
            "global; the RoPE table and the attention window both follow from it"
        )

    src = model or MODEL_DEFAULT
    qm = gw.Q4nxModel(src)
    # Before the layers, not after, and for the reason the 8B's config gives:
    # this bundle's LM head is a real Q4NX tensor, and dequantizing 262208x2560
    # peaks well above the 2.5 GiB result it produces. Paying that spike while
    # 34 layers are already resident sets the high-water mark for the whole
    # load; paying it against an empty heap does not.
    embed, final_norm, lm_head = qm.embed_norm_lmhead()
    layers = []
    for k in range(N_LAYERS):
        w = qm.layer_weights(k)
        attn_norm, post_attn_norm, ffn_norm, post_ffn_norm = qm.layer_rms(k)
        q_norm, k_norm = qm.layer_qk_norm(k)
        layers.append(
            dict(
                attn_norm=attn_norm,
                post_attn_norm=post_attn_norm,
                ffn_norm=ffn_norm,
                post_ffn_norm=post_ffn_norm,
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


def rope_lut(seq_len, dtype=None, global_layer=False):
    """[seq_len, DH] = [cos_0..cos_127, sin_0..sin_127] per position.

    Built by calling mlir-air's own single-position generator once per
    position, so the table this forward rotates K with is the one the fused
    decode rotates each new token with, bf16 rounding included.

    `global_layer` picks which of Gemma's two tables this is: theta 1e6 with
    positions divided by 8, or theta 1e4 with no scaling. There is no single
    correct table for the model -- the forward needs both, and picks per layer.
    """
    import numpy as np

    _add_air_paths()
    from gemma3_4b_q4nx_weights import generate_rope_lut

    if dtype is None:
        from ml_dtypes import bfloat16

        dtype = bfloat16
    theta = ROPE_GLOBAL_THETA if global_layer else ROPE_LOCAL_THETA
    lf = ROPE_GLOBAL_LINEAR_FACTOR if global_layer else 1.0
    lut = np.stack([generate_rope_lut(p, theta, lf) for p in range(seq_len)])
    return lut.astype(dtype)
