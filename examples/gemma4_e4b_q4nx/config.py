# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E4B dims, the weight loader, and the RoPE tables.

Gemma4-E4B is Gemma4-E2B's architecture at a larger size; `../gemma4_e2b_q4nx/
config.py` describes the family. Against E2B:

* **width and depth** -- D 2560 and 42 layers;
* **attention pattern** -- five sliding layers then one full, repeating;
* **KV heads** -- two, each shared by four query heads, where E2B has one;
* **KV sharing** -- layers from 24 up attend a lower layer's cache;
* **FFN** -- 10240 wide on every layer; E2B doubles it on its KV-shared layers.

mlir-air has no separate E4B code. Its `gemma4_e4b_q4nx` package selects E4B
with `GEMMA4_VARIANT=e4b` and runs `gemma4_e2b_q4nx`'s modules, which read their
dims from that variable when they are first imported. This module sets it on
import, so everything a Gemma4-E4B process loads from mlir-air is at E4B's dims.
"""

import os
import sys

#: Before any of mlir-air's Gemma4 modules are imported; see the docstring.
os.environ["GEMMA4_VARIANT"] = "e4b"

#: The harness every Q4NX example shares.
_SHARED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_q4nx"
)

#: This example's model, and the key into llm_q4nx/registry.py.
MODEL_NAME = "gemma4-e4b"

# Gemma4-E4B-IT. Mirrors mlir-air's gemma4_e2b_q4nx_weights.VARIANTS["e4b"];
# `load_q4nx` checks the two agree.
D = 2560  # model dim
N_Q_HEADS = 8
N_KV_HEADS = 2  # GQA -- four q heads per kv head
DH_SLIDING = 256  # head dim on a sliding layer
DH_GLOBAL = 512  # head dim on a full-attention layer
INTER = 10240  # mlp intermediate, every layer
N_LAYERS = 42
VOCAB = 262144
PLI_D = 256  # per-layer input dim (hidden_size_per_layer_input)

#: Layers from here up read another layer's KV.
FIRST_KV_SHARED = 24
#: Every FULL_PERIOD-th layer is a full-attention one.
FULL_PERIOD = 6
#: Sliding-attention window, on the sliding layers only.
SLIDING_WINDOW = 512

RMS_EPS = 1e-6
ROPE_SLIDING_THETA = 10000.0
ROPE_GLOBAL_THETA = 1000000.0
#: 1.0, not `head_dim**-0.5`, as on E2B (`query_pre_attn_scalar` is 1.0).
ATTN_SCALE = 1.0
FINAL_LOGIT_SOFTCAP = 30.0
#: No `1 + w` fold in RMSNorm, as on E2B.
RMS_NORM_ADDS_ONE = False
PLE_MODEL_PROJ_SCALE = float(D**-0.5)
PLE_INPUT_SCALE = float(2.0**-0.5)
#: The bundle has the embedding scale folded in already, as on E2B.
EMBED_SCALE = 1.0

#: Bound to the widest layer, as on E2B: `llm_q4nx/llama_prefill.py` imports
#: these and sizes the base KV cache from them; the forward uses `head_dim(L)`.
DH = DH_GLOBAL
DQ = N_Q_HEADS * DH_GLOBAL
DK = DV = N_KV_HEADS * DH_GLOBAL
Q_PER_KV = N_Q_HEADS // N_KV_HEADS


def is_sliding(layer_idx):
    """Five sliding layers then one full, repeating."""
    return (layer_idx % FULL_PERIOD) != FULL_PERIOD - 1


def head_dim(layer_idx):
    return DH_SLIDING if is_sliding(layer_idx) else DH_GLOBAL


def mlp_inter(layer_idx):
    """This layer's FFN width. Read back against the bundle in `load_q4nx`."""
    return INTER


def owns_kv(layer_idx):
    return layer_idx < FIRST_KV_SHARED


def kv_source_layer(layer_idx):
    """Which layer's cache this one attends. Itself, below FIRST_KV_SHARED;
    mlir-air's map otherwise."""
    return int(_weights().kv_source_layer(layer_idx))


# The Paris gate, as on E2B: `2` is <bos>, which this tokenizer does not add.
PROMPT = [2, 818, 5279, 529, 7001, 563]  # "<bos>The capital of France is"
EXPECT_FIRST = 9079  # " Paris"
BOS = 2

#: What `--greedy` must generate from PROMPT: mlir-air's `PARIS_GREEDY` for
#: this model, then the stop on 106 (<end_of_turn>).
EXPECT_IDS = [9079, 236761, 106]

EOS_IDS = (106, 1)  # <end_of_turn>, <eos>

MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Gemma4-E4B-IT-NPU2")


if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import airsrc  # noqa: E402

#: mlir-air's E4B package; its modules put `gemma4_e2b_q4nx` on sys.path.
AIR_PACKAGES = ("gemma4_e4b_q4nx",)


def _add_air_paths():
    airsrc.add_air_paths(*AIR_PACKAGES)


def _weights():
    """mlir-air's Gemma4 weights module, at E4B's dims."""
    _add_air_paths()
    import gemma4_e4b_q4nx_weights as gw

    if gw.VARIANT != "e4b":
        raise RuntimeError(
            f"mlir-air's Gemma4 weights module is loaded as {gw.VARIANT}; a "
            f"Gemma4-E4B process must not import Gemma4-E2B's first"
        )
    return gw


def _fingerprint(src):
    """Short digest of the bundle's safetensors header, as on E2B."""
    import hashlib

    try:
        path = _weights().resolve_q4nx_model(src)
        with open(path, "rb") as f:
            hlen = int.from_bytes(f.read(8), "little")
            return hashlib.sha256(f.read(hlen)).hexdigest()[:8]
    except Exception:
        return ""


def load_q4nx(model=None):
    """Host-dequantized Gemma4-E4B weights, in the layout E2B's loader returns
    (`../gemma4_e2b_q4nx/config.py::load_q4nx`)."""
    import numpy as np

    gw = _weights()
    if (gw.NUM_LAYERS, gw.D, gw.VOCAB, gw.PLI_D) != (N_LAYERS, D, VOCAB, PLI_D):
        raise RuntimeError(
            f"mlir-air's Gemma4-E4B dims moved ({gw.NUM_LAYERS} layers of "
            f"{gw.D}, vocab {gw.VOCAB}, pli {gw.PLI_D}); this example says "
            f"{N_LAYERS} of {D}, vocab {VOCAB}, pli {PLI_D}"
        )
    if (gw.FIRST_KV_SHARED_LAYER, gw.N_KV_HEADS) != (FIRST_KV_SHARED, N_KV_HEADS):
        raise RuntimeError(
            f"mlir-air shares KV from layer {gw.FIRST_KV_SHARED_LAYER} with "
            f"{gw.N_KV_HEADS} kv heads; this example says {FIRST_KV_SHARED} "
            f"with {N_KV_HEADS}"
        )

    src = model or MODEL_DEFAULT
    qm = gw.Q4nxModel(src)
    g = qm.globals()

    layers = []
    for k in range(N_LAYERS):
        inter = qm.mlp_inter(k)
        if inter != mlp_inter(k):
            raise RuntimeError(
                f"layer {k}: the bundle's FFN width is {inter}, this example "
                f"says {mlp_inter(k)}"
            )
        # Transposed to [K, out], as E2B's loader does.
        w = {nm: np.ascontiguousarray(t.T) for nm, t in qm.layer_weights(k).items()}
        ple = {nm: np.ascontiguousarray(t.T) for nm, t in qm.layer_ple(k).items()}
        nm = qm.layer_norms(k)
        layers.append(
            dict(
                attn_norm=nm["input"].astype(np.float32),
                post_attn_norm=nm["post_attn"].astype(np.float32),
                ffn_norm=nm["pre_ffn"].astype(np.float32),
                post_ffn_norm=nm["post_ffn"].astype(np.float32),
                post_ple_norm=nm["post_ple"].astype(np.float32),
                q_norm=nm["q_norm"].astype(np.float32),
                k_norm=nm["k_norm"].astype(np.float32),
                out_scale=float(nm["out_scale"]),
                **w,
                **ple,
            )
        )

    return dict(
        layers=layers,
        embed=qm.embed_rows("model.embed_tokens.weight", np.arange(VOCAB)),
        ple_rows=lambda ids: qm.embed_rows(
            "model.per_layer_token_embd.weight", np.asarray(ids)
        ).reshape(len(ids), N_LAYERS, PLI_D),
        final_norm=g["final_norm"].astype(np.float32),
        ple_proj_norm=g["ple_proj_norm"].astype(np.float32),
        lm_head=qm.lm_head_rows(0, VOCAB),
        rope_freqs=qm.rope_freqs(),
        fingerprint=_fingerprint(src),
    )


def rope_lut(seq_len, layer_idx, rope_freqs=None, dtype=None):
    """[seq_len, dh] = [cos.., sin..] per position, from mlir-air's generator,
    as on E2B."""
    import numpy as np

    _one = _weights().rope_lut
    if dtype is None:
        from ml_dtypes import bfloat16

        dtype = bfloat16
    rows = []
    for p in range(seq_len):
        cos, sin, _dh = _one(p, layer_idx, rope_freqs=rope_freqs)
        rows.append(np.concatenate([cos, sin]))
    return np.stack(rows).astype(dtype)


# Loaded here, before any driver: mlir-air's E4B wrapper modules put their own
# directory first on sys.path when they run, and the harness puts E2B's package
# back in front only once, after loading the driver (`extra_packages`).
_weights()
