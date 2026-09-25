# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E2B dims, the weight loader, and the RoPE tables.

The dims and the loader are mlir-air's -- there is no reason to re-derive
either. This module only locates mlir-air's LLM packages on sys.path and
re-exports what the prefill needs.

**This is the first model here whose layers are not all the same shape**, and
it is the reason its decode runs on a different engine (see
`llm_q4nx/registry.py::ModelSpec.engine`). Three things vary per layer:

* **attention type** -- four sliding layers then one full, repeating. Sliding
  layers have a 512-token window, a 256-wide head and RoPE theta 1e4; full
  layers have no window, a 512-wide head and theta 1e6 with a partial rotary.
* **who owns the KV** -- layers below 15 project their own k/v; the 20 above
  read a lower layer's cache and ship no usable k/v projection at all.
* **the FFN width** -- 6144 below layer 15, 12288 at and above it.

On top of that it carries **per-layer embeddings**: a 256-wide vector per layer
per token, computed once from the input embeddings and injected after the MLP
through a gated projection. `gemma4_prefill.py` implements all of it.
"""

import os
import sys

#: The harness every Q4NX example shares.
_SHARED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_q4nx"
)

#: This example's model, and the key into llm_q4nx/registry.py. The forward
#: asserts against it (llm_q4nx/llama_prefill.py::bound_model).
MODEL_NAME = "gemma4-e2b"

# Gemma4-E2B-IT. Mirrors mlir-air's gemma4_e2b_q4nx_weights; kept here so the
# kernels can be read without chasing an import.
D = 1536  # model dim
N_Q_HEADS = 8
N_KV_HEADS = 1  # MQA -- one kv head broadcast over all eight q heads
DH_SLIDING = 256  # head dim on a sliding layer
DH_GLOBAL = 512  # head dim on a full-attention layer
INTER = 6144  # mlp intermediate on the narrow layers
INTER_WIDE = 2 * INTER  # 12288, on layers >= FIRST_KV_SHARED
N_LAYERS = 35
VOCAB = 262144
PLI_D = 256  # per-layer input dim (hidden_size_per_layer_input)

#: Layers from here up read another layer's KV and carry the wide FFN.
FIRST_KV_SHARED = 15
#: Sliding-attention window, on the sliding layers only.
SLIDING_WINDOW = 512

# The deployed fused decode's epsilon (fused_decode/models/all_models.h).
RMS_EPS = 1e-6

#: RoPE bases, one per attention type.
ROPE_SLIDING_THETA = 10000.0
ROPE_GLOBAL_THETA = 1000000.0

#: The attention scale. **1.0, not `head_dim**-0.5`** -- this model's config
#: sets `query_pre_attn_scalar` to 1.0, and the usual scale would be 1/16 or
#: 1/22.6 here. Nothing errors if it is wrong; the softmax simply runs at the
#: wrong temperature and the text degrades.
ATTN_SCALE = 1.0

#: Logits are squashed through `cap * tanh(logits / cap)` before the argmax.
#: It is monotonic, so it cannot change which token wins -- recorded and applied
#: because the reference does, so our logits are comparable to its.
FINAL_LOGIT_SOFTCAP = 30.0

#: Gemma4 does **not** fold `1 + w` into its RMSNorm the way Gemma2 and Gemma3
#: do. The bundle ships raw HF weights and the device kernel multiplies by `w`
#: with no add. Stated here because the sibling Gemma3 example does fold it and
#: copying that is the obvious mistake: upstream measured the +1 taking layer 0
#: from cosine 0.986 to 0.962 and collapsing everything downstream.
RMS_NORM_ADDS_ONE = False

#: The two PLE combine scales. mlir-air's note is worth carrying: FLM's header
#: swaps these two NAMES relative to the reference implementation, so they are
#: keyed off the values rather than the names.
PLE_MODEL_PROJ_SCALE = float(D**-0.5)  # 0.0255
PLE_INPUT_SCALE = float(2.0**-0.5)  # 0.7071

#: The bundle has already folded Gemma's `sqrt(width)` embedding scale in, so
#: applying it here double-scales. Quiet when wrong -- RMSNorm is
#: scale-invariant, so only the residual stream is off, by 39x, which makes all
#: 35 layers a near-passthrough and yields fluent contentless text.
EMBED_SCALE = 1.0


#: Names `llm_q4nx/llama_prefill.py` imports at module scope, which every
#: config here must therefore define. On this model they are **not constants**
#: -- the head dim, and so the q/k/v widths, depend on the layer -- so they are
#: bound to the WIDEST layer's values and `Gemma4Prefill` reads none of them.
#: Widest rather than narrowest deliberately: the only thing the base class
#: does with them is size the KV cache, and oversizing that is wasted memory
#: where undersizing it is a silent truncation. The forward uses `head_dim(L)`.
DH = DH_GLOBAL
DQ = N_Q_HEADS * DH_GLOBAL
DK = DV = N_KV_HEADS * DH_GLOBAL
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 8 -- MQA, the widest ratio here


def is_sliding(layer_idx):
    """Four sliding layers then one full, repeating: `ssssF` x 7."""
    return (layer_idx % 5) != 4


def head_dim(layer_idx):
    return DH_SLIDING if is_sliding(layer_idx) else DH_GLOBAL


def mlp_inter(layer_idx):
    """This layer's FFN width. Read back against the bundle in `load_q4nx`."""
    return INTER if layer_idx < FIRST_KV_SHARED else INTER_WIDE


def owns_kv(layer_idx):
    return layer_idx < FIRST_KV_SHARED


def kv_source_layer(layer_idx):
    """Which layer's cache this one attends. Itself, below FIRST_KV_SHARED.

    Taken from mlir-air rather than restated: the sharing map is not a formula
    anyone should re-derive, and a wrong one produces fluent wrong text.
    """
    _add_air_paths()
    from gemma4_e2b_q4nx_weights import kv_source_layer as _src

    return int(_src(layer_idx))


# The Paris gate, from gemma4_e2b_q4nx_inference.py.
#
# `2` is <bos>, prepended by hand: the Gemma4 tokenizer adds none, and without
# it the model is out of distribution and its logits go near-flat. Every one of
# mlir-air's own entry points prepends it too, and the one place that does not
# -- their shared top-k verify runner -- is a known failure for exactly this
# reason. So the BOS is part of the prompt here, not an argument.
PROMPT = [2, 818, 5279, 529, 7001, 563]  # "<bos>The capital of France is"
EXPECT_FIRST = 9079  # " Paris"

#: Only `make_session_class`'s warmup turn uses this, and that path is reached
#: only by `--interactive`, which this model's driver does not support.
BOS = 2

#: What `--greedy` must generate from PROMPT, first token included --
#: mlir-air's own `PARIS_GREEDY` for this model. Only two tokens, where every
#: sibling here records ten: this is an instruction-tuned checkpoint that
#: answers " Paris." and stops, and upstream's own gate additionally requires
#: the run to STOP on 106 (<end_of_turn>) after them.
EXPECT_IDS = [9079, 236761]

#: Upstream's gate additionally requires the run to STOP on 106
#: (<end_of_turn>) after those two tokens, and ours does not: the shared
#: harness compares the leading ids and has no notion of where a generation
#: ended. Recorded as a KNOWN GAP rather than as a constant, because a
#: constant nothing reads is worse than none -- it reads as a check that is
#: running. Closing it means teaching the harness about the stop token, which
#: belongs with the first decode run on this model rather than ahead of it.

#: Stop tokens for the GPU decode loop. mlir-air's own gate requires the run
#: to stop on 106 (<end_of_turn>) after EXPECT_IDS; the NPU path leaves that to
#: its driver, and the GPU path here honours it directly.
EOS_IDS = (106, 1)  # <end_of_turn>, <eos>

MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Gemma4-E2B-IT-NPU2")


if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import airsrc  # noqa: E402

#: mlir-air llms packages this model needs on sys.path. Only its own: unlike
#: the Qwen2.5 pair, it borrows no config or RoPE table from a sibling.
AIR_PACKAGES = ("gemma4_e2b_q4nx",)


def _add_air_paths():
    airsrc.add_air_paths(*AIR_PACKAGES)


def _fingerprint(src):
    """Short digest of the bundle's safetensors header.

    The same rule as the other bundle-backed configs -- sha256 of the header
    bytes alone, never the multi-GB payload, first 8 hex digits. Falls back to
    the empty string rather than raising: a fingerprint is printed next to a
    result, and failing a run over one would be out of proportion.
    """
    import hashlib

    _add_air_paths()
    try:
        from gemma4_e2b_q4nx_weights import resolve_q4nx_model

        path = resolve_q4nx_model(src)
        with open(path, "rb") as f:
            hlen = int.from_bytes(f.read(8), "little")
            return hashlib.sha256(f.read(hlen)).hexdigest()[:8]
    except Exception:
        return ""


def load_q4nx(model=None):
    """Host-dequantized Gemma4-E2B weights.

    Dequantizing on the host is deliberate: prefill is compute-bound, so W4A16
    would cut memory traffic that is not the bottleneck and add unpack work to
    every GEMM tile.

    Returns a dict with

        layers: list of 35 dicts, each with
            attn_norm, post_attn_norm, ffn_norm, post_ffn_norm,
            post_ple_norm, q_norm, k_norm : float32
            out_scale                     : float (a scalar, not a vector)
            q, o, gate, up, down          : bfloat16 [K, out]  (y = x @ W)
            k, v                          : bfloat16, ONLY where owns_kv(L)
            inp_gate, per_layer_projection, model_proj : bfloat16, the PLE trio
        embed, per_layer_embed, final_norm, ple_proj_norm, lm_head, rope_freqs

    Three things differ from every sibling loader and each is load-bearing:

    * **k/v are absent on 20 of the 35 layers.** Those layers ship the tensors
      but the reference never evaluates them -- they attend a lower layer's
      cache. Omitted rather than loaded-and-ignored so a wrong sharing map is a
      KeyError instead of silently-unused weights, which is mlir-air's own
      choice here and worth keeping.
    * **the projections are not all one shape.** `q` is `[D, 8*dh]` with dh
      per layer, and `gate`/`up`/`down` are at that layer's FFN width. So the
      usual Q|K|V and gate|up fusions are done per layer rather than once.
    * **`out_scale` is a scalar** multiplying the whole block output. Missing
      it leaves every layer slightly too large and compounds with depth.

    **The LM head is not tied**, and it is this model's single largest tensor
    (262144 x 1536). `rope_freqs` is the bundle's partial-rotary divisor table;
    see `rope_lut`.
    """
    import numpy as np

    _add_air_paths()
    import gemma4_e2b_q4nx_weights as gw

    if (gw.NUM_LAYERS, gw.D, gw.VOCAB, gw.PLI_D) != (N_LAYERS, D, VOCAB, PLI_D):
        raise RuntimeError(
            f"mlir-air's Gemma4-E2B dims moved ({gw.NUM_LAYERS} layers of "
            f"{gw.D}, vocab {gw.VOCAB}, pli {gw.PLI_D}); this example says "
            f"{N_LAYERS} of {D}, vocab {VOCAB}, pli {PLI_D}"
        )
    if gw.FIRST_KV_SHARED_LAYER != FIRST_KV_SHARED:
        raise RuntimeError(
            f"mlir-air shares KV from layer {gw.FIRST_KV_SHARED_LAYER}; this "
            f"example says {FIRST_KV_SHARED}"
        )

    src = model or MODEL_DEFAULT
    qm = gw.Q4nxModel(src)
    g = qm.globals()

    layers = []
    for k in range(N_LAYERS):
        # The FFN width is read from the bundle's own chunk count rather than
        # assumed, and checked against the rule above -- upstream reads it for
        # the same reason, and a disagreement means the class map moved.
        inter = qm.mlp_inter(k)
        if inter != mlp_inter(k):
            raise RuntimeError(
                f"layer {k}: the bundle's FFN width is {inter}, this example "
                f"says {mlp_inter(k)}"
            )
        # **Transposed here.** mlir-air's accessors for this model return
        # `[out, in]` and its reference forward writes `x @ W.T`; every prefill
        # in `llm_q4nx/` takes `[K, out]` and writes `x @ W`. Gemma3's reader
        # transposes on the way out and this one does not, so the transpose has
        # to happen somewhere -- doing it once, here, keeps the forward reading
        # like its siblings. A missed one is not a silent failure: the GEMM
        # shapes stop matching.
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
        # The per-layer embedding table is [VOCAB, 35, 256] -- 2.3 billion
        # entries, which is why it is a gather rather than a tensor. Returned
        # as a callable so the prompt's rows can be fetched at prefill time
        # without keeping the whole table, or the reader, in the caller's head.
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
    """[seq_len, dh] = [cos_0..cos_{dh/2-1}, sin_0..sin_{dh/2-1}] per position.

    Per layer, because both the base and the width depend on the layer's
    attention type: theta 1e4 over 256 lanes on a sliding layer, theta 1e6 over
    512 on a full one.

    Built by calling mlir-air's own single-position generator once per position
    rather than by vectorizing it here: that generator is what the fused decode
    rotates each new token with, and a reimplementation that agrees today is a
    thing that can stop agreeing silently.

    **The partial rotary is in the table, not in the rotation.** The full
    layers rotate only 25% of their 512 lanes, and the bundle expresses that by
    shipping a frequency DIVISOR of ~1e30 for the dead entries -- which makes
    cos 1 and sin 0 there, an identity. So the pairing stays (i, i + dh/2) and
    the ordinary half-split `_rope` is correct; there is no partial-rotary
    special case to write, and writing one would pair the wrong lanes.
    """
    import numpy as np

    _add_air_paths()
    from gemma4_e2b_q4nx_weights import rope_lut as _one

    if dtype is None:
        from ml_dtypes import bfloat16

        dtype = bfloat16
    rows = []
    for p in range(seq_len):
        cos, sin, _dh = _one(p, layer_idx, rope_freqs=rope_freqs)
        rows.append(np.concatenate([cos, sin]))
    return np.stack(rows).astype(dtype)
