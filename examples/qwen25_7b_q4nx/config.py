# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen2.5-7B dims, the weight loader, and the RoPE table.

The dims and the loader are mlir-air's -- there is no reason to re-derive
either. This module only locates mlir-air's LLM packages on sys.path and
re-exports what the prefill needs.

Same shape as the other configs here, with one thing none of them have: the
**weight source is not a Q4NX bundle**. There is no FastFlowLM Qwen2.5-7B NPU2
bundle to download, so mlir-air quantizes an ungated upstream HF checkpoint on
load, and `load_q4nx` goes through its `open_weight_source` rather than naming
a reader. See that function's docstring below.

What differs in the model is Qwen2.5's, and only Qwen2.5's: a bias on each of
the q, k and v projections. No qk-norm -- that is Qwen3.
"""

import os
import sys

#: The harness every Q4NX example shares.
_SHARED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_q4nx"
)

#: This example's model, and the key into llm_q4nx/registry.py. The forward
#: asserts against it (llm_q4nx/llama_prefill.py::bound_model).
MODEL_NAME = "qwen2.5-7b"

# Qwen2.5-7B-Instruct. Mirrors mlir-air's qwen25_7b_q4nx_weights; kept here so
# the kernels can be read without chasing an import.
D = 3584  # model dim
DH = 128  # head dim
N_Q_HEADS = 28
N_KV_HEADS = 4
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 7 -- the widest GQA ratio here
DQ = N_Q_HEADS * DH  # 3584 -- equal to D, so o_proj is square
DK = N_KV_HEADS * DH  # 512
DV = DK
INTER = 18944  # mlp intermediate -- the largest here, 5.3x the model dim
N_LAYERS = 28
VOCAB = 152064

# The deployed fused decode's epsilon (fused_decode/models/all_models.h), also
# what mlir-air's Qwen2.5 reference forward uses.
RMS_EPS = 1e-6

#: RoPE base. Single theta, no llama3 frequency scaling and no second theta.
#: `rope_lut` gets this from mlir-air rather than from here; the value is
#: restated for the reader.
ROPE_THETA = 1000000.0

# The Paris gate, from qwen25_7b_q4nx_inference.py. The Qwen2.5 tokenizer emits
# no BOS, and it shares Qwen3's vocabulary for these tokens, so the prompt ids
# and the expected first token are the same as Qwen3's -- which is a
# coincidence of tokenizer lineage, not something to assume for a new model.
PROMPT = [785, 6722, 315, 9625, 374]  # "The capital of France is"
EXPECT_FIRST = 12095  # " Paris"

#: Only `make_session_class`'s warmup turn uses this, and that path is reached
#: only by `--interactive`, which this model's driver does not support. Named
#: so the harness has something to warm with if it ever does: Qwen2.5 has no
#: BOS, so this is <|endoftext|> standing in for one.
BOS = 151643

#: What `--greedy` must generate from PROMPT, first token included --
#: mlir-air's own PARIS_GREEDY for this model, which it recorded on NPU2
#: (" Paris. It is located in the northern part of"). It is the only assertion
#: that covers the decode at all; the first-token gate above is pure prefill
#: and cannot see the builder environment, -DMODEL_TYPE, GLU_SLICE or the core
#: stack.
#:
#: It diverges from the other models' after two tokens: this is an Instruct
#: checkpoint answering, where the bundle-backed models continue the pattern
#: with another capital.
EXPECT_IDS = [12095, 13, 1084, 374, 7407, 304, 279, 18172, 949, 315]

MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "Qwen/Qwen2.5-7B-Instruct")

#: Lower the model-dim GEMMs with the driver-generated matmul schedule rather
#: than the shared hand-written one. Forced by this model's fused `gate_up`:
#: 2*18944 = 37888 columns, for which the shared schedule emits a DMA stride of
#: 2424832 against a hardware range of [1, 1048576]. Measured -- N=32768 lowers
#: under it and N=37888 does not. `llm_q4nx/qwen25_prefill.py` explains why the
#: flag covers `qkv` and `o` too, and why `down` is excluded.
MATMUL_GENERATED_SCHEDULE = True


if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import airsrc  # noqa: E402

#: mlir-air llms packages this model needs on sys.path: its own q4nx package,
#: which holds the readers and the reference forward, and `qwen25_3b`, which
#: holds the `LlamaConfig` the dims are read back against. mlir-air's own
#: module reaches the latter the same way, by inserting it at import.
AIR_PACKAGES = ("qwen25_7b_q4nx", "qwen25_3b")


def _add_air_paths():
    airsrc.add_air_paths(*AIR_PACKAGES)


def _fingerprint(src):
    """Short digest of the weight source's safetensors header(s).

    Same rule as the bundle-backed configs -- sha256 of header bytes alone,
    never the multi-GB payload, first 8 hex digits -- widened to cover a
    sharded HF checkpoint, which is what this model's default source is. The
    shards are hashed in resolved order, so it is stable across runs and
    changes whenever the checkpoint is re-exported.

    Falls back to the empty string rather than raising: a fingerprint is
    something printed next to a result, and failing the run over one would be
    out of proportion.
    """
    import hashlib

    _add_air_paths()
    # `q4_0_codec` lives in `fused_decode/`, which is not one of the `llms/`
    # packages `_add_air_paths` puts on sys.path -- mlir-air's own weights
    # module inserts it at import time. Do the same rather than relying on
    # having imported that module first, which is an ordering this function
    # does not otherwise depend on.
    if airsrc.fused_decode_dir() not in sys.path:
        sys.path.insert(0, airsrc.fused_decode_dir())
    try:
        from q4_0_codec import HFModel

        files = HFModel._resolve(src)
    except Exception:
        return ""
    h = hashlib.sha256()
    for path in files:
        try:
            with open(path, "rb") as f:
                hlen = int.from_bytes(f.read(8), "little")
                h.update(f.read(hlen))
        except OSError:
            return ""
    return h.hexdigest()[:8]


def load_q4nx(model=None):
    """Host Qwen2.5-7B weights, on the Q4NX grid, from an HF checkpoint.

    Every other example here dequantizes a `model.q4nx` bundle. This one has no
    bundle to dequantize: FastFlowLM's Qwen2.5 line stops at 3B, and that
    converter's output uses a nibble interleave the Llama/Qwen3 bundles do not,
    so it would not be interchangeable even if it existed. mlir-air's answer is
    to round an ungated upstream checkpoint onto the Q4NX grid at load time,
    through `quantize_dequantize_q4nx` -- the same quantizer that builds the
    decode's cascade cache, so prefill and decode see bit-identical weights.

    `open_weight_source` is what decides: a path that really resolves to a
    `model.q4nx` gets the bundle reader, anything else -- a repo id, a
    checkpoint directory -- gets quantized on load. Both expose the same
    accessors, so nothing below branches on which one it got. It costs: the
    quantizer runs over every projection of every layer, which is most of this
    model's load time.

    **Qwen2.5-7B does not tie its LM head**, and the biases are not quantized
    on either side -- the reference design leaves them, like the norms, in
    bf16.

    Returns a dict with
        layers: list of 28 dicts, each with
            attn_norm, ffn_norm : float32 [D]
            qkv_bias            : float32 [DQ+DK+DV]  (Qwen2.5's projection bias)
            q, k, v, o, gate, up, down : bfloat16 [K, out]  (y = x @ W)
        embed, final_norm, lm_head : float32   (lm_head is its own tensor)

    `qkv_bias` arrives concatenated rather than as three tensors because the
    projection it corrects is itself fused into one GEMM by `load_weights`;
    splitting it here only to re-join it there would be a layout the forward
    never wants. The order is q, k, v -- the same order the QKV weight is
    concatenated in, and a permutation of it produces fluent wrong text.
    """
    import numpy as np

    _add_air_paths()
    import qwen25_7b_q4nx_weights as gw

    if (gw.NUM_LAYERS, gw.D, gw.INTER, gw.VOCAB) != (N_LAYERS, D, INTER, VOCAB):
        raise RuntimeError(
            f"mlir-air's Qwen2.5-7B dims moved ({gw.NUM_LAYERS} layers of {gw.D}, "
            f"inter {gw.INTER}, vocab {gw.VOCAB}); this example says "
            f"{N_LAYERS} of {D}, inter {INTER}, vocab {VOCAB}"
        )

    src = model or MODEL_DEFAULT
    qm = gw.open_weight_source(src)
    embed, final_norm, lm_head = qm.embed_norm_lmhead()

    layers = []
    for k in range(N_LAYERS):
        w = qm.layer_weights(k)
        attn_norm, ffn_norm = qm.layer_rms(k)
        bq, bk, bv = qm.layer_qkv_bias(k)
        layers.append(
            dict(
                attn_norm=attn_norm,
                ffn_norm=ffn_norm,
                qkv_bias=np.concatenate([bq, bk, bv]).astype(np.float32),
                **w,
            )
        )
    return dict(
        layers=layers,
        embed=embed,
        final_norm=final_norm,
        lm_head=lm_head,
        fingerprint=_fingerprint(src),
    )


def rope_lut(seq_len, dtype=None):
    """[seq_len, DH] = [cos_0..cos_63, sin_0..sin_63] per position.

    Built by calling mlir-air's own single-position generator once per
    position, rather than by vectorizing it here: that generator is what the
    fused decode rotates each new token with, including its bf16 rounding of
    cos and sin, and a reimplementation that agrees today is a thing that can
    stop agreeing silently. `seq_len` is a prompt length, so the loop is
    thousands of iterations at most.

    Single theta (1e6) and no llama3 frequency scaling. The decode's rope
    kernel also adds the q/k/v bias out of this same slab, but that is the
    decode's packing -- here the bias travels with the weights.
    """
    import numpy as np

    _add_air_paths()
    from qwen25_7b_q4nx_weights import generate_rope_lut

    if dtype is None:
        from ml_dtypes import bfloat16

        dtype = bfloat16
    lut = np.stack([generate_rope_lut(p) for p in range(seq_len)])
    return lut.astype(dtype)
