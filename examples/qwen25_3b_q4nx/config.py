# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen2.5-3B dims, the weight loader, and the RoPE table.

The dims and the loader are mlir-air's -- there is no reason to re-derive
either. This module only locates mlir-air's LLM packages on sys.path and
re-exports what the prefill needs.

Qwen2.5-7B's config at 2048, running the same forward
(`../llm_q4nx/qwen25_prefill.py`): the q/k/v projection bias, no qk-norm. Two
differences from its sibling, and neither is in the block -- the weight source
is a bundle by default rather than a checkpoint, and the decode runs with the
dual-channel weight feed OFF.
"""

import os
import sys

#: The harness every Q4NX example shares.
_SHARED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_q4nx"
)

#: This example's model, and the key into llm_q4nx/registry.py. The forward
#: asserts against it (llm_q4nx/llama_prefill.py::bound_model).
MODEL_NAME = "qwen2.5-3b"

# Qwen2.5-3B-Instruct. Mirrors mlir-air's qwen25_3b_q4_weights; kept here so
# the kernels can be read without chasing an import.
D = 2048  # model dim
DH = 128  # head dim
N_Q_HEADS = 16
N_KV_HEADS = 2
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 8 -- the widest GQA ratio here
DQ = N_Q_HEADS * DH  # 2048 -- equal to D, so o_proj is square
DK = N_KV_HEADS * DH  # 256 -- the narrowest KV here
DV = DK
INTER = 11008  # mlp intermediate, LOGICAL
N_LAYERS = 36
VOCAB = 151936

#: The device pads the GLU dim to a multiple of 512, so the decode's own
#: buffers are 11264 wide. That padding is mlir-air's and stays there:
#: `layer_weights` unpads on the way out, so everything below -- and the whole
#: forward -- is at the logical 11008. Recorded because a reader comparing
#: these dims against the decode's headers will meet the other number.
INTER_DEVICE_PAD = 11264

# The deployed fused decode's epsilon (fused_decode/models/all_models.h), also
# what mlir-air's Qwen2.5 reference forward uses.
RMS_EPS = 1e-6

#: RoPE base. Single theta, no llama3 frequency scaling and no second theta.
ROPE_THETA = 1000000.0

#: The shared hand-written matmul schedule serves every GEMM here, unlike
#: Qwen2.5-7B, which needs the driver-generated one. The reason is width: this
#: model's fused `gate_up` is 2*11008 = 22016 columns against the 7B's 37888,
#: and 32768 is the widest measured to lower under the shared schedule.
#: `llm_q4nx/qwen25_prefill.py` explains the flag; stated here explicitly
#: rather than left to the default, because it is a property of these dims.
MATMUL_GENERATED_SCHEDULE = False

# The Paris gate, from qwen25_3b_q4_inference.py. Qwen2.5 emits no BOS, and it
# shares Qwen3's ids for these tokens.
PROMPT = [785, 6722, 315, 9625, 374]  # "The capital of France is"
EXPECT_FIRST = 12095  # " Paris"

#: Only `make_session_class`'s warmup turn uses this, and that path is reached
#: only by `--interactive`, which this model's driver does not support. Qwen2.5
#: has no BOS, so this is <|endoftext|> standing in for one.
BOS = 151643

#: What `--greedy` must generate from PROMPT, first token included.
#: (" Paris. The capital of Italy is Rome. The")
#:
#: **Not** mlir-air's recorded `PARIS_GREEDY` for this model, which is
#: `[12095, 13, 12095, 374, 279, 7772, 3283, 304, 9625, 13]` and does not
#: reproduce here. What is recorded instead is what mlir-air's own **numpy
#: oracle** prefill produces through this same decode -- `generate(...,
#: numpy_prefill=True)`, which seeds the KV from `forward_prompt` with no NPU
#: prefill and no Triton code involved. This example reproduces that token for
#: token.
#:
#: The artifacts are not in question: our `decode_L2048.insts.bin` is
#: byte-identical to what mlir-air's own builder emits for this model under
#: this environment, checked at the L the run actually uses as well as at 16.
#: So their oracle and their decode, on their own instruction stream, disagree
#: with their own recorded constant -- which says the constant was recorded
#: under a configuration that is not this one (`W_DUAL_CHAN` is the obvious
#: candidate: this is the only model that sets it to 0, and their Makefile
#: warns that a warm template built with the other setting must not be reused).
#:
#: Left as a golden rather than dropped, because it is still the only
#: assertion here that covers the decode at all: the first-token gate is pure
#: prefill and cannot see the builder environment, -DMODEL_TYPE, GLU_SLICE or
#: W_DUAL_CHAN -- and W_DUAL_CHAN is exactly what this model gets wrong if the
#: harness stops propagating it (see `harness.air_inference_module`).
EXPECT_IDS = [12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576]

MODEL_DEFAULT = os.environ.get(
    "Q4NX_MODEL_SOURCE", "FastFlowLM/Qwen2.5-3B-Instruct-NPU2"
)


if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import airsrc  # noqa: E402

#: mlir-air llms packages this model needs on sys.path: its own package, and
#: `qwen25_3b`, which holds the `LlamaConfig` and the weight containers. Note
#: the first is `qwen25_3b_q4`, not `_q4nx` -- that name is about the weight
#: codec, not the decode, which is the same Q4NX engine as every other model
#: here.
AIR_PACKAGES = ("qwen25_3b_q4", "qwen25_3b")


def _add_air_paths():
    airsrc.add_air_paths(*AIR_PACKAGES)


def _fingerprint(src):
    """Short digest of the weight source's safetensors header(s).

    The same rule as the bundle-backed configs -- sha256 of header bytes alone,
    never the multi-GB payload, first 8 hex digits -- over whichever files the
    source resolves to, since this model accepts either a `model.q4nx` bundle
    or a sharded HF checkpoint. Falls back to the empty string rather than
    raising: a fingerprint is printed next to a result, and failing a run over
    one would be out of proportion.
    """
    import hashlib

    _add_air_paths()
    try:
        from qwen25_3b_q4_weights import resolve_q4nx_model

        files = [resolve_q4nx_model(src)]
    except Exception:
        # Not a bundle: an HF checkpoint, resolved the way its reader does.
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
    """Host-dequantized Qwen2.5-3B weights.

    Dequantizing on the host is deliberate: prefill is compute-bound, so W4A16
    would cut memory traffic that is not the bottleneck and add unpack work to
    every GEMM tile.

    `open_weight_source` decides what it is reading: a `model.q4nx` bundle --
    the default, and what `MODEL_DEFAULT` points at -- or an HF checkpoint,
    Q4_0-quantized on load. Both expose the same accessors, so nothing below
    branches on which one it got.

    **The LM head is not tied here, despite the config saying it is.** Qwen2.5
    ties `lm_head` to the embedding, but the bundle's bf16 embedding has itself
    been through a 4-bit round trip, so mlir-air's reader dequantizes the
    shipped `lm_head` instead of reusing the embedding -- the tied shortcut
    would gain nothing and lose accuracy. This follows the reader, not the HF
    config.

    Returns a dict with
        layers: list of 36 dicts, each with
            attn_norm, ffn_norm : float32 [D]
            qkv_bias            : float32 [DQ+DK+DV]  (Qwen2.5's projection bias)
            q, k, v, o, gate, up, down : bfloat16 [K, out]  (y = x @ W)
        embed, final_norm, lm_head : float32   (lm_head is its own tensor)

    `qkv_bias` arrives concatenated, in q, k, v order, to match the fused QKV
    GEMM `load_weights` builds -- see the 7B's `load_q4nx`, which says the same
    and for the same reason.

    The GLU projections come back at the LOGICAL `INTER`; `layer_weights`
    unpads the device's 11264 on the way out, so no caller here sees it.
    """
    import numpy as np

    _add_air_paths()
    import qwen25_3b_q4_weights as gw

    if (gw.NUM_LAYERS, gw.D, gw.INTER, gw.VOCAB) != (N_LAYERS, D, INTER, VOCAB):
        raise RuntimeError(
            f"mlir-air's Qwen2.5-3B dims moved ({gw.NUM_LAYERS} layers of {gw.D}, "
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
    stop agreeing silently.

    Single theta (1e6) and no llama3 frequency scaling.
    """
    import numpy as np

    _add_air_paths()
    from qwen25_3b_q4_weights import generate_rope_lut

    if dtype is None:
        from ml_dtypes import bfloat16

        dtype = bfloat16
    lut = np.stack([generate_rope_lut(p) for p in range(seq_len)])
    return lut.astype(dtype)
