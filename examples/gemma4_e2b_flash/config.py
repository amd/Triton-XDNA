# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E2B "flash": the same checkpoint as the text example, plus an image.

**This is not a second model.** `gemma4e-flash:e2b` and `gemma4-it:e2b` are the
same weights, the same 35 layers and the same decode; in FastFlowLM they differ
only in the engine that runs prefill (one `fused_prefill.xclbin` covering the
whole array instead of three overlays reconfigured per layer) and in two config
numbers, `max_prefill_len` and `default_context_length`, both 1024 on the flash
path. Everything about the language model is therefore imported from
`../gemma4_e2b_q4nx/config.py` rather than restated here.

What this directory adds is the part Triton-XDNA did not have: the **vision
tower**, and the plumbing to put an image in front of the prompt.

The model id is `gemma4e-flash:e2b`. `FLM_Xclbin/kernel_assignment.md` says
`gemma4-flash:e2b`, which is wrong; FLM does not error on an unknown id, it
falls back to Llama-3.2-1B and answers in fluent English from the wrong model.
Not something this file can be wrong about -- nothing here talks to FLM -- but
worth carrying next to the name.

Only the image half is implemented. The checkpoint also carries an audio tower
(`audio_weight.q4nx`: 12 layers, conv1d/conv2d subsampling, chunked local
attention) which shares nothing with this but the final splice, and video,
which shares the tower but not the preprocessing.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

#: The text example whose config this one is. Loaded by path under a distinct
#: module name: both directories name their config `config`, so a plain import
#: would resolve to whichever is first on `sys.path` -- here, always this one.
_TEXT_DIR = os.path.join(os.path.dirname(_HERE), "gemma4_e2b_q4nx")


def _load_text_config():
    spec = importlib.util.spec_from_file_location(
        "gemma4_e2b_text_config", os.path.join(_TEXT_DIR, "config.py")
    )
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec: the sibling puts the shared harness on sys.path
    # during import, and anything it imports transitively should see it.
    sys.modules[spec.name] = mod
    if _TEXT_DIR not in sys.path:
        sys.path.insert(0, _TEXT_DIR)
    spec.loader.exec_module(mod)
    return mod


text = _load_text_config()

#: Re-export the language model wholesale. `_SHARED` and `_add_air_paths` come
#: with it -- `prefill.py` and the inference driver reach for both -- which is
#: why the underscore names are not filtered out. Re-exporting rather than
#: proxying through `text.` keeps this module a drop-in `config` for the shared
#: harness, which reads attributes off it by name.
globals().update({k: v for k, v in vars(text).items() if not k.startswith("__")})

#: The key into `llm_q4nx/registry.py` stays the TEXT model's. The decode is
#: byte-identical -- same 35 layers, same PLE engine, same template -- so a
#: second spec would be a second name for one build.
MODEL_NAME = text.MODEL_NAME

# ---------------------------------------------------------------- vision
# From `vision_config` in the checkpoint's config.json. Restated here rather
# than parsed so the tower can be read without a model directory in hand;
# `llm_q4nx/gemma4_vision_weights.py` asserts the weights agree with them.
VISION_HIDDEN = 768
VISION_N_LAYERS = 16
VISION_N_HEADS = 12
VISION_HEAD_DIM = 64
VISION_INTER = 3072
VISION_PATCH = 16
VISION_ROPE_THETA = 100.0
VISION_POSITION_EMBEDDING_SIZE = 10240

#: Patches are averaged over `3 x 3` tiles into one soft token.
POOLING_KERNEL_SIZE = 3

#: `default_output_length`: the soft-token budget the image processor pads to.
#: A 448x448 image resizes to 768x768 = 2304 real patches = 256 soft tokens,
#: padded to 2520 patches; the padding is dropped after pooling.
MAX_SOFT_TOKENS = 280

#: The MM tile FLM's converter packed the vision weights against. Not a tuning
#: knob: it is how the bytes in `vision_weight.q4nx` are laid out.
VISION_MM_TILE_N = 64
VISION_MM_TILE_K = 256

# ---------------------------------------------------------------- tokens
#: The placeholder the vision tower's soft tokens replace, one per token.
IMAGE_TOKEN_ID = 258880
#: ...bracketed by these in the rendered chat template.
BOI_TOKEN_ID = 255999
EOI_TOKEN_ID = 258882

#: Substituted at every image position before the PER-LAYER EMBEDDING table is
#: gathered -- see `llm_q4nx/gemma4_prefill.py::prefill`. The placeholder id is
#: a vocabulary entry that never occurs in text, so its own PLE row is not what
#: the reference feeds the layers; the pad token's is.
PAD_TOKEN_ID = 0


def load_vision(model=None):
    """The de-tiled vision tower weights. See `llm_q4nx/gemma4_vision_weights.py`.

    A separate 337 MB `vision_weight.q4nx` in the same HF repo as `model.q4nx`
    -- the text bundle carries no vision tensor at all.
    """
    if text._SHARED not in sys.path:
        sys.path.insert(0, text._SHARED)
    import gemma4_vision_weights

    return gemma4_vision_weights.load_vision(model)
