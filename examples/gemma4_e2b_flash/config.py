# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E2B "flash": the text example's model, with an image in front.

`gemma4e-flash:e2b` and `gemma4-it:e2b` are the same weights, the same layers
and the same decode; in FastFlowLM they differ only in the engine that runs
prefill. Everything about the language model is therefore the text example's
(`../gemma4_e2b_q4nx/config.py`), re-exported here wholesale, and the decode
build is that example's too. The vision tower ships beside the text bundle as
`vision_weight.q4nx`; `../llm_q4nx/gemma4_vision_q4nx.py` reads it.
"""

import importlib.util
import os
import sys

_TEXT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gemma4_e2b_q4nx"
)


def _load_text_config():
    # By path, under its own name: both directories call their config `config`.
    spec = importlib.util.spec_from_file_location(
        "gemma4_e2b_text_config", os.path.join(_TEXT_DIR, "config.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


text = _load_text_config()

# Underscore names included: the shared harness reads `_SHARED` and the
# loaders off this module by name.
globals().update({k: v for k, v in vars(text).items() if not k.startswith("__")})
