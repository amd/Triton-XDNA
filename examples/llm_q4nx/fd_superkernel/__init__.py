# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The decode of the models air-FLM's shared `fused_decode` builder serves, as
one Triton superkernel dispatch a token. A model's module (`qwen3_4b`,
`gemma3_4b`, ...) is its entry point: importing it selects the model
(`select`), whose sizes (`geom`) and kernel (`wave`) the package then serves.
The sizes are module constants, so a process serves one model.
"""

_model = None


def select(name):
    """Make `name` (a key of `spec.MODELS`) the package's model. Selecting
    another one after it raises: `spec`, `geom` and `wave` are read once."""
    global _model
    if _model is not None and _model != name:
        raise RuntimeError(
            f"fd_superkernel serves {_model} in this process; it cannot also "
            f"serve {name}"
        )
    _model = name
