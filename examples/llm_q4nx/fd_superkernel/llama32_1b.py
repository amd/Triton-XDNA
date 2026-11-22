# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Llama-3.2-1B's superkernel decode: the registry's entry for it. Its
mlir-air driver constructs the decoder with the sequence length alone."""

import os

os.environ["FD_SUPERKERNEL_MODEL"] = "llama-3.2-1b"

from .decoder import SuperkernelDecoder as _Decoder  # noqa: E402


class SuperkernelDecoder(_Decoder):
    def __init__(self, max_L=None, verbose=True):
        super().__init__(max_L=max_L, verbose=verbose)
