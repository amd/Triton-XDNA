# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""A Gemma4 model's decode as one Triton superkernel; see `decoder.py`."""

from .decoder import SuperkernelDecoder

__all__ = ["SuperkernelDecoder"]
