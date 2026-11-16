# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Gemma4-E2B's decode as one Triton superkernel; see `decoder.py`."""

from .decoder import SuperkernelDecoder

__all__ = ["SuperkernelDecoder"]
