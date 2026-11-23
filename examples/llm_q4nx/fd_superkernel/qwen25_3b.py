# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Qwen2.5-3B's superkernel decode: the registry's entry for it."""

from . import select

select("qwen2.5-3b")

from .decoder import SuperkernelDecoder  # noqa: E402

__all__ = ["SuperkernelDecoder"]
