# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Qwen3-8B's superkernel decode: the registry's entry for it."""

import os

os.environ["FD_SUPERKERNEL_MODEL"] = "qwen3-8b"

from .decoder import SuperkernelDecoder  # noqa: E402

__all__ = ["SuperkernelDecoder"]
