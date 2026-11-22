# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Llama-3.1-8B's superkernel decode: the registry's entry for it, with the
interface of mlir-air's prefiller drivers (FusedDecode8B's): constructed
from the model and the decode's directory, its KV seeded a layer at a time."""

import os

os.environ["FD_SUPERKERNEL_MODEL"] = "llama-3.1-8b"

from .decoder import PrefillerDecoder as SuperkernelDecoder  # noqa: E402

__all__ = ["SuperkernelDecoder"]
