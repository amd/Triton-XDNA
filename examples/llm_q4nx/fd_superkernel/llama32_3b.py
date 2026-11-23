# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Llama-3.2-3B's superkernel decode: the registry's entry for it, with the
interface of mlir-air's prefiller drivers (FusedDecode3B's): constructed
from the model and the decode's directory, its KV seeded a layer at a time."""

from . import select

select("llama-3.2-3b")

from .decoder import PrefillerDecoder as SuperkernelDecoder  # noqa: E402

__all__ = ["SuperkernelDecoder"]
