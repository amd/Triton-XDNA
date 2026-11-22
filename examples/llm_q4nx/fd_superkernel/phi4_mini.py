# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Phi-4-mini's superkernel decode: the registry's entry for it, with the
interface of mlir-air's prefiller drivers (FusedDecodePhi4's): constructed
from the model and the decode's directory, its KV seeded a layer at a time."""

import os

os.environ["FD_SUPERKERNEL_MODEL"] = "phi4-mini"

from .decoder import PrefillerDecoder as SuperkernelDecoder  # noqa: E402

__all__ = ["SuperkernelDecoder"]
