# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""MLIR-AIR's dataflow in Triton: ``tl.extra.air``.

Installed like ``tl.extra.npu``, from this backend's ``language/`` directory.
These ops lower to AIR (herds, channels, L2 buffers), which is not specific
to one device; see ``superkernel.py``.
"""

from .superkernel import (
    Channel,
    ChannelGet,
    ChannelPut,
    allocate_l2,
    spatial,
    table,
)

__all__ = ["Channel", "ChannelGet", "ChannelPut", "allocate_l2", "spatial", "table"]
