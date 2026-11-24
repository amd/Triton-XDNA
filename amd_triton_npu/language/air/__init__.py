# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""MLIR-AIR's dataflow in Triton: ``tl.extra.air``.

Installed like ``tl.extra.npu``, from this backend's ``language/`` directory.
These ops lower to AIR (herds, channels, L2 buffers), which is not specific
to one device; see ``superkernel.py``. A core's code may call aie_api's
operations one by one: ``from triton.language.extra.air import aie``.
That module is not imported with this one: its functions are compiled by
``triton.jit`` when it is imported, which needs ``triton.language`` loaded.
"""

from .superkernel import (
    L1,
    L2,
    Channel,
    ChannelGet,
    ChannelPut,
    alloc,
    backend,
    herd,
    herd_id,
    table,
)

__all__ = [
    "Channel",
    "ChannelGet",
    "ChannelPut",
    "L1",
    "L2",
    "alloc",
    "backend",
    "herd",
    "herd_id",
    "table",
]
