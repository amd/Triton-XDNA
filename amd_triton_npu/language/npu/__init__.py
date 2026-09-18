# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""NPU-specific Triton language ops: ``tl.extra.npu``.

Installed by Triton's own build. `setup.py`'s `BackendInstaller.prepare` looks
for a `language/` directory beside every backend's `backend/`, and it does that
for plugins from `TRITON_PLUGIN_DIRS` exactly as for in-tree backends; each
subdirectory found there is installed as `triton.language.extra.<name>`. So this
package needs no patch to Triton and no registration call -- `third_party/amd`'s
`language/hip` reaches `tl.extra.hip` the same way.

What lives here is the ops that are real on this device and have no portable
tile semantics. `tl.extra.cuda.libdevice` is the precedent: not a tile
primitive, still a language op.
"""

from .fused_decode import DecodeConfig, fused_decode

__all__ = ["DecodeConfig", "fused_decode"]
