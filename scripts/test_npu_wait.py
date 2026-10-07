# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""`npu_wait` builds, and pyxrt hands it the xrt::run it waits on.

MultiLaunchRunner falls back to pyxrt's own wait, which holds the GIL, when
either fails, and nothing else would notice: results stay right and only
concurrency is lost. Needs XRT and g++, not an NPU.
"""

import sys

from triton.backends.amd_triton_npu.driver import npu_wait_module


def main():
    mod = npu_wait_module()
    if mod is None:
        print("FAIL: npu_wait did not build")
        return 1
    import pyxrt

    if not mod.supported(pyxrt.run()):
        print("FAIL: pyxrt declined to hand over its xrt::run (C++ ABI differs)")
        return 1
    if mod.supported(object()):
        print("FAIL: a non-run object was accepted")
        return 1
    try:
        mod.wait(object())
    except TypeError:
        pass
    else:
        print("FAIL: wait() accepted a non-run object")
        return 1
    print("PASS: npu_wait")
    return 0


if __name__ == "__main__":
    sys.exit(main())
