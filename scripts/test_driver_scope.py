# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""`driver_scope` switches Triton's active driver and puts back exactly what
was there.

Each check is a bug one of the copies it replaced had:

- a first scope in a process left the NPU driver active behind it, so the next
  iGPU kernel compiled for the NPU;
- reading `.active` to find what to restore resolved the auto-detected default,
  which raises "0 active drivers" on an iGPU-free host;
- the iGPU scope built a new AMD driver on every entry.

Needs neither an NPU nor an iGPU; the AMD checks run only where torch sees one.
"""

import sys

import triton
from triton.backends.amd_triton_npu.driver import NPUDriver
from triton.backends.amd_triton_npu.driver_scope import driver_scope

config = triton.runtime.driver
failures = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def main():
    config._active = None

    with driver_scope("npu") as d:
        check(isinstance(d, NPUDriver), "npu scope makes an NPUDriver active")
        check(config._active is d, "the yielded driver is the active one")
    check(config._active is None, "nothing active before -> nothing after")

    outer = NPUDriver()
    config.set_active(outer)
    with driver_scope("npu") as d:
        check(d is outer, "an NPU driver already active is kept, not replaced")
    check(config._active is outer, "and is still active after")

    try:
        driver_scope("cuda").__enter__()
        check(False, "an unknown kind is refused")
    except ValueError:
        check(True, "an unknown kind is refused")

    try:
        with driver_scope("npu"):
            raise KeyError("inside")
    except KeyError:
        pass
    check(config._active is outer, "restored when the body raises")

    try:
        import torch

        has_gpu = torch.cuda.is_available() and torch.version.hip is not None
    except ImportError:
        has_gpu = False
    if not has_gpu:
        print("  skip AMD checks: no iGPU visible to torch")
    else:
        config._active = None
        with driver_scope("amd") as a1:
            with driver_scope("npu") as n:
                check(isinstance(n, NPUDriver), "npu nested inside amd")
            check(config._active is a1, "amd restored after the nested npu")
        check(config._active is None, "amd scope restores nothing-active")
        with driver_scope("amd") as a2:
            pass
        check(a1 is a2, "the AMD driver is built once and reused")

    config._active = None
    if failures:
        print(f"FAIL: {len(failures)} check(s)")
        return 1
    print("PASS: driver_scope")
    return 0


if __name__ == "__main__":
    sys.exit(main())
