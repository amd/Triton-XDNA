# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Choose which device Triton compiles and launches for, within a scope.

Triton holds one active driver per thread and every launch goes to it, so a
program that runs kernels on both the NPU and the iGPU has to switch it
around each group of launches:

    with driver_scope("npu"):
        kernel[grid](...)          # compiled and launched for the NPU

    with driver_scope("gpu"):
        gpu_kernel[grid](...)      # compiled and launched for the iGPU

This module does not import `driver.py` at load time, so iGPU-only callers do
not pull in the NPU toolchain.
"""

import contextlib

# "gpu" is the AMD iGPU; Triton registers its driver as the "amd" backend.
_KINDS = ("npu", "gpu")

# The iGPU driver is built once and reused: constructing one is far more
# expensive than switching to it. The NPU driver is built on each entry
# because `NPUDriver()` reads the launch runtime from the environment.
_gpu = None


def _is_kind(driver, kind):
    if driver is None:
        return False
    if kind == "npu":
        from .driver import NPUDriver

        return isinstance(driver, NPUDriver)
    return type(driver).__module__ == "triton.backends.amd.driver"


def _make(kind):
    global _gpu
    if kind == "npu":
        from .driver import NPUDriver

        return NPUDriver()
    if _gpu is None:
        from triton.backends import backends

        _gpu = backends["amd"].driver()
    return _gpu


@contextlib.contextmanager
def driver_scope(kind):
    """Make the `kind` driver ("npu" or "gpu") active for the duration.

    On exit the driver that was active on entry is restored, including none
    at all. `_active` is read and assigned directly because the `.active`
    property would resolve an auto-detected default, which raises on a host
    with no iGPU, and `set_active` cannot restore "none". Entering a scope
    whose driver is already active does nothing.

    Switching does not invalidate compiled kernels: Triton caches them per
    device, and the NPU and the iGPU are different devices.
    """
    if kind not in _KINDS:
        raise ValueError(f"driver_scope: kind must be one of {_KINDS}, not {kind!r}")
    import triton

    config = triton.runtime.driver
    prev = getattr(config, "_active", None)
    if _is_kind(prev, kind):
        yield prev
        return
    config.set_active(_make(kind))
    try:
        yield config._active
    finally:
        config._active = prev
