# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Switch Triton's active driver between the NPU and the AMD iGPU for a scope.

A model that splits its work between the two launches Triton kernels for both
from one process, and Triton compiles and launches for whichever driver is
active. This is the one helper for choosing it; NPUChain's warmup, the Q4NX
examples and the gpt2/qwen2_5 examples all go through it.

    with driver_scope("npu"):
        kernel[grid](...)          # compiled and launched for the NPU

    with driver_scope("amd"):
        gpu_kernel[grid](...)      # compiled and launched for the iGPU

Kept free of `driver.py`'s imports (aie, air) so an iGPU-only caller does not
pay for the NPU toolchain.
"""

import contextlib

_KINDS = ("npu", "amd")

#: The AMD driver, built once. Construction is ~0.14 ms (it builds HIPUtils),
#: against ~2 us for an NPUDriver, and a hetero prefill switches to it twice per
#: layer. The NPU driver is built per scope instead: `NPUDriver()` reads the
#: launch runtime from the environment, which a caller may change in between.
_amd = None


def _is_kind(driver, kind):
    if driver is None:
        return False
    if kind == "npu":
        from .driver import NPUDriver

        return isinstance(driver, NPUDriver)
    return type(driver).__module__ == "triton.backends.amd.driver"


def _make(kind):
    global _amd
    if kind == "npu":
        from .driver import NPUDriver

        return NPUDriver()
    if _amd is None:
        from triton.backends import backends

        _amd = backends["amd"].driver()
    return _amd


@contextlib.contextmanager
def driver_scope(kind):
    """Make the `kind` driver ("npu" or "amd") active for the duration.

    Three rules, each one a bug an earlier copy of this had:

    * It reads ``_active``, not ``.active``. The property resolves the
      auto-detected default when nothing is active yet, and on an iGPU-free
      host that raises "0 active drivers".
    * It restores exactly what was active on entry, ``None`` included, by
      assigning ``_active``: ``set_active`` cannot express "nothing chosen".
      Leaving the NPU driver behind a process's first scope meant the next
      unscoped iGPU kernel was compiled for the NPU and failed in aircc.
    * It does nothing when a driver of that kind is already active, so nested
      and repeated scopes cost nothing.

    Switching does not drop compiled kernels: Triton's JIT cache is keyed by
    the active driver's device, which is "npu" for the NPU and an ordinal for
    the iGPU, so both stay cached across switches.
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
