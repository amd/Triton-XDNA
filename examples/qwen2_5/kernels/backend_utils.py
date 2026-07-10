# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Backend utilities for heterogeneous iGPU+NPU execution."""

import triton
from contextlib import contextmanager


@contextmanager
def npu_driver_scope():
    """Temporarily activate NPU driver for a kernel launch, then restore."""
    from triton.backends.amd_triton_npu.driver import NPUDriver

    triton.runtime.driver.set_active(NPUDriver())
    try:
        yield
    finally:
        triton.runtime.driver.reset_active()


class CachedNPUKernel:
    """Fast-path NPU kernel dispatch that bypasses Triton JIT after first compilation.

    First call goes through the full Triton JIT (compilation + XRT setup).
    Subsequent calls with the same constexpr values call the compiled C extension
    directly: 0.1ms vs ~27ms per dispatch.

    Usage::

        _cached_add = CachedNPUKernel()

        # In the NPU wrapper:
        _cached_add(add_kernel_npu, grid, a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    All arguments after ``grid`` must be in kernel signature order.
    Keyword arguments are constexprs (appended after positional args for mod.launch).
    The cache key is ``(grid, constexpr_kwargs)`` — different shapes or block sizes
    get separate compiled modules.
    """

    def __init__(self):
        self._cache = {}

    def __call__(self, kernel, grid, *args, **constexpr_kwargs):
        # Cache key: grid dimensions + constexpr values determine the compiled binary
        cache_key = (grid, tuple(constexpr_kwargs.items()))

        if cache_key in self._cache:
            # Fast path: direct C extension call (~0.1ms)
            mod = self._cache[cache_key]
            gX = grid[0] if len(grid) > 0 else 1
            gY = grid[1] if len(grid) > 1 else 1
            gZ = grid[2] if len(grid) > 2 else 1
            all_args = list(args) + list(constexpr_kwargs.values())
            mod.launch(gX, gY, gZ, None, None, None, None, *all_args)
        else:
            # Slow path: full Triton JIT compilation (~27ms + compile time)
            with npu_driver_scope():
                kernel[grid](*args, **constexpr_kwargs)
            from triton.backends.amd_triton_npu.driver import _last_dispatched_module

            self._cache[cache_key] = _last_dispatched_module
