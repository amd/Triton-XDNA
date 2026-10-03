#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Checks for `NPUChain`'s process-wide cap on open hardware contexts.

Every chain that has run holds an `xrt::hw_context`, and the NPU refuses the
30th or so in one process (`DRM_IOCTL_AMDXDNA_CREATE_HWCTX ... err=-2`). A
model that caches chains by shape opens a fresh set per prompt length, so a
long-lived process used to die at its third length. `NPUChain.max_open` bounds
them: opening one more closes the least recently dispatched, which reopens on
its next `run()`.

1. more chains than the device has contexts all dispatch, and no more than
   `max_open` hold one at any time;
2. an evicted chain's results are right after it reopens -- including its
   static operand, which the reopened context no longer holds and has to
   stage again. A reopen that skipped that would read an unwritten buffer;
3. dispatching a chain makes it the most recent, so a working set that fits
   under the cap is never evicted from under itself.

Needs an npu2 device. Exits 77 without one.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "examples", "zero_copy"))

#: More than the device can hold at once, so the cap is what lets this finish.
N_CHAINS = 36
N = 4096


def skip(msg):
    print(f"SKIP: {msg}")
    sys.exit(77)


def main():
    try:
        import numpy as np
        import torch
        import triton
        from triton.backends.amd_triton_npu.driver import NPUDriver
        from triton.backends.amd_triton_npu.multilaunch import NPUChain
        from common import add_chain
    except ImportError as e:
        skip(f"cannot import the backend: {e}")
    try:
        import pyxrt

        pyxrt.device(0)
    except Exception as e:
        skip(f"no NPU: {e}")
    triton.runtime.driver.set_active(NPUDriver())

    def live():
        return [r() for r in NPUChain._open.values() if r() is not None]

    rng = np.random.default_rng(0)
    a = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    addends = [np.full(N, float(i), np.float32) for i in range(N_CHAINS)]

    def dispatch(i, chain):
        out = chain.run(
            [a, b, np.zeros(N, np.float32), addends[i], np.zeros(N, np.float32)],
            bo_key=f"cap_{i}",
            static_indices={add_chain.I_ADDEND},
            intermediate_indices=add_chain.INTERMEDIATE,
            output_indices=add_chain.OUTPUT,
        )[add_chain.I_OUT]
        # Absolute, not np.allclose's default: the array's f32 add rounds
        # within an ulp of numpy's, which a relative test fails near zero.
        return bool(np.allclose(out, a + b + addends[i], rtol=0, atol=1e-5))

    failures = 0

    def check(ok, what):
        nonlocal failures
        print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
        failures += not ok

    print(f"NPUChain.max_open = {NPUChain.max_open}")
    chains = [add_chain.build(f"cap_{i}", N) for i in range(N_CHAINS)]
    ok, peak = True, 0
    for i, chain in enumerate(chains):
        ok &= dispatch(i, chain)
        peak = max(peak, len(live()))
    check(ok, f"{N_CHAINS} chains dispatch, each with its own static operand")
    check(peak <= NPUChain.max_open, f"at most max_open open at once (peak {peak})")

    first = chains[0]
    check(first._runner is None, "the least recent chain was closed")
    check(dispatch(0, first), "and is right after reopening, static re-staged")

    # A working set below the cap, cycled while one chain outside it keeps
    # being opened: the set must stay open the whole time.
    hot = chains[1:4]
    for i, chain in enumerate(hot, 1):
        dispatch(i, chain)
    stayed = True
    for j in range(4, 4 + NPUChain.max_open):
        dispatch(j, chains[j])
        for i, chain in enumerate(hot, 1):
            dispatch(i, chain)
        stayed &= all(c._runner is not None for c in hot)
    check(stayed, "a recently dispatched working set is never evicted")

    for chain in chains:
        chain.close()
    check(not live(), "close() releases every context")

    print("\nRESULT:", "PASS" if not failures else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
