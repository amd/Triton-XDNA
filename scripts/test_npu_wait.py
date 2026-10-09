# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""MultiLaunchRunner's NPU waits let other Python threads run.

Two ways, both checked here:

- `npu_wait`, which reaches the xrt::run behind a pyxrt.run through pybind11's
  cpp conduit and waits with the GIL released. It must build, and pyxrt must
  hand it the run wherever pyxrt has the conduit.
- polling `run.state()`, for a pyxrt built without the conduit.

A wait that held the GIL would leave results right and only lose
concurrency, so nothing else would notice. Needs XRT and g++, not an NPU:
`npu_wait` is checked through `_block`, which releases the GIL the same way
`wait` does, and the polling through a stand-in run.
"""

import sys
import threading
import time

from triton.backends.amd_triton_npu import multilaunch
from triton.backends.amd_triton_npu.driver import npu_wait_module, pyxrt_abi_ids


def _others_run_during(wait):
    """Whether a pure-Python thread runs while `wait()` blocks for ~0.3 s.

    Such a thread cannot run at all while another holds the GIL. It records
    when it ran, and with the GIL released some of those times fall well
    inside the wait.
    """
    seen, stop = [], threading.Event()

    def spin():
        n = 0
        while not stop.is_set():
            n += 1
            if n % 1000 == 0:
                seen.append(time.perf_counter())

    t = threading.Thread(target=spin)
    t.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    wait()
    t1 = time.perf_counter()
    stop.set()
    t.join()
    return any(t0 + 0.05 < s < t1 - 0.05 for s in seen)


class _Run:
    """Pending until `until`, then in `final` state, as pyxrt's run reports."""

    def __init__(self, seconds, final):
        self.until, self.final = time.perf_counter() + seconds, final

    def state(self):
        return 3 if time.perf_counter() < self.until else self.final


def main():
    mod = npu_wait_module()
    if mod is None:
        print("FAIL: npu_wait did not build")
        return 1
    import pyxrt

    if hasattr(pyxrt.run, "_pybind11_conduit_v1_"):
        if not mod.supported(pyxrt.run()):
            print(
                "FAIL: pyxrt declined to hand over its xrt::run; it was built "
                f"for ABI {', '.join(pyxrt_abi_ids()) or 'unknown'} "
                f"({pyxrt.__file__}). Per spelling: {mod._probe(pyxrt.run())}"
            )
            return 1
    else:
        print(f"note: {pyxrt.__file__} has no cpp conduit; waits poll")
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
    if not _others_run_during(lambda: mod._block(300)):
        print("FAIL: no other Python thread ran during an npu_wait wait")
        return 1

    if not _others_run_during(lambda: multilaunch._wait_polling(_Run(0.3, 4))):
        print("FAIL: no other Python thread ran during a polled wait")
        return 1
    try:
        multilaunch._wait_polling(_Run(0.01, 5))
    except RuntimeError:
        pass
    else:
        print("FAIL: a polled wait returned for a run that ended in error")
        return 1
    print("PASS: npu_wait")
    return 0


if __name__ == "__main__":
    sys.exit(main())
