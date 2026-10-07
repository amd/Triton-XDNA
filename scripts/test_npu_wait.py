# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""`npu_wait` builds, pyxrt hands it the xrt::run it waits on, and its waits
let other Python threads run.

MultiLaunchRunner falls back to pyxrt's own wait, which holds the GIL, when
the first two fail, and nothing else would notice: results stay right and
only concurrency is lost. Needs XRT and g++, not an NPU: the GIL check blocks
through `_block`, which releases the GIL the same way `wait` does.
"""

import sys
import threading
import time

from triton.backends.amd_triton_npu.driver import npu_wait_module, pyxrt_abi_ids


def main():
    mod = npu_wait_module()
    if mod is None:
        print("FAIL: npu_wait did not build")
        return 1
    import pyxrt

    if not mod.supported(pyxrt.run()):
        print(
            "FAIL: pyxrt declined to hand over its xrt::run; it was built "
            f"for ABI {', '.join(pyxrt_abi_ids()) or 'unknown'}"
        )
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
    # A thread running pure Python cannot run at all while another holds the
    # GIL. It records when it ran; with the GIL released during the block,
    # some of those times fall well inside it.
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
    mod._block(300)
    t1 = time.perf_counter()
    stop.set()
    t.join()
    inside = sum(t0 + 0.05 < s < t1 - 0.05 for s in seen)
    if inside == 0:
        print("FAIL: no other Python thread ran during a wait")
        return 1
    print("PASS: npu_wait")
    return 0


if __name__ == "__main__":
    sys.exit(main())
