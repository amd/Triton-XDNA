# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The fused prefill's hardware contexts, around `run_fused`.

They are released after every prefill so a decoder can open its own. When
the device refuses to reopen them because Triton prefill chains hold the
rest, the chains are closed and the prefill runs again. No NPU needed: the
fused prefill and the chains are stand-ins.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triton.backends.amd_triton_npu.multilaunch import NPUChain  # noqa: E402

from dense_fused_prefill import run_fused  # noqa: E402

REFUSED = "DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22): Invalid argument"


class _Ctx:
    def __init__(self, fused):
        self.fused, self.open = fused, True

    def close(self):
        self.open = False

    def reopen(self):
        if self.open:
            raise AssertionError("reopened a context that was still open")
        if self is self.fused.ctxs[-1] and self.fused.refuse:
            self.fused.refuse -= 1
            raise RuntimeError(REFUSED)
        self.open = True


class _Fused:
    def __init__(self):
        self.refuse = 0
        self.ctxs = [_Ctx(self), _Ctx(self)]
        self.suspended, self.runs = False, 0

    def contexts(self):
        return self.ctxs

    def suspend(self):
        if not self.suspended:
            for c in self.ctxs:
                c.close()
            self.suspended = True

    def prefill(self, ids):
        if self.suspended:
            # As mlir-air's resume: the first context opens, the second is
            # the one refused, and the engine stays marked suspended.
            for c in self.ctxs:
                c.reopen()
            self.suspended = False
        self.runs += 1
        return list(ids)


def _with_open_chains(fn):
    saved = NPUChain._open, NPUChain._close_stalest
    closed = []
    NPUChain._open = {"a": None, "b": None}

    def close_stalest():
        if not NPUChain._open:
            return False
        closed.append(NPUChain._open.pop(next(iter(NPUChain._open))))
        return True

    NPUChain._close_stalest = staticmethod(close_stalest)
    try:
        return fn(), closed
    finally:
        NPUChain._open, NPUChain._close_stalest = saved


def test_released_after_prefill():
    f = _Fused()
    assert run_fused(f, [1, 2]) == [1, 2]
    assert f.suspended and not any(c.open for c in f.ctxs)
    run_fused(f, [3])
    assert f.runs == 2 and f.suspended


def test_refusal_closes_chains_and_retries():
    f = _Fused()
    run_fused(f, [1])
    f.refuse = 1
    out, closed = _with_open_chains(lambda: run_fused(f, [2]))
    assert out == [2] and len(closed) == 2 and f.runs == 2 and f.suspended


def test_other_errors_propagate():
    f = _Fused()
    run_fused(f, [1])
    f.refuse = 2  # refused again after the chains are closed
    try:
        _with_open_chains(lambda: run_fused(f, [2]))
    except RuntimeError as e:
        assert "HWCTX" in str(e)
    else:
        raise AssertionError("a second refusal was swallowed")
    assert f.suspended


def main():
    for t in (
        test_released_after_prefill,
        test_refusal_closes_chains_and_retries,
        test_other_errors_propagate,
    ):
        t()
        print(f"PASS: {t.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
