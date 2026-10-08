# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The fused prefill's hardware contexts, around `run_fused`.

They stay open between prompts and are released only when something else
needs the room: a decoder about to open its own, or a Triton chain or launch
the device refused. When the device refuses to reopen them because Triton
prefill chains hold the rest, the chains are closed and the prefill runs
again. No NPU needed: the fused prefill, the chains and the launches are
stand-ins.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triton.backends.amd_triton_npu import driver, multilaunch  # noqa: E402
from triton.backends.amd_triton_npu.multilaunch import NPUChain  # noqa: E402

from dense_fused_prefill import release_fused_prefills, run_fused  # noqa: E402
from harness import _releasing_fused_prefills  # noqa: E402

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


def test_kept_open_after_prefill():
    f = _Fused()
    assert run_fused(f, [1, 2]) == [1, 2]
    assert not f.suspended and all(c.open for c in f.ctxs)
    run_fused(f, [3])
    assert f.runs == 2 and not f.suspended
    assert release_fused_prefills() and f.suspended
    assert not release_fused_prefills()
    run_fused(f, [4])
    assert f.runs == 3 and not f.suspended


def test_refusal_closes_chains_and_retries():
    f = _Fused()
    run_fused(f, [1])
    release_fused_prefills()
    f.refuse = 1
    out, closed = _with_open_chains(lambda: run_fused(f, [2]))
    assert out == [2] and len(closed) == 2 and f.runs == 2 and not f.suspended


def test_other_errors_propagate():
    f = _Fused()
    run_fused(f, [1])
    release_fused_prefills()
    f.refuse = 2  # refused again after the chains are closed
    try:
        _with_open_chains(lambda: run_fused(f, [2]))
    except RuntimeError as e:
        assert "HWCTX" in str(e)
    else:
        raise AssertionError("a second refusal was swallowed")
    assert f.suspended


def test_refused_chain_releases_fused_prefill():
    """Before closing any chain of ours, which would re-stage its weights."""
    f = _Fused()
    run_fused(f, [1])
    refusals = [1]

    class Runner:
        def __init__(self, *a):
            if refusals:
                refusals.pop()
                raise RuntimeError(REFUSED)

    chain = NPUChain("t")
    chain._elf_path = "t.elf"
    saved = multilaunch.MultiLaunchRunner
    multilaunch.MultiLaunchRunner = Runner
    try:
        _, closed = _with_open_chains(chain._open_runner)
    finally:
        multilaunch.MultiLaunchRunner = saved
        NPUChain._open.pop(id(chain), None)
    assert isinstance(chain._runner, Runner) and f.suspended and not closed


def test_refused_launch_releases_fused_prefill():
    f = _Fused()
    run_fused(f, [1])

    class Launcher:
        refuse = 1

        def launch(self):
            if self.refuse:
                self.refuse -= 1
                raise RuntimeError(REFUSED)
            return "ran"

        def release_session(self):
            pass

    try:
        assert driver._launch_with_session("t", Launcher()) == "ran"
    finally:
        driver._live_sessions.pop("t", None)
    assert f.suspended


def test_decoder_releases_fused_prefill_first():
    f = _Fused()
    run_fused(f, [1])
    seen = []

    class Decoder:
        def __init__(self, n):
            seen.append((n, f.suspended))

    _releasing_fused_prefills(Decoder)(7)
    assert seen == [(7, True)]


def main():
    for t in (
        test_kept_open_after_prefill,
        test_refusal_closes_chains_and_retries,
        test_other_errors_propagate,
        test_refused_chain_releases_fused_prefill,
        test_refused_launch_releases_fused_prefill,
        test_decoder_releases_fused_prefill_first,
    ):
        t()
        print(f"PASS: {t.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
