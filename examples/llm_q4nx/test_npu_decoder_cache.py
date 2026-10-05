# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The NPU decoder kept across turns belongs to one prefiller.

`harness._install_shared_kv` replaces `air.FusedDecoder` with the adapter
`make_npu_decoder_class` returns, once per generation. A second prefiller in
the same process must get its own decoder bound to its own slab, not the first
one's. No NPU needed: mlir-air's decoder is replaced by a stand-in.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "gemma4_e2b_q4nx"))

from gemma4_prefill import Gemma4Prefill, make_npu_decoder_class  # noqa: E402

SHAPE = (2, 8)


class _FakeFusedDecoder:
    def __init__(self, model=None, max_L=None, verbose=True):
        self.ATTN_MAXL = max_L
        self.KV = np.zeros(SHAPE)
        # Two layers, both sliding, as mlir-air's decoder records them.
        self.UNI = SHAPE[0]
        self.SWA = {0, 1}
        self.closed = False

    def close(self):
        self.closed = True


def _prefiller(tag):
    return SimpleNamespace(
        kv_attn_maxl=16,
        _kv_host=np.zeros(SHAPE),
        _kv_slab=SimpleNamespace(bo=f"bo-{tag}"),
    )


def _install(air, pf):
    air.FusedDecoder = make_npu_decoder_class(air, pf)
    return air.FusedDecoder()


def test_each_prefiller_gets_its_own_decoder():
    air = SimpleNamespace(FusedDecoder=_FakeFusedDecoder)
    p1, p2 = _prefiller("a"), _prefiller("b")
    d1 = _install(air, p1)
    d2 = _install(air, p2)
    assert d2 is not d1
    assert d1.kvc == "bo-a" and d2.kvc == "bo-b"
    # Each prefiller's later turns reuse its own decoder.
    assert _install(air, p1) is d1
    assert _install(air, p2) is d2


def test_close_keeps_the_cached_decoder_and_release_closes_it():
    air = SimpleNamespace(FusedDecoder=_FakeFusedDecoder)
    pf = _prefiller("a")
    dec = _install(air, pf)
    dec.close()
    assert not dec.closed
    Gemma4Prefill.release_npu_decoder(pf)
    assert dec.closed
    assert getattr(pf, "_npu_decoder", None) is None


class _RefusedOnce(_FakeFusedDecoder):
    """A decoder whose first hardware context is refused, as amdxdna does when
    other contexts hold what it grants."""

    attempts = 0

    def __init__(self, *a, **kw):
        _RefusedOnce.attempts += 1
        if _RefusedOnce.attempts == 1:
            raise RuntimeError(
                "DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22): Invalid argument"
            )
        super().__init__(*a, **kw)


class _Chain:
    def __init__(self):
        self._runner = object()

    def close(self):
        self._runner = None


def test_a_refused_context_closes_the_chains_and_retries():
    import weakref

    from triton.backends.amd_triton_npu.multilaunch import NPUChain

    chain = _Chain()
    NPUChain._open[id(chain)] = weakref.ref(chain)
    _RefusedOnce.attempts = 0
    air = SimpleNamespace(FusedDecoder=_RefusedOnce)
    dec = _install(air, _prefiller("a"))
    assert _RefusedOnce.attempts == 2
    assert chain._runner is None and not NPUChain._open
    assert dec.kvc == "bo-a"


def test_other_errors_are_not_retried():
    class _Broken(_FakeFusedDecoder):
        def __init__(self, *a, **kw):
            raise RuntimeError("weight cache holds 3 elements, the build wants 4")

    air = SimpleNamespace(FusedDecoder=_Broken)
    try:
        _install(air, _prefiller("a"))
    except RuntimeError as e:
        assert "weight cache" in str(e)
    else:
        raise AssertionError("a non-context error was swallowed")
