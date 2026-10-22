# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The NPU decoder kept across turns belongs to one prefiller.

`harness._install_shared_kv` replaces `air.FusedDecoder` with the adapter
`make_npu_decoder_class` returns, once per generation. A second prefiller in
the same process must get its own decoder bound to its own slab, not the first
one's. The adapter also refuses a decoder that reads the slabs in another
order, and rebinds an ELF decoder's run to the slab. No NPU needed: mlir-air's
decoder is replaced by a stand-in.
"""

import pytest

import os
import sys
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "gemma4_e2b_q4nx"))

from gemma4_prefill import Gemma4Prefill, make_npu_decoder_class  # noqa: E402

SHAPE = (2, 8)


class _FakeElf:
    def bind(self, bos):
        self.bound = list(bos)


class _FakeFusedDecoder:
    elf_mode = False

    def __init__(self, model=None, max_L=None, verbose=True):
        self.ATTN_MAXL = max_L
        self.KV = np.zeros(SHAPE)
        # Two layers, both sliding and owning their cache, as mlir-air's
        # decoder records them.
        self.UNI = SHAPE[0]
        self.SWA = {0, 1}
        self.KV_SLABS, self.KV_SLOT = SHAPE[0], [0, 1]
        self.kvb = SimpleNamespace(extra=())
        self.x_bo, self.w_bo, self.r_bo, self.y_bo = "x", "w", "r", "y"
        self.pw_bo, self.px_bo = "pw", "px"
        self._elfdec = _FakeElf()
        self.closed = False

    def close(self):
        self.closed = True


def _prefiller(tag, slot=(0, 1)):
    return SimpleNamespace(
        kv_attn_maxl=16,
        _kv_host=np.zeros(SHAPE),
        _kv_slab=SimpleNamespace(bo=f"bo-{tag}"),
        _kv_slabs=SHAPE[0],
        _kv_slot=list(slot),
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


def test_a_decoder_reading_other_slabs_is_refused():
    air = SimpleNamespace(FusedDecoder=_FakeFusedDecoder)
    with pytest.raises(RuntimeError, match="slab"):
        _install(air, _prefiller("a", slot=(1, 0)))


def test_an_elf_decoder_is_rebound_to_the_slab():
    class _Elf(_FakeFusedDecoder):
        elf_mode = True

    air = SimpleNamespace(FusedDecoder=_Elf)
    dec = _install(air, _prefiller("a"))
    assert dec._elfdec.bound == ["x", "w", "r", "y", "bo-a", "pw", "px"]
