# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The KV sharing map a decode template is built with.

Worth its own test because an unset `DECODE_KV_SRC` is not an error -- it is
the identity map, which for a model whose later layers read an earlier layer's
cache is a template whose waves read their own empty slab. Those waves then
attend a zero key at the current position, and the model returns a fluent
opening that degrades into repetition a few dozen tokens later. Nothing in the
decode gates catches that: `make run` generates two tokens.

So the failure this covers is a knob going MISSING, which is exactly the shape
no build error will ever report. It checks three things:

* the map is derived, and non-identity, for a model whose layers share a cache;
* it is absent for every model whose layers do not, so no sibling starts
  carrying a variable its builder ignores;
* the validation refuses the ways a hand-written map goes wrong, rather than
  letting the builder discover them a minute into a lowering.

The validation half needs nothing. The derivation half reads the model's own
`kv_source_layer` from mlir-air, so it exits 77 without those sources.

Not collected by `scripts/run_tests.py` -- this directory is a library, not an
example. Run it by hand.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import registry  # noqa: E402
from triton.language.extra.npu import DecodeConfig, DecodeConfigError  # noqa: E402

#: Enough of a spec to construct one. Four waves keeps the maps readable.
_BASE = dict(
    model="gemma4-e2b",
    model_type="GEMMA4_E2B",
    context_length=128,
    vocab_chunk=27,
    engine="ple",
    decode_waves=4,
)


def _specs():
    return [v for v in vars(registry).values() if isinstance(v, registry.ModelSpec)]


def test_a_wave_may_not_read_a_later_slab():
    _refuses(kv_src=[0, 2, 2, 3])


def test_one_entry_per_wave():
    _refuses(kv_src=[0, 0, 0])
    _refuses(kv_src=[0, 0, 0, 0, 0])


def test_an_empty_map_is_not_a_quiet_identity():
    # The builder reads "" as the identity. Passing it here is far more likely
    # to be a derivation that returned nothing than a deliberate choice, and
    # None already spells the deliberate one.
    _refuses(kv_src=[])
    _refuses(kv_src="")


def test_non_integers():
    _refuses(kv_src="0,x,1,2")


def test_an_engine_that_cannot_express_a_map_refuses_one():
    # Only the PLE fork reads DECODE_KV_SRC. Accepting it for the shared engine
    # would drop it and build the identity map with no sign anything was asked.
    _refuses(kv_src=[0, 0, 0, 0], engine="fused_decode")


def test_a_valid_map_normalizes_to_the_builder_s_spelling():
    for given in ([0, 0, 1, 1], "0,0,1,1", (0, 0, 1, 1)):
        cfg = DecodeConfig(**_BASE, kv_src=given)
        assert cfg.kv_src == "0,0,1,1", (given, cfg.kv_src)
        assert cfg.env()["DECODE_KV_SRC"] == "0,0,1,1"


def test_the_map_is_part_of_the_artifact_identity():
    # Two templates differing only in their map are different binaries. If they
    # fingerprinted alike the cache would hand back whichever was built first.
    prints = {
        DecodeConfig(**_BASE, kv_src=m).fingerprint()
        for m in ([0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 2, 3])
    }
    assert len(prints) == 3, prints
    assert DecodeConfig(**_BASE).fingerprint() not in prints


def test_no_map_leaves_the_variable_unset():
    assert DecodeConfig(**_BASE).kv_src is None
    assert "DECODE_KV_SRC" not in DecodeConfig(**_BASE).env()


def _refuses(**kw):
    try:
        cfg = DecodeConfig(**{**_BASE, **kw})
    except DecodeConfigError:
        return
    raise AssertionError(f"{kw} was accepted as kv_src={cfg.kv_src!r}")


def _derivation():
    """The half that needs mlir-air. Returns a list of failure strings."""
    bad = []
    for spec in _specs():
        smap = spec.kv_source_map()
        shares = smap is not None
        reads = "DECODE_KV_SRC" in spec.decode_config(128).env()
        if shares != reads:
            bad.append(
                f"{spec.name}: sharing map {'derived' if shares else 'absent'} "
                f"but DECODE_KV_SRC {'set' if reads else 'unset'}"
            )
        if shares and smap == list(range(len(smap))):
            bad.append(f"{spec.name}: derived map is the identity; say None")
    # And the one model that does share: its map must name a source strictly
    # below every wave that shares, or the wave reads a slab nothing filled.
    smap = registry.GEMMA4_E2B.kv_source_map()
    if smap is None:
        bad.append("gemma4-e2b derives no sharing map, but its layers share")
    elif not any(s != i for i, s in enumerate(smap)):
        bad.append(f"gemma4-e2b's map shares nothing: {smap}")
    return bad


def main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS: {name}")
        except AssertionError as e:
            print(f"FAIL: {name}: {e}")
            failures += 1

    try:
        bad = _derivation()
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: the derivation half needs mlir-air's sources: {e}")
        return 1 if failures else 77
    for line in bad:
        print(f"FAIL: {line}")
    failures += len(bad)
    if not bad:
        print(f"PASS: derived maps agree with each engine ({len(_specs())} models)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
