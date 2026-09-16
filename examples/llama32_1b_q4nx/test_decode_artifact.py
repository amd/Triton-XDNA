# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Which decode artifact this example asks mlir-air for.

Worth its own test because nothing else covers it. The prefill gate CI runs
(`--prefill-only`) never reaches `_air_inference_module()`, so the selection is
never executed there; and a full decode run needs a `make compile-decode` and
1.3 GB of weights, which the gate deliberately avoids.

This needs no NPU, no weights and no mlir-air sources -- it drives
`config.select_decode_artifact` against a dict standing in for the environment.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402


def test_defaults_to_the_templates_we_build():
    env = {}
    assert config.select_decode_artifact(env) == "0"
    assert env["DECODE_ELF"] == "0"


def test_leaves_an_explicit_zero_alone():
    env = {"DECODE_ELF": "0"}
    assert config.select_decode_artifact(env) == "0"


def test_rejects_the_elf_we_do_not_build():
    # mlir-air's default. Has to fail here, with a reason, rather than inside
    # mlir-air on a missing decode_scratchpad.maxl -- or, worse, in an abort
    # from a duplicate LLVM option registration.
    for asked in ("1", "yes", "true"):
        env = {"DECODE_ELF": asked}
        try:
            config.select_decode_artifact(env)
        except config.DecodeArtifactError as e:
            assert "does not produce that ELF" in str(e)
        else:
            raise AssertionError(f"DECODE_ELF={asked!r} should have been refused")


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
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
