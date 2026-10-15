# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Which decode artifact this example asks mlir-air for.

Worth its own test because nothing else covers it. The prefill gate CI runs
(`--prefill-only`) never reaches `_air_inference_module()`, so the selection is
never executed there; and a full decode run needs a `make compile-decode` and
1.3 GB of weights, which the gate deliberately avoids.

This needs no NPU, no weights and no mlir-air sources -- it drives
`airsrc.select_decode_artifact` against a dict standing in for the environment.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import airsrc  # noqa: E402


def test_defaults_to_the_templates_we_build():
    env = {}
    assert airsrc.select_decode_artifact(env) == "0"
    assert env["DECODE_ELF"] == "0"


def test_leaves_an_explicit_zero_alone():
    env = {"DECODE_ELF": "0"}
    assert airsrc.select_decode_artifact(env) == "0"


def test_rejects_mlir_airs_own_elf_dispatch():
    # mlir-air's default. Has to fail here, with a reason, rather than inside
    # mlir-air on a missing decode_scratchpad.maxl -- or, worse, in an abort
    # from a duplicate LLVM option registration.
    for asked in ("1", "yes", "true"):
        env = {"DECODE_ELF": asked}
        try:
            airsrc.select_decode_artifact(env)
        except airsrc.DecodeArtifactError as e:
            assert "mlir-air's own full-ELF dispatch" in str(e)
            # The refusal names the switch that does choose the HSA decode.
            assert "AMD_TRITON_NPU_HSA_DECODE" in str(e)
        else:
            raise AssertionError(f"DECODE_ELF={asked!r} should have been refused")


def _built_by(fmt):
    """The (L, format) pairs `decode_build.py --format fmt` lowers, compiling nothing."""
    import decode_build

    built = []
    real = decode_build.lower_template
    decode_build.lower_template = lambda L, out, f, model: built.append((L, f))
    try:
        decode_build.main(["--format", fmt, "--max-context-length", "2048"])
    finally:
        decode_build.lower_template = real
    return sorted(built)


def test_hsa_builds_bring_the_xclbin_pair_the_base_decoder_loads():
    # mlir-air's decoder, which the HSA decoders subclass, loads the xclbin pair.
    pair = [(2047, "xclbin"), (2048, "xclbin")]
    assert _built_by("xclbin") == pair
    assert _built_by("pdi") == sorted(pair + [(2047, "pdi"), (2048, "pdi")])
    assert _built_by("elf") == sorted(pair + [(2048, "elf")])


def test_hsa_decoder_names_the_missing_xclbin_pair():
    import tempfile

    import hsa_decode

    with tempfile.TemporaryDirectory() as d:

        def touch(*names):
            for n in names:
                open(os.path.join(d, n), "w").close()

        # ELF and PDIs without the xclbin pair.
        touch("decode_scratchpad.elf", "decode_scratchpad.params.txt")
        touch("decode_L2048.pdi", "decode_L2048.insts.bin")
        touch("decode_L2047.pdi", "decode_L2047.insts.bin")
        try:
            hsa_decode.require_xclbin_pair(d)
        except hsa_decode.HsaDecodeError as e:
            assert "make compile-decode RUNTIME=hsa" in str(e)
        else:
            raise AssertionError("ELF and PDIs without the xclbin pair accepted")

        touch("decode_L2048.xclbin")  # one template is no pair
        try:
            hsa_decode.require_xclbin_pair(d)
        except hsa_decode.HsaDecodeError:
            pass
        else:
            raise AssertionError("a lone xclbin template accepted")

        touch("decode_L2047.xclbin")
        hsa_decode.require_xclbin_pair(d)


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
