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


def test_takes_mlir_airs_full_elf_on_xrt():
    for env in (
        {"DECODE_ELF": "1"},
        {"DECODE_ELF": "1", "AMD_TRITON_NPU_RUNTIME": "xrt"},
    ):
        assert airsrc.select_decode_artifact(env) == "1"


def test_refuses_mlir_airs_full_elf_on_hsa():
    # The HSA decoder picks its own artifact; mlir-air's would dispatch via pyxrt.
    env = {"DECODE_ELF": "1", "AMD_TRITON_NPU_RUNTIME": "hsa"}
    try:
        airsrc.select_decode_artifact(env)
    except airsrc.DecodeArtifactError as e:
        assert "AMD_TRITON_NPU_HSA_DECODE" in str(e)
    else:
        raise AssertionError("DECODE_ELF=1 under HSA should have been refused")


def test_refuses_an_unknown_decode_elf():
    for asked in ("yes", "true", "2"):
        try:
            airsrc.select_decode_artifact({"DECODE_ELF": asked})
        except airsrc.DecodeArtifactError:
            pass
        else:
            raise AssertionError(f"DECODE_ELF={asked!r} should have been refused")


def test_xrt_scratchpad_encodes_like_mlir_aie():
    import tempfile
    import types

    import numpy as np

    import xrt_elf

    words = np.zeros(2, np.uint32)
    synced = []
    bo = types.SimpleNamespace(map=lambda: memoryview(words), sync=synced.append)
    run = types.SimpleNamespace(get_ctrl_scratchpad_bo=lambda: bo)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "params.txt")
        with open(path, "w") as f:
            f.write(
                "2\n"
                "__air_param_argoff_5_x256_m256 0 i32 addr -2147483648 2147483647\n"
                "__air_param_attn_blk_0 1 i32 core - -\n"
                "0\n"
            )
        fake = types.SimpleNamespace(
            xclBOSyncDirection=types.SimpleNamespace(XCL_BO_SYNC_BO_TO_DEVICE=0)
        )
        real = sys.modules.get("pyxrt")
        sys.modules["pyxrt"] = fake  # no XRT needed to check the encoding
        try:
            sp = xrt_elf.Scratchpad(run, path)
        finally:
            if real is None:
                del sys.modules["pyxrt"]
            else:
                sys.modules["pyxrt"] = real
        sp.write("__air_param_argoff_5_x256_m256", np.int32(7 * 256 - 256))
        sp.write("__air_param_attn_blk_0", np.int32(7))
        sp.sync()
    assert words.tolist() == [7 * 256 - 256, 7 << 2]  # addr raw, core << 2
    assert len(synced) == 1


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


def _elf_dir(d, model, maxl=2048):
    """An artifact directory as `decode_build.py --format elf` leaves it."""
    import types

    import decode_build

    src = os.path.join(d, "build")
    os.makedirs(src)
    for n in ("decode.elf", "params.txt"):
        open(os.path.join(src, n), "w").close()
    art = {"elf_path": os.path.join(src, "decode.elf"), "attn_maxl": maxl}
    decode_build._copy_elf(art, d, types.SimpleNamespace(name=model))


def test_elf_window_is_the_one_the_elf_was_built_for():
    import tempfile

    import hsa_decode

    with tempfile.TemporaryDirectory() as d:
        _elf_dir(d, "llama-3.2-1b", maxl=2048)
        assert hsa_decode.elf_window(d) == 2048
        assert hsa_decode.elf_window(d, max_L=1024) == 2048
        try:
            hsa_decode.elf_window(d, max_L=4096)
        except hsa_decode.HsaDecodeError:
            pass
        else:
            raise AssertionError("a max_L beyond the ELF's window accepted")


def test_elf_keeps_its_own_model_stamp():
    import tempfile

    import decode_build
    import hsa_decode

    with tempfile.TemporaryDirectory() as d:
        _elf_dir(d, "llama-3.2-1b")
        # A template build for another model rewrites the shared stamp only.
        with open(os.path.join(d, decode_build.STAMP), "w") as f:
            f.write("llama-3.2-3b\n")
        try:
            hsa_decode._elf_program(d, 2048, 256, model="llama-3.2-3b")
        except hsa_decode.HsaDecodeError as e:
            assert "'llama-3.2-1b'" in str(e)
        else:
            raise AssertionError("another model's ELF accepted")


def test_programs_sharing_a_handle_share_a_lock():
    import hsa_decode

    assert hsa_decode._program_lock(0x1000) is hsa_decode._program_lock(0x1000)
    assert hsa_decode._program_lock(0x1000) is not hsa_decode._program_lock(0x2000)


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
