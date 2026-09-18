#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Check that `tl.extra.npu` survived the build, and that its op validates.

The first check is the point of this file. `amd_triton_npu/language/npu/` is
installed into `triton.language.extra` by *Triton's* build -- `setup.py` finds a
`language/` directory beside each backend's `backend/`, for plugins from
`TRITON_PLUGIN_DIRS` as for in-tree backends -- so whether the op exists at all
is a property of the build, not of the source tree.

That failure is silent in the worst way: nothing in this repo imports
`tl.extra.npu` yet, so if the build stopped installing it, every test would
still pass and the op would simply not be there. Developing it by copying the
package into site-packages by hand -- which is how it was first exercised --
hides the same failure. Hence a check that runs in CI against a real build.

Needs no NPU, no weights and no mlir-air sources: it imports the language
package and exercises argument validation, nothing more. Building an artifact is
covered where the decode is built.

    python3 scripts/test_tl_extra_npu.py
"""

import sys


def check(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 -- the report is the point
        print(f"  FAIL: {name}: {type(e).__name__}: {e}")
        return False
    print(f"  ok: {name}")
    return True


def test_installed():
    """`tl.extra.npu` is present, which is what the build decides."""
    import triton.language as tl

    if not hasattr(tl.extra, "npu"):
        raise AssertionError(
            "tl.extra.npu is missing. It is installed from "
            "amd_triton_npu/language/npu by Triton's build (setup.py's "
            "BackendInstaller.prepare -> triton.language.extra.<name>); a build "
            "that no longer does that removes the op without failing anything "
            f"else. tl.extra has: {sorted(getattr(tl.extra, '__all__', []))}"
        )
    for attr in ("fused_decode", "DecodeConfig"):
        if not hasattr(tl.extra.npu, attr):
            raise AssertionError(f"tl.extra.npu.{attr} is missing")


def test_model_is_required():
    """The builder's default is Llama-3.2-1B; inheriting it builds the wrong
    model silently, so the op refuses rather than defaulting."""
    import triton.language as tl

    for bad in ("", None):
        try:
            tl.extra.npu.DecodeConfig(model=bad, context_length=2048, vocab_chunk=30)
        except ValueError:
            continue
        raise AssertionError(f"model={bad!r} was accepted")


def test_rejects_nonsense_shapes():
    import triton.language as tl

    for kwargs in (
        dict(model="qwen3-4b", context_length=0, vocab_chunk=30),
        dict(model="qwen3-4b", context_length=2048, vocab_chunk=0),
    ):
        try:
            tl.extra.npu.DecodeConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} was accepted")


def test_env_is_derived_not_inherited():
    """The knobs come from the config, and `attn_maxl` rounds as the builder
    does -- two context lengths that round together share an artifact."""
    import triton.language as tl

    cfg = tl.extra.npu.DecodeConfig(
        model="qwen3-4b", context_length=2047, vocab_chunk=30, unified=1
    )
    env = cfg.env()
    expected = {
        "DECODE_MODEL": "qwen3-4b",
        "DECODE_GOLDEN_L": "2047",
        "VOCAB_CHUNK_I2": "30",
        "UNIFIED": "1",
    }
    for k, v in expected.items():
        if env.get(k) != v:
            raise AssertionError(f"env[{k}] = {env.get(k)!r}, expected {v!r}")
    if cfg.attn_maxl != 2048:
        raise AssertionError(f"attn_maxl = {cfg.attn_maxl}, expected 2048")
    # Unset knobs stay out, so the builder keeps its own default for them.
    if "DECODE_WGROUP" in env:
        raise AssertionError("an unset knob leaked into the environment")


def test_config_must_be_a_config():
    """A dict of knobs is what this op exists to replace; it is not accepted
    in place of one."""
    import triton.language as tl

    try:
        tl.extra.npu.fused_decode({"model": "qwen3-4b"}, "/nonexistent")
    except ValueError:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"raised {type(e).__name__} rather than ValueError: {e}")
    raise AssertionError("a plain dict was accepted as a config")


def main():
    tests = [
        ("tl.extra.npu is installed by the build", test_installed),
        ("model is required", test_model_is_required),
        ("nonsense shapes are rejected", test_rejects_nonsense_shapes),
        (
            "the environment is derived from the config",
            test_env_is_derived_not_inherited,
        ),
        ("a dict is not a DecodeConfig", test_config_must_be_a_config),
    ]
    print("tl.extra.npu:")
    failed = sum(not check(n, f) for n, f in tests)
    if failed:
        print(f"{failed}/{len(tests)} failed")
        return 1
    print(f"all {len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
