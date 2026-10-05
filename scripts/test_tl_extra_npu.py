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

import importlib
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
            tl.extra.npu.DecodeConfig(
                model=bad, model_type="QWEN3_4B", context_length=2048, vocab_chunk=30
            )
        except ValueError:
            continue
        raise AssertionError(f"model={bad!r} was accepted")


def test_rejects_nonsense_shapes():
    import triton.language as tl

    for kwargs in (
        dict(model="qwen3-4b", model_type="QWEN3_4B", context_length=0, vocab_chunk=30),
        dict(
            model="qwen3-4b", model_type="QWEN3_4B", context_length=2048, vocab_chunk=0
        ),
    ):
        try:
            tl.extra.npu.DecodeConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} was accepted")


def test_env_is_derived_not_inherited():
    """The knobs come from the config, and `attn_maxl` rounds as the builder
    does -- two context lengths that round together share an artifact.

    On the default engine, which is the shared `fused_decode`. The PLE fork
    places four of them differently; `test_engine_decides_where_knobs_go`
    covers that.
    """
    import triton.language as tl

    cfg = tl.extra.npu.DecodeConfig(
        model="qwen3-4b",
        model_type="QWEN3_4B",
        context_length=2047,
        vocab_chunk=30,
        unified=1,
    )
    env = cfg.env()
    expected = {
        "DECODE_MODEL": "qwen3-4b",
        "DECODE_GOLDEN_L": "2047",
    }
    for k, v in expected.items():
        if env.get(k) != v:
            raise AssertionError(f"env[{k}] = {env.get(k)!r}, expected {v!r}")
    if cfg.attn_maxl != 2048:
        raise AssertionError(f"attn_maxl = {cfg.attn_maxl}, expected 2048")
    # Unset knobs stay out, so the builder keeps its own default for them.
    if "DECODE_WGROUP" in env:
        raise AssertionError("an unset knob leaked into the environment")
    # Since mlir-air `deffe6f1` these four are properties of the model, read
    # from its `_MODELS` entry and NOT from the environment. Setting them would
    # be inert, so the config must not; `_check_model_table` verifies them
    # against the builder instead. `UNIFIED` is not here either: every one of
    # mlir-air's Makefiles sets it and none of its code reads it.
    for k in ("VOCAB_CHUNK_I2", "W_DUAL_CHAN", "DECODE_STACK", "UNIFIED"):
        if k in env:
            raise AssertionError(
                f"{k} is owned by mlir-air's _MODELS table, but the config put "
                f"it in the environment, where it is ignored"
            )


def test_model_table_mismatch_is_refused():
    """A knob that disagrees with mlir-air's `_MODELS` entry stops the build.

    The builder would quietly use its own value, leaving the spec's recorded
    one a lie -- the same class of silent-wrong this module exists to prevent.
    """
    import importlib
    import triton.language as tl

    # By name, `fused_decode` is both the submodule and the function the
    # package exports; `__init__` binds the function, so the module has to be
    # imported explicitly.
    fd = importlib.import_module("triton.language.extra.npu.fused_decode")

    cfg = tl.extra.npu.DecodeConfig(
        model="qwen3-4b", model_type="QWEN3_4B", context_length=16, vocab_chunk=30
    )

    class _Builder:  # stands in for the imported mlir-air module
        MODEL = {"VOCAB_CHUNK_I2": 5, "W_DUAL_CHAN": 1}

    try:
        fd._check_model_table(_Builder, cfg)
    except tl.extra.npu.DecodeConfigError as e:
        if "VOCAB_CHUNK_I2" not in str(e):
            raise AssertionError(f"unhelpful message: {e}")
    else:
        raise AssertionError("a mismatched _MODELS value was not refused")

    # Agreement is silent, and a builder without the table is not second-guessed.
    _Builder.MODEL = {"VOCAB_CHUNK_I2": 30, "W_DUAL_CHAN": 1}
    fd._check_model_table(_Builder, cfg)
    fd._check_model_table(object(), cfg)


def test_engine_decides_where_knobs_go():
    """The same four knobs are environment on one engine and table on the other.

    mlir-air's PLE fork predates `deffe6f1`, so it still reads `W_DUAL_CHAN`,
    `VOCAB_CHUNK_I2`, `DECODE_WGROUP` and `DECODE_STACK` from the environment
    where the shared engine takes them from `_MODELS`. Getting that backwards
    is silent in both directions -- setting a variable nothing reads builds the
    wrong geometry, checking a table key that is not there checks nothing -- so
    it is a property of the engine and worth asserting rather than assuming.
    """
    import importlib
    import triton.language as tl

    fd = importlib.import_module("triton.language.extra.npu.fused_decode")
    moved = ("VOCAB_CHUNK_I2", "W_DUAL_CHAN", "DECODE_STACK", "DECODE_WGROUP")

    def cfg(engine):
        return tl.extra.npu.DecodeConfig(
            model="gemma4-e2b",
            model_type="GEMMA4_E2B",
            context_length=128,
            vocab_chunk=27,
            dual_channel=1,
            weight_group=0,
            stack_size=10240,
            engine=engine,
        )

    ple, shared = cfg("ple"), cfg("fused_decode")
    for k in moved:
        if k not in ple.env():
            raise AssertionError(
                f"{k} is missing from the PLE environment; that engine reads it "
                f"there, so the build would silently take its own default"
            )
        if k in shared.env():
            raise AssertionError(
                f"{k} leaked into the shared engine's environment, where it is "
                f"ignored -- it belongs to that engine's _MODELS entry"
            )

    # And the check runs only where the value was resolved rather than applied.
    # Gemma4's PLE entry carries none of these keys, so a table that disagrees
    # is not an error there: nothing read it.
    class _Builder:
        MODEL = {"VOCAB_CHUNK_I2": 5}

    fd._check_model_table(_Builder, ple)
    try:
        fd._check_model_table(_Builder, shared)
    except tl.extra.npu.DecodeConfigError:
        pass
    else:
        raise AssertionError("the shared engine did not check its model table")

    # Two engines building the same model are different artifacts, and the
    # name they are built under has to say so -- for the shared engine the
    # moved knobs are not even in `env()` to tell them apart.
    if ple.fingerprint() == shared.fingerprint():
        raise AssertionError("both engines fingerprinted alike")

    try:
        cfg("fused_decode_ple")  # the directory name, not the engine key
    except tl.extra.npu.DecodeConfigError:
        pass
    else:
        raise AssertionError("an unknown engine was accepted")


def test_build_scope_clears_what_it_does_not_set():
    """An inherited value cannot stand in for a knob the caller left unset.

    On an engine that reads these from the environment, omitting one is how a
    spec asks for the builder's own default -- Gemma4 names no `W_DUAL_CHAN`
    at all. Merely not setting it would leave an exported `W_DUAL_CHAN=0` in
    place, so the builder would answer with 0, the artifact would be built for
    a different shim channel split, and `fingerprint` would not record it: it
    hashes `env()`, which by construction does not contain what was never set.
    """
    import importlib
    import os

    import triton.language as tl

    fd = importlib.import_module("triton.language.extra.npu.fused_decode")
    # `dual_channel=None` is how a spec says "this model names no
    # W_DUAL_CHAN", which is Gemma4's case -- the field defaults to 1, so
    # leaving it out would emit a value rather than omit one. That is exactly
    # what `registry.ModelSpec.decode_config` passes for this model.
    cfg = tl.extra.npu.DecodeConfig(
        model="gemma4-e2b",
        model_type="GEMMA4_E2B",
        context_length=128,
        vocab_chunk=27,
        dual_channel=None,
        engine="ple",
    )
    if "W_DUAL_CHAN" in cfg.env():
        raise AssertionError("an omitted knob was emitted anyway")

    saved = {k: os.environ.get(k) for k in ("W_DUAL_CHAN", "DECODE_STACK")}
    try:
        os.environ["W_DUAL_CHAN"] = "0"
        os.environ["DECODE_STACK"] = "99999"
        with fd._environment(cfg.env(), fd.ENGINES[cfg.engine].env):
            for var in ("W_DUAL_CHAN", "DECODE_STACK"):
                if var in os.environ:
                    raise AssertionError(
                        f"{var} survived into the build scope, so the builder "
                        f"would read {os.environ[var]!r} rather than its own "
                        f"default"
                    )
            # What the caller DID set is still applied.
            if os.environ.get("VOCAB_CHUNK_I2") != "27":
                raise AssertionError("a configured knob was not applied")
        # And the caller's environment comes back untouched either way.
        if os.environ.get("W_DUAL_CHAN") != "0":
            raise AssertionError("a cleared variable was not restored")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_refuses_another_models_kernels():
    """The objects are per-model and keep the same filenames, so linking the
    wrong ones builds cleanly and decodes to garbage. A stamp naming a different
    -DMODEL_TYPE is refused; an unstamped directory is allowed, because
    mlir-air's own Makefile writes no stamp."""
    import os
    import tempfile

    import triton.language as tl
    from triton.language.extra.npu.fused_decode import _verify_kernels

    with tempfile.TemporaryDirectory() as d:
        objs = [os.path.join(d, n) for n in ("rope.o", "rms_residual.o")]
        for o in objs:
            open(o, "w").close()

        _verify_kernels(objs, "QWEN3_4B")  # unstamped: allowed

        open(os.path.join(d, ".decode_kernels.GEMMA3_4B.json"), "w").close()
        try:
            _verify_kernels(objs, "QWEN3_4B")
        except ValueError as e:
            if "GEMMA3_4B" not in str(e) or "QWEN3_4B" not in str(e):
                raise AssertionError(f"message names neither side: {e}")
        else:
            raise AssertionError("another model's objects were accepted")

        os.rename(
            os.path.join(d, ".decode_kernels.GEMMA3_4B.json"),
            os.path.join(d, ".decode_kernels.QWEN3_4B.json"),
        )
        _verify_kernels(objs, "QWEN3_4B")  # matching: accepted

    try:
        _verify_kernels(["/nonexistent/rope.o"], "QWEN3_4B")
    except ValueError:
        return
    raise AssertionError("a missing object was accepted")


def test_model_type_is_required():
    """It is what makes the check above possible."""
    import triton.language as tl

    try:
        tl.extra.npu.DecodeConfig(
            model="qwen3-4b", model_type="", context_length=2048, vocab_chunk=30
        )
    except ValueError:
        return
    raise AssertionError("an empty model_type was accepted")


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


def test_prefill_config_refuses_other_models():
    """mlir-air has a fused prefill builder for one model; any other is refused
    instead of building with that model's shapes."""
    import triton.language as tl

    npu = tl.extra.npu
    npu.PrefillConfig("gemma4-e2b")
    try:
        npu.PrefillConfig("llama-3.2-1b")
    except npu.PrefillConfigError:
        return
    raise AssertionError("a model without a fused prefill builder was accepted")


def test_prefill_config_must_be_a_config():
    import triton.language as tl

    npu = tl.extra.npu
    try:
        npu.fused_prefill({"model": "gemma4-e2b"}, "/nonexistent")
    except npu.PrefillConfigError:
        return
    raise AssertionError("a plain dict was accepted as a PrefillConfig")


_FAKE_BUILDER = """
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
(out / "manifest.json").write_text(json.dumps(dict(gemm=["g"], attn={}, lm="lm")))
for n in ("g", "lm"):
    for ext in (".xclbin", ".insts.bin"):
        (out / (n + ext)).write_bytes(b"")
(out / "libhostops.so").write_bytes(b"")
"""


def _fake_air_tree(root, builder="raise SystemExit('the builder must not run')\n"):
    """The files `fused_prefill.fingerprint` reads, in mlir-air's layout."""
    from pathlib import Path

    ex = Path(root) / "programming_examples"
    fp = ex / "llms" / "gemma4_e2b_q4nx" / "fused_prefill"
    fp.mkdir(parents=True)
    (fp / "build.py").write_text(builder)
    (fp.parent / "gemma4_e2b_q4nx_weights.py").write_text("D = 1536\n")
    for rel in (
        "matrix_multiplication/bf16_in_fp32_out",
        "flash_attention/kernel_fusion_based",
    ):
        (ex / rel).mkdir(parents=True)
        (ex / rel / "k.cc").write_text("// kernel\n")
    return fp, ex


def test_prefill_fingerprint_follows_sources():
    """Same inputs, same digest; a changed kernel source or toolchain path
    changes it; a missing kernel directory is an error, not a smaller digest."""
    import shutil
    import tempfile

    m = importlib.import_module("triton.language.extra.npu.fused_prefill")

    cfg = m.PrefillConfig("gemma4-e2b")
    with tempfile.TemporaryDirectory() as tmp:
        d, ex = _fake_air_tree(tmp)
        a = m.fingerprint(cfg, d, "/peano", "/aie")
        assert a == m.fingerprint(cfg, d, "/peano", "/aie"), "not deterministic"
        assert a != m.fingerprint(cfg, d, "/other-peano", "/aie"), "ignores Peano"
        assert a != m.fingerprint(cfg, d, "/peano", "/other-aie"), "ignores mlir-aie"
        (ex / "flash_attention/kernel_fusion_based/k.cc").write_text("// edited\n")
        assert a != m.fingerprint(cfg, d, "/peano", "/aie"), "ignores kernel sources"
        shutil.rmtree(ex / "matrix_multiplication")
        try:
            m.fingerprint(cfg, d, "/peano", "/aie")
        except FileNotFoundError:
            return
        raise AssertionError("a missing kernel directory was hashed as empty")


def test_prefill_reuses_a_complete_build():
    """A complete build under the current fingerprint is returned as is (the
    fake builder in `_fake_air_tree` fails if it runs), and a build missing the
    host library the runtime loads is not complete."""
    import json
    import tempfile
    from pathlib import Path

    import triton.backends.amd_triton_npu.driver as drv

    m = importlib.import_module("triton.language.extra.npu.fused_prefill")
    cfg = m.PrefillConfig("gemma4-e2b")
    saved = drv.find_peano_root, m._mlir_aie_root
    drv.find_peano_root = lambda: "/peano"
    m._mlir_aie_root = lambda: "/aie"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = _fake_air_tree(tmp)
            cache = Path(tmp) / "cache"
            out = cache / f"gemma4-e2b_{m.fingerprint(cfg, d, '/peano', '/aie')}"
            out.mkdir(parents=True)
            man = dict(gemm=["g"], attn={"a": {"base": "b"}}, lm="lm")
            (out / "manifest.json").write_text(json.dumps(man))
            for n in ("g", "b", "lm"):
                for ext in (".xclbin", ".insts.bin"):
                    (out / f"{n}{ext}").write_bytes(b"")
            (out / "libhostops.so").write_bytes(b"")
            r = m.fused_prefill(cfg, d, cache_root=cache)
            assert r["cached"] and r["build_dir"] == str(out.resolve()), r
            (out / "libhostops.so").unlink()
            assert not m._complete(out), "a build without libhostops.so is complete"
    finally:
        drv.find_peano_root, m._mlir_aie_root = saved


def test_prefill_rebuild_publishes_a_new_generation():
    """A rebuild publishes a new generation and switches the name to it; the
    build it replaces, including one published as a plain directory, stays."""
    import json
    import tempfile
    from pathlib import Path

    import triton.backends.amd_triton_npu.driver as drv

    m = importlib.import_module("triton.language.extra.npu.fused_prefill")
    cfg = m.PrefillConfig("gemma4-e2b")
    saved = drv.find_peano_root, m._mlir_aie_root
    drv.find_peano_root = lambda: "/peano"
    m._mlir_aie_root = lambda: "/aie"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = _fake_air_tree(tmp, builder=_FAKE_BUILDER)
            cache = Path(tmp) / "cache"
            name = cache / f"gemma4-e2b_{m.fingerprint(cfg, d, '/peano', '/aie')}"
            name.mkdir(parents=True)
            (name / "old").write_text("plain directory")
            r1 = m.fused_prefill(cfg, d, cache_root=cache, rebuild=True)
            assert name.is_symlink() and not r1["cached"], r1
            old = [p for p in cache.iterdir() if (p / "old").exists()]
            assert old and old[0] != name, "the plain directory was not kept"
            r2 = m.fused_prefill(cfg, d, cache_root=cache, rebuild=True)
            assert r2["build_dir"] != r1["build_dir"], "rebuild reused a generation"
            assert Path(
                r1["build_dir"], "manifest.json"
            ).is_file(), "old generation gone"
            assert json.loads(Path(r2["build_dir"], "fingerprint.json").read_text())
            assert (
                m.fused_prefill(cfg, d, cache_root=cache)["build_dir"]
                == r2["build_dir"]
            )
    finally:
        drv.find_peano_root, m._mlir_aie_root = saved


def main():
    tests = [
        ("tl.extra.npu is installed by the build", test_installed),
        ("model is required", test_model_is_required),
        ("nonsense shapes are rejected", test_rejects_nonsense_shapes),
        (
            "the environment is derived from the config",
            test_env_is_derived_not_inherited,
        ),
        (
            "a knob disagreeing with mlir-air's _MODELS is refused",
            test_model_table_mismatch_is_refused,
        ),
        (
            "the engine decides which knobs are environment",
            test_engine_decides_where_knobs_go,
        ),
        (
            "the build scope clears knobs it does not set",
            test_build_scope_clears_what_it_does_not_set,
        ),
        ("model_type is required", test_model_type_is_required),
        ("another model's kernels are refused", test_refuses_another_models_kernels),
        ("a dict is not a DecodeConfig", test_config_must_be_a_config),
        (
            "fused_prefill refuses models without a builder",
            test_prefill_config_refuses_other_models,
        ),
        ("a dict is not a PrefillConfig", test_prefill_config_must_be_a_config),
        (
            "the fused prefill fingerprint follows its inputs",
            test_prefill_fingerprint_follows_sources,
        ),
        (
            "a complete fused prefill build is reused",
            test_prefill_reuses_a_complete_build,
        ),
        (
            "a fused prefill rebuild publishes a new generation",
            test_prefill_rebuild_publishes_a_new_generation,
        ),
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
