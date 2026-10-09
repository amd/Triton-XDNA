# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compile the fused decode's AIE kernels with Peano.

The decode links six hand-written AIE kernels -- seven on the PLE engine, which
adds its own `ple.cc`. mlir-air's Makefile builds them and this used to shell
out to it; doing the compiler invocations here instead means `FusedDecodeOp`
needs mlir-air's *sources* rather than its build system, and everything else
the compile needs -- Peano and the aie_api headers -- already comes from wheels
this repo pins.

It is also most of what `make compile-decode` was waiting for: measured at
~12 s for all six, against ~15 min for the delegated target, nearly all of
which was building templates we then relower ourselves.

Outputs are cached on a hash of the source, the flags and the compiler, so a
rebuild is free unless something that matters changed.

    python decode_kernels.py            # build (or reuse) all six
    python decode_kernels.py --force    # ignore the cache
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import airsrc  # noqa: E402
import registry  # noqa: E402

# Optimization level is load-bearing and NOT a tuning knob. From mlir-air's
# Makefile, which found both the hard way:
#
#   attn_qk/attn_kv MUST be -O1  -- -O2 deadlocks the attention do-while loop
#   the other four  MUST be -O2  -- rope miscompiles at -O1
#
# Neither failure is a build error. One hangs the device, the other returns
# wrong tokens.
NONATTN_KERNELS = ("proj_qmm", "rms_residual", "glu", "rope")
ATTN_KERNELS = ("attn_qk", "attn_kv")

#: Kernels an engine adds to those six, compiled out of its OWN `kernels/` at
#: the -O2 group's level. The PLE fork's `ple.cc` is the only one: it takes the
#: other six from its parent's directory, at its parent's optimization levels,
#: and adds this. See `fused_decode_ple/Makefile`.
ENGINE_KERNELS = {
    "fused_decode": (),
    "ple": ("ple",),
}

# attn_qk/attn_kv are emitted as LLVM IR, not objects: they are merge-linked
# into the core rather than object-linked (mlir-aie #3399), which is what
# `link_with_mode = "merge"` in the AIR module selects.
ATTN_SUFFIX = ".ll"

BASE_FLAGS = (
    "-std=c++20",
    "--target=aie2p-none-unknown-elf",
    "-Wno-parentheses",
    "-Wno-attributes",
    "-Wno-macro-redefined",
    "-Wno-empty-body",
    "-Wno-deprecated-declarations",
    "-DNDEBUG",
    "-D__AIE_API_AIE_ADF_HPP__",
    "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
)

#: One cache file per model: the kernels differ by -DMODEL_TYPE and
#: -DGLU_SLICE_EXPECTED, and two models sharing a cache file would each see the
#: other's entries as its own and skip a rebuild it needs.
CACHE_FILE_FMT = ".decode_kernels.{model_type}.json"


def _peano_clang():
    from triton.backends.amd_triton_npu.driver import find_peano_root

    root = find_peano_root()
    if not root:
        raise SystemExit(
            "no AIE-capable LLVM (Peano) found. Install llvm-aie -- "
            "`source utils/env_setup.sh` does -- or set PEANO_INSTALL_DIR."
        )
    clang = os.path.join(root, "bin", "clang++")
    if not os.path.exists(clang):
        raise SystemExit(f"{clang} does not exist")
    return clang


def _aie_include():
    """aie_api headers, from the mlir_aie wheel."""
    import importlib.util

    spec = importlib.util.find_spec("mlir_aie")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(
            "mlir_aie is not installed; `source utils/env_setup.sh` installs it."
        )
    inc = os.path.join(list(spec.submodule_search_locations)[0], "include")
    if not os.path.isdir(inc):
        raise SystemExit(f"{inc} does not exist")
    return inc


def glu_slice_expected(spec):
    """The builder's own GLU_SLICE for this model, or None.

    `glu.cc` checks its compile-time slice against `GLU_SLICE_EXPECTED` and the
    builder derives that from the model's egress-round parity, so the two have
    to be told the same thing. mlir-air's Makefile asks the builder rather than
    tabulating it (fused_decode/Makefile:65), and so does this -- a table would
    be one more thing to get silently wrong per model.

    Cheap and XRT-free: the print hook returns before `run()` imports pyxrt.
    Returns None if the builder declines to answer, which leaves the define off
    exactly as the Makefile's `$(if ...)` does.

    Asked of the engine this model actually builds through, not of the shared
    one: the two derive it from the same rule but from different models, and
    the fork is the only engine that knows Gemma4's.
    """
    from triton.language.extra.npu import ENGINES

    engine = ENGINES[spec.engine]
    eng_dir = airsrc.fused_decode_dir(spec.engine)
    env = dict(os.environ, **spec.decode_env, FUSED_DECODE_PRINT_CONST="GLU_SLICE")
    r = subprocess.run(
        [sys.executable, os.path.join(eng_dir, engine.module)],
        capture_output=True,
        text=True,
        cwd=eng_dir,
        env=env,
    )
    out = r.stdout.strip().splitlines()
    if r.returncode != 0 or not out or not out[-1].strip().isdigit():
        return None
    return out[-1].strip()


def _command(clang, src_dir, src, out, opt, extra=(), defines=()):
    """One Peano invocation. `src_dir` is the SHARED engine's directory.

    The include paths are the shared engine's even when `src` is the PLE fork's
    own kernel, because that is where `models/` and the kernel headers live for
    both -- `fused_decode_ple/Makefile` passes `-I $(DEC)/kernels -I
    $(DEC)/models` with `DEC` pointing back at the parent.
    """
    flags = [
        clang,
        *BASE_FLAGS,
        *defines,
        "-I",
        _aie_include(),
        "-I",
        os.path.join(src_dir, "kernels"),
        "-I",
        os.path.join(src_dir, "models"),
        opt,
        *extra,
        src,
        "-o",
        out,
    ]
    return flags


def _fingerprint(clang, src, cmd):
    """Hash what the output actually depends on: the command line, the source,
    and the compiler's own identity. Headers are covered transitively by the
    source pin -- they move only when the checkout does."""
    h = hashlib.sha256()
    h.update(" ".join(cmd).encode())
    with open(src, "rb") as f:
        h.update(f.read())
    try:
        h.update(subprocess.run([clang, "--version"], capture_output=True).stdout)
    except Exception:
        pass
    return h.hexdigest()[:16]


def kernel_dir(spec, src_dir=None):
    """Where this model's kernel objects are built.

    Per model, because the objects carry the model in their defines while
    keeping fixed names: one shared directory and two models would silently
    hand each other the wrong `proj_qmm.o`. Under the model's own ENGINE
    directory, because the same is true one level up -- the PLE fork links a
    seventh object the shared engine has no place for.
    """
    src_dir = src_dir or airsrc.fused_decode_dir(spec.engine)
    return os.path.join(src_dir, f"kernels_{spec.model_type.lower()}")


def build(spec=None, src_dir=None, out_dir=None, force=False, verbose=True):
    """Compile every kernel `spec`'s engine links. Returns their paths.

    `src_dir` is the SHARED engine's directory whichever engine is in play: it
    holds the six kernels and the headers both engines compile against. Only
    the extras in `ENGINE_KERNELS` come from elsewhere.
    """
    spec = spec or registry.spec()
    src_dir = src_dir or airsrc.fused_decode_dir()
    eng_dir = airsrc.fused_decode_dir(spec.engine)
    out_dir = out_dir or kernel_dir(spec, eng_dir)
    if not os.path.isdir(os.path.join(src_dir, "kernels")):
        raise SystemExit(f"no kernels/ under {src_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # Both defines are model-specific and neither failure is a build error:
    # the wrong MODEL_TYPE compiles and decodes to garbage, and a GLU_SLICE
    # mismatch trips glu.cc's own static check.
    defines = [f"-DMODEL_TYPE={spec.model_type}"]
    glu_slice = glu_slice_expected(spec)
    if glu_slice is not None:
        defines.append(f"-DGLU_SLICE_EXPECTED={glu_slice}")

    clang = _peano_clang()
    cache_path = os.path.join(
        out_dir, CACHE_FILE_FMT.format(model_type=spec.model_type)
    )
    cache = {}
    if not force and os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}

    # -c on the object path: without it clang links, and the failure is a
    # confusing "undefined symbol: main" from crt1.o rather than a missing flag.
    def _src(directory, name):
        return os.path.join(directory, "kernels", f"{name}.cc")

    jobs = (
        [(_src(src_dir, n), f"{n}.o", "-O2", ("-c",)) for n in NONATTN_KERNELS]
        + [
            (
                _src(src_dir, n),
                f"{n}{ATTN_SUFFIX}",
                "-O1",
                ("-DDECODE_INLINE_ATTN", "-S", "-emit-llvm"),
            )
            for n in ATTN_KERNELS
        ]
        # The engine's own, from its own directory, at the -O2 group's level.
        + [
            (_src(eng_dir, n), f"{n}.o", "-O2", ("-c",))
            for n in ENGINE_KERNELS[spec.engine]
        ]
    )

    outputs, rebuilt = [], 0
    t0 = time.time()
    for src, out_name, opt, extra in jobs:
        name = os.path.basename(src)
        out = os.path.join(out_dir, out_name)
        cmd = _command(clang, src_dir, src, out, opt, extra, defines)
        fp = _fingerprint(clang, src, cmd)
        if not force and cache.get(out_name) == fp and os.path.exists(out):
            outputs.append(out)
            continue
        if verbose:
            print(f"[decode-kernels] {name} ({opt}) -> {out_name}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stdout + r.stderr)
            raise SystemExit(f"Peano failed on {name}")
        cache[out_name] = fp
        outputs.append(out)
        rebuilt += 1

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    if verbose:
        n = len(jobs)
        print(
            f"[decode-kernels] {rebuilt}/{n} rebuilt in {time.time() - t0:.1f}s "
            f"({n - rebuilt} cached)",
            flush=True,
        )
    return outputs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default=registry.DEFAULT,
        help=f"model family ({', '.join(sorted(registry.SPECS))})",
    )
    ap.add_argument("--src-dir", default=None, help="mlir-air's fused_decode directory")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args(argv)
    build(registry.spec(args.model), args.src_dir, args.out_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
