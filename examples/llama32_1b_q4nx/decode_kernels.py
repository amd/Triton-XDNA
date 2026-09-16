# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compile the fused decode's AIE kernels with Peano.

The decode links six hand-written AIE kernels. mlir-air's Makefile builds them
and this used to shell out to it; doing the six compiler invocations here
instead means `FusedDecodeOp` needs mlir-air's *sources* rather than its build
system, and everything else the compile needs -- Peano and the aie_api headers
-- already comes from wheels this repo pins.

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

import config  # noqa: E402

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
    "-DMODEL_TYPE=LLAMA_3_2_1B",
    "-D__AIE_API_AIE_ADF_HPP__",
    "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
)

CACHE_FILE = ".decode_kernels.json"


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


def _command(clang, src_dir, name, out, opt, extra=()):
    flags = [
        clang,
        *BASE_FLAGS,
        "-I",
        _aie_include(),
        "-I",
        os.path.join(src_dir, "kernels"),
        "-I",
        os.path.join(src_dir, "models"),
        opt,
        *extra,
        os.path.join(src_dir, "kernels", f"{name}.cc"),
        "-o",
        out,
    ]
    return flags


def _fingerprint(clang, src_dir, name, cmd):
    """Hash what the output actually depends on: the command line, the source,
    and the compiler's own identity. Headers are covered transitively by the
    source pin -- they move only when the checkout does."""
    h = hashlib.sha256()
    h.update(" ".join(cmd).encode())
    with open(os.path.join(src_dir, "kernels", f"{name}.cc"), "rb") as f:
        h.update(f.read())
    try:
        h.update(subprocess.run([clang, "--version"], capture_output=True).stdout)
    except Exception:
        pass
    return h.hexdigest()[:16]


def build(src_dir=None, out_dir=None, force=False, verbose=True):
    """Compile all six kernels. Returns the list of artifact paths."""
    src_dir = src_dir or str(config._air_llms_root().parent / "fused_decode")
    out_dir = out_dir or src_dir
    if not os.path.isdir(os.path.join(src_dir, "kernels")):
        raise SystemExit(f"no kernels/ under {src_dir}")

    clang = _peano_clang()
    cache_path = os.path.join(out_dir, CACHE_FILE)
    cache = {}
    if not force and os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}

    # -c on the object path: without it clang links, and the failure is a
    # confusing "undefined symbol: main" from crt1.o rather than a missing flag.
    jobs = [(n, f"{n}.o", "-O2", ("-c",)) for n in NONATTN_KERNELS] + [
        (n, f"{n}{ATTN_SUFFIX}", "-O1", ("-DDECODE_INLINE_ATTN", "-S", "-emit-llvm"))
        for n in ATTN_KERNELS
    ]

    outputs, rebuilt = [], 0
    t0 = time.time()
    for name, out_name, opt, extra in jobs:
        out = os.path.join(out_dir, out_name)
        cmd = _command(clang, src_dir, name, out, opt, extra)
        fp = _fingerprint(clang, src_dir, name, cmd)
        if not force and cache.get(out_name) == fp and os.path.exists(out):
            outputs.append(out)
            continue
        if verbose:
            print(f"[decode-kernels] {name}.cc ({opt}) -> {out_name}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stdout + r.stderr)
            raise SystemExit(f"Peano failed on {name}.cc")
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
    ap.add_argument("--src-dir", default=None, help="mlir-air's fused_decode directory")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args(argv)
    build(args.src_dir, args.out_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
