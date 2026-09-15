# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the fused-decode templates through Triton-XDNA's own lowering.

The AIR IR comes from mlir-air's `fused_decode.build_module()`; the compile
goes through `FusedDecodeOp`, which hands it to the same `_aircc_compile`
every other kernel in this backend uses. From there the artifact is an
ordinary AIR design -- same aircc invocation, same artifact handling -- which
is the point: the decode stops being a thing mlir-air's Makefile builds behind
our back and becomes a thing this backend lowers.

A "template" is one xclbin + insts.bin pair, compiled for one context length.
Two are needed: the L the decode runs at, and an L-1 the loader uses as a
slope reference.

    python decode_build.py                      # both templates
    python decode_build.py --context-length 2048   # one, in this process

`build_module()` reads its geometry from the environment at import time, so
each context length needs its own process; `--context-length` is how the
parent re-enters for each.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

# mlir-air's Makefile builds the L=2048 template under exactly this
# environment (its DECODE_ENV plus DECODE_GOLDEN_L). Diverging here produces a
# module that builds and then decodes to garbage, so it is copied verbatim
# rather than reconstructed.
DECODE_ENV = dict(
    VOCAB_CHUNK_I2="18",
    LM_HEAD="0",
    NLAYERS="1",
    DECODE_GOLDEN="1",
    W_DUAL_CHAN="1",
    PROJ_RC_CACHE="1",
)

# The kernels aircc links into the module. Built by mlir-air's Makefile with
# Peano (`make compile-decode` builds them before the templates); this step
# consumes them rather than rebuilding them.
KERNEL_OBJECTS = (
    "proj_qmm.o",
    "rms_residual.o",
    "glu.o",
    "rope.o",
    "attn_qk.ll",
    "attn_kv.ll",
)

DEFAULT_L = 2048


def fused_decode_dir():
    return str(config._air_llms_root().parent / "fused_decode")


def lower_template(L, out_dir=None, output_format="xclbin"):
    """Lower one decode template -- the xclbin + insts pair for context length L.

    Returns their paths. Must run in a process that has not yet imported
    `fused_decode`: that module reads DECODE_GOLDEN_L (and the rest of its
    geometry) at import time, so a second L in the same process would silently
    reuse the first one's.
    """
    fd_dir = fused_decode_dir()
    out_dir = out_dir or fd_dir
    os.environ.update(DECODE_ENV, DECODE_GOLDEN_L=str(L))

    sys.path.insert(0, fd_dir)
    spec = importlib.util.spec_from_file_location(
        "fused_decode", os.path.join(fd_dir, "fused_decode.py")
    )
    fd = importlib.util.module_from_spec(spec)
    sys.modules["fused_decode"] = fd
    spec.loader.exec_module(fd)

    objs = [os.path.join(fd_dir, o) for o in KERNEL_OBJECTS]
    missing = [o for o in objs if not os.path.exists(o)]
    if missing:
        raise SystemExit(
            "missing decode kernel objects:\n  "
            + "\n  ".join(missing)
            + "\n\nBuild them first (they are Peano compiles, not AIR):\n"
            f"  make -C {fd_dir} compile-decode"
        )

    from triton.backends.amd_triton_npu.fused_decode_op import FusedDecodeOp

    op = FusedDecodeOp.from_builder(
        fd.build_module, kernel_objects=objs, name=f"fused_decode_L{L}"
    )
    print(
        f"[decode-build] L={L} model={fd.MODEL_NAME} ATTN_MAXL={fd.ATTN_MAXL} "
        f"stack={op.stack_size} aircc_args={' '.join(op.aircc_args)}",
        flush=True,
    )
    t0 = time.time()
    art = op.lower(output_format=output_format)
    print(f"[decode-build] L={L} lowered in {time.time() - t0:.1f}s", flush=True)

    # Name them the way the decode's template loader expects to find them.
    ext = "pdi" if output_format == "pdi" else "xclbin"
    dst_x = os.path.join(out_dir, f"decode_L{L}.{ext}")
    dst_i = os.path.join(out_dir, f"decode_L{L}.insts.bin")
    shutil.copyfile(art["bin_path"], dst_x)
    shutil.copyfile(art["insts_path"], dst_i)
    print(f"[decode-build] L={L} -> {dst_x}", flush=True)
    return dst_x, dst_i


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="lower ONE template, for this context length, in this process",
    )
    ap.add_argument(
        "--max-context-length",
        type=int,
        default=DEFAULT_L,
        help=f"the context length the decode runs at (default {DEFAULT_L}); "
        "its L-1 slope reference is built too",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--format",
        default="xclbin",
        choices=("xclbin", "pdi"),
        help="pdi is what the HSA runtime consumes; xclbin is XRT's",
    )
    args = ap.parse_args(argv)

    if args.context_length is not None:
        lower_template(args.context_length, args.out_dir, args.format)
        return 0

    # The loader wants the L template and an L-1 slope reference. Separate
    # processes because the builder's geometry is import-time state.
    for L in (args.max_context_length, args.max_context_length - 1):
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--context-length",
            str(L),
            "--format",
            args.format,
        ]
        if args.out_dir:
            cmd += ["--out-dir", args.out_dir]
        subprocess.run(cmd, check=True)
    print("[decode-build] both templates lowered through FusedDecodeOp.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
