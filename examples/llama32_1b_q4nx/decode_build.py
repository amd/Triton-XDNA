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

    if output_format == "elf":
        # One artifact for every context length: L is a scratchpad parameter,
        # not something baked into an instruction stream, so there is no
        # template pair and nothing per-L to build. params.txt is half of it --
        # it maps each parameter to its slot, and the host cannot write one
        # without it -- so it is copied out beside the ELF rather than left in
        # the project directory.
        dst_e = os.path.join(out_dir, "decode_scratchpad.elf")
        dst_p = os.path.join(out_dir, "decode_scratchpad.params.txt")
        shutil.copyfile(art["elf_path"], dst_e)
        params = os.path.join(os.path.dirname(art["elf_path"]), "params.txt")
        if not os.path.exists(params):
            raise SystemExit(
                f"{params} is missing: the build emitted no scratchpad "
                "parameters, so L has no way to reach the device. It needs an "
                "mlir-air whose fused_decode routes DYNSEQ through the "
                "scratchpad, and DECODE_DYNSEQ=1."
            )
        shutil.copyfile(params, dst_p)
        # The build's ATTN_MAXL, recorded rather than defaulted: a driver
        # guessing 2048 against an ELF built at another L mis-sizes the KV
        # cache and the mask threshold together, and both read as bad numerics.
        with open(os.path.join(out_dir, "decode_scratchpad.maxl"), "w") as f:
            f.write(f"{fd.ATTN_MAXL}\n")
        print(f"[decode-build] -> {dst_e} + params.txt", flush=True)
        return dst_e, dst_p

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
        choices=("xclbin", "pdi", "elf"),
        help="elf is what the HSA runtime consumes -- one artifact for every "
        "context length, with L a scratchpad parameter; pdi is the older HSA "
        "path, which needs a template pair and patches L into the instruction "
        "stream; xclbin is XRT's",
    )
    args = ap.parse_args(argv)

    if args.context_length is not None:
        lower_template(args.context_length, args.out_dir, args.format)
        return 0

    # One ELF serves every context length, so there is no pair to build and no
    # slope to calibrate -- that is the point of the scratchpad.
    if args.format == "elf":
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--context-length",
            str(args.max_context_length),
            "--format",
            "elf",
        ]
        if args.out_dir:
            cmd += ["--out-dir", args.out_dir]
        subprocess.run(cmd, check=True)
        print("[decode-build] one full ELF lowered through FusedDecodeOp.", flush=True)
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
