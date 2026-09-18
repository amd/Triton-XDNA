# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the fused-decode templates through Triton-XDNA's own lowering.

The AIR IR comes from mlir-air's `fused_decode.build_module()`; the compile
goes through `tl.extra.npu.fused_decode`, which hands it to the same
`_aircc_compile` every other kernel in this backend uses. From there the artifact is an
ordinary AIR design -- same aircc invocation, same artifact handling -- which
is the point: the decode stops being a thing mlir-air's Makefile builds behind
our back and becomes a thing this backend lowers.

A "template" is one xclbin + insts.bin pair, compiled for one context length.
Two are needed: the L the decode runs at, and an L-1 the loader uses as a
slope reference.

    python decode_build.py                      # both templates
    python decode_build.py --context-length 2048   # one, in this process

Both templates are built in one process. `build_module()` reads its geometry
from the environment at import time, which is why this used to fork per context
length; `tl.extra.npu.fused_decode` takes the configuration as arguments and
scopes that environment to a single build, so it no longer has to.
`--context-length` still builds just one, for when that is all you want.
"""

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import airsrc  # noqa: E402
import decode_kernels  # noqa: E402
import registry  # noqa: E402

# The kernels aircc links into the module. Built by decode_kernels.py with
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


#: Written beside the artifacts, naming the model that built them.
#:
#: mlir-air's loader resolves the decode directory as a fixed path, so every
#: model's `decode_L2048.xclbin` lands on the same name. Loading one model's
#: artifacts under another does not fail -- it decodes to fluent nonsense --
#: so the stamp is what turns that into an error. See `check_stamp`.
STAMP = "decode_model.txt"


def read_stamp(out_dir=None):
    """The model whose artifacts are in `out_dir`, or None if unstamped."""
    path = os.path.join(out_dir or airsrc.fused_decode_dir(), STAMP)
    try:
        with open(path) as f:
            return f.read().strip() or None
    except OSError:
        return None


def check_stamp(model, out_dir=None):
    """Refuse to run one model's decode against another's artifacts."""
    built = read_stamp(out_dir)
    if built is not None and built != model.name:
        raise SystemExit(
            f"the decode artifacts in {out_dir or airsrc.fused_decode_dir()} were "
            f"built for {built!r}, not {model.name!r}. They share a directory and "
            f"a filename, so rebuild before switching:\n"
            f"  make compile-decode MODEL={model.name}"
        )


def lower_template(L, out_dir=None, output_format="xclbin", model=None):
    """Lower one decode template -- the binary + insts pair for context length L.

    Returns their paths. Runs entirely in this process: the op scopes the
    builder's environment to one build and restores it, so several context
    lengths -- or several models -- can be built one after another without
    forking, which is what this used to do.
    """
    model = model or registry.spec()
    fd_dir = airsrc.fused_decode_dir()
    out_dir = out_dir or fd_dir

    kdir = decode_kernels.kernel_dir(model, fd_dir)
    objs = [os.path.join(kdir, o) for o in KERNEL_OBJECTS]
    missing = [o for o in objs if not os.path.exists(o)]
    if missing:
        # The op refuses missing objects too, but this says how to get them.
        raise SystemExit(
            "missing decode kernel objects:\n  "
            + "\n  ".join(missing)
            + "\n\nBuild them first (they are Peano compiles, not AIR):\n"
            f"  make compile-decode MODEL={model.name}"
        )

    from triton.language.extra.npu import fused_decode

    cfg = model.decode_config(L)
    print(f"[decode-build] L={L} {cfg}", flush=True)
    t0 = time.time()
    art = fused_decode(
        cfg,
        fd_dir,
        kernel_objects=objs,
        output_format=output_format,
        name=f"fused_decode_{model.name}_L{L}",
    )
    print(
        f"[decode-build] L={L} ATTN_MAXL={art['attn_maxl']} lowered in "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )

    # Name them the way the decode's template loader expects to find them.
    ext = "pdi" if output_format == "pdi" else "xclbin"
    dst_x = os.path.join(out_dir, f"decode_L{L}.{ext}")
    dst_i = os.path.join(out_dir, f"decode_L{L}.insts.bin")
    shutil.copyfile(art["bin_path"], dst_x)
    shutil.copyfile(art["insts_path"], dst_i)
    with open(os.path.join(out_dir, STAMP), "w") as f:
        f.write(model.name + "\n")
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
        "--model",
        default=registry.DEFAULT,
        help=f"model family ({', '.join(sorted(registry.SPECS))})",
    )
    ap.add_argument(
        "--format",
        default="xclbin",
        choices=("xclbin", "pdi"),
        help="pdi is what the HSA runtime consumes; xclbin is XRT's",
    )
    args = ap.parse_args(argv)
    model = registry.spec(args.model)

    if args.context_length is not None:
        lower_template(args.context_length, args.out_dir, args.format, model)
        return 0

    # The loader wants the L template and an L-1 slope reference. Both in this
    # process: the builder's geometry used to be import-time state that a
    # second build in the same process would have inherited, which is why this
    # forked. `tl.extra.npu.fused_decode` takes the configuration as arguments
    # and scopes the environment to one build, so it no longer can.
    for L in (args.max_context_length, args.max_context_length - 1):
        lower_template(L, args.out_dir, args.format, model)
    print(
        "[decode-build] both templates lowered through tl.extra.npu.fused_decode.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
