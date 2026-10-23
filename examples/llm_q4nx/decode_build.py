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
    python decode_build.py --format elf         # one full ELF for every L

The full ELF is what the HSA decode runs on a ROCR that can resolve the device
address of a host allocation: the context length reaches it as scratchpad
parameters, so one artifact serves every L and there is no template pair to
dispatch. `--format elf` and `--format pdi` also build the xclbin pair, which
mlir-air's decoder loads at construction.

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
# consumes them, and builds only the variants for a split attention.
KERNEL_OBJECTS = (
    "proj_qmm.o",
    "rms_residual.o",
    "glu.o",
    "rope.o",
    "attn_qk.ll",
    "attn_kv.ll",
)

#: What each engine links beyond those six. Mirrors
#: `decode_kernels.ENGINE_KERNELS` in object form: the PLE fork's own `ple.o`,
#: which the shared engine has no herd for.
ENGINE_OBJECTS = {
    "fused_decode": (),
    "ple": ("ple.o",),
}

DEFAULT_L = 2048

#: The full ELF's artifacts, written beside the templates.
ELF_NAME = "decode_scratchpad.elf"
ELF_PARAMS = "decode_scratchpad.params.txt"
ELF_MAXL = "decode_scratchpad.maxl"
#: Which sequence argument each runtime scalar is, which mlir-air's ElfDecode
#: computes the scratchpad parameters from.
ELF_ARGS = "decode_scratchpad.args"
#: The model the ELF was built for. Separate from STAMP, which every template
#: build rewrites while leaving the ELF in place.
ELF_MODEL = "decode_scratchpad.model"


#: Written beside the artifacts, naming the model that built them.
#:
#: mlir-air's loader resolves the decode directory as a fixed path, so every
#: model's `decode_L2048.xclbin` lands on the same name. Loading one model's
#: artifacts under another does not fail -- it decodes to fluent nonsense --
#: so the stamp is what turns that into an error. See `check_stamp`.
STAMP = "decode_model.txt"


def read_stamp(out_dir=None, engine="fused_decode"):
    """The model whose artifacts are in `out_dir`, or None if unstamped."""
    path = os.path.join(out_dir or airsrc.fused_decode_dir(engine), STAMP)
    try:
        with open(path) as f:
            return f.read().strip() or None
    except OSError:
        return None


def check_stamp(model, out_dir=None):
    """Refuse to run one model's decode against another's artifacts."""
    built = read_stamp(out_dir, model.engine)
    if built is not None and built != model.name:
        raise SystemExit(
            f"the decode artifacts in "
            f"{out_dir or airsrc.fused_decode_dir(model.engine)} were "
            f"built for {built!r}, not {model.name!r}. They share a directory and "
            f"a filename, so rebuild before switching:\n"
            f"  make compile-decode MODEL={model.name}"
        )


def check_elf_stamp(model, out_dir=None):
    """Refuse to run the full ELF built for another model.

    The ELF keeps its own stamp because a template build rewrites `STAMP` and
    leaves the ELF in place, so `check_stamp` passing says nothing about it.
    """
    out_dir = out_dir or airsrc.fused_decode_dir(model.engine)
    try:
        with open(os.path.join(out_dir, ELF_MODEL)) as f:
            built = f.read().strip()
    except OSError:
        built = None
    if built != model.name:
        what = f"was built for {built!r}" if built else "has no model stamp"
        raise SystemExit(
            f"the full ELF in {out_dir} {what}, not {model.name!r}; rebuild it:\n"
            f"  make compile-decode MODEL={model.name} DECODE_ELF=1"
        )


def lower_template(L, out_dir=None, output_format="xclbin", model=None):
    """Lower one decode template -- the binary + insts pair for context length L.

    Returns their paths. Runs entirely in this process: the op scopes the
    builder's environment to one build and restores it, so several context
    lengths -- or several models -- can be built one after another without
    forking, which is what this used to do.
    """
    model = model or registry.spec()
    fd_dir = airsrc.fused_decode_dir(model.engine)
    out_dir = out_dir or fd_dir

    from triton.language.extra.npu import fused_decode

    if output_format == "elf" and model.engine != "fused_decode":
        raise SystemExit(
            f"--format elf builds the shared engine's decode; {model.name} uses "
            f"the {model.engine!r} engine, which has no full-ELF build."
        )
    cfg = model.decode_config(L, dynseq=1 if output_format == "elf" else None)

    # A build that splits attention links kernels compiled for its part count;
    # those are built here, from the cache when nothing changed.
    parts = decode_kernels.kv_parts(model, cfg)
    if parts and parts > 1:
        decode_kernels.build(model, parts=parts)
    kdir = decode_kernels.kernel_dir(model, fd_dir, parts)
    objs = [
        os.path.join(kdir, o) for o in KERNEL_OBJECTS + ENGINE_OBJECTS[model.engine]
    ]
    missing = [o for o in objs if not os.path.exists(o)]
    if missing:
        # The op refuses missing objects too, but this says how to get them.
        raise SystemExit(
            "missing decode kernel objects:\n  "
            + "\n  ".join(missing)
            + "\n\nBuild them first (they are Peano compiles, not AIR):\n"
            f"  make compile-decode MODEL={model.name}"
        )
    print(f"[decode-build] L={L} {cfg}", flush=True)
    t0 = time.time()
    art = fused_decode(
        cfg,
        fd_dir,
        kernel_objects=objs,
        output_format=output_format,
        name=f"fused_decode_{model.name}_L{L}"
        + ("_elf" if output_format == "elf" else ""),
    )
    print(
        f"[decode-build] L={L} ATTN_MAXL={art['attn_maxl']} lowered in "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )

    if output_format == "elf":
        return _copy_elf(art, out_dir, model)

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


def _copy_elf(art, out_dir, model):
    """Copy the full ELF, its parameter list, its scalar arguments and its
    ATTN_MAXL into `out_dir`."""
    # aiecc writes params.txt beside the ELF. Without it the host cannot find
    # the parameters in the scratchpad, so a build that emitted none is an error.
    params = os.path.join(os.path.dirname(art["elf_path"]), "params.txt")
    if not os.path.exists(params):
        raise SystemExit(
            f"{params} is missing: the build emitted no scratchpad parameters, "
            "so the context length has no way to reach the device. It needs an "
            "mlir-air whose fused_decode honours DECODE_DYNSEQ."
        )
    scalars = art.get("seq_scalar_args") or {}
    if "L" not in scalars:
        raise SystemExit(
            "the build reported no sequence argument for the context length, so "
            f"{ELF_ARGS} cannot be written. It needs an mlir-air whose "
            "fused_decode records SEQ_SCALAR_ARGS."
        )
    dst_e = os.path.join(out_dir, ELF_NAME)
    shutil.copyfile(art["elf_path"], dst_e)
    shutil.copyfile(params, os.path.join(out_dir, ELF_PARAMS))
    # The format mlir-air's own ELF build writes: the readback takes whole
    # blocks, one part, and no window skip.
    with open(os.path.join(out_dir, ELF_ARGS), "w") as f:
        for name, n in scalars.items():
            f.write(f"arg {n} {name}\n")
        f.write("align 1\nparts 1\n")
    # Recorded, not defaulted: a driver assuming another ATTN_MAXL mis-sizes the
    # KV cache and the mask threshold.
    with open(os.path.join(out_dir, ELF_MAXL), "w") as f:
        f.write(f"{art['attn_maxl']}\n")
    with open(os.path.join(out_dir, ELF_MODEL), "w") as f:
        f.write(model.name + "\n")
    with open(os.path.join(out_dir, STAMP), "w") as f:
        f.write(model.name + "\n")
    print(f"[decode-build] -> {dst_e} + {ELF_PARAMS}", flush=True)
    return dst_e, os.path.join(out_dir, ELF_PARAMS)


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
        choices=("xclbin", "pdi", "elf"),
        help="elf: one full ELF for every context length, for the HSA decode on "
        "a ROCR that resolves device addresses; pdi: the template pair as PDIs; "
        "xclbin: XRT's. elf and pdi also build the xclbin pair",
    )
    args = ap.parse_args(argv)
    model = registry.spec(args.model)

    if args.context_length is not None:
        lower_template(args.context_length, args.out_dir, args.format, model)
        return 0

    # The ELF takes L at dispatch, so one build serves every context length.
    if args.format == "elf":
        lower_template(args.max_context_length, args.out_dir, "elf", model)
    else:
        _lower_pair(args.max_context_length, args.out_dir, args.format, model)

    # The HSA decoders subclass mlir-air's, which loads the xclbin pair to set
    # up weights and geometry even though HSA dispatches the ELF or PDIs.
    if args.format != "xclbin":
        _lower_pair(args.max_context_length, args.out_dir, "xclbin", model)
    return 0


def _lower_pair(L, out_dir, output_format, model):
    """The template at L and the L-1 build the loader takes the slope from."""
    for l in (L, L - 1):
        lower_template(l, out_dir, output_format, model)
    print(
        f"[decode-build] both {output_format} templates lowered through "
        "tl.extra.npu.fused_decode.",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
