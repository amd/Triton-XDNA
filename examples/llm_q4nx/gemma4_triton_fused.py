# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4-E2B's fused one-device NPU prefill with the GEMMs in Triton.

mlir-air's fused prefill (`--prefill-engine air-fused`) runs every op of a
128-token chunk on one device. This builds the same device with every GEMM
compiled from `gemma4_gemm.mm_g` instead of mlir-air's GEMM herd, and the
attention ops from mlir-air's own builder, linked into one module by
`fused_device.FusedDevice`. The rest of the build (the LM head, the host
library, the kernel objects) is mlir-air's, taken from its build, so its
runtime drives the device unchanged.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEDULE = HERE / "transform_gemma4_gemm_aie2p.mlir"
#: What aircc builds the device with, beyond the defaults.
AIRCC_ARGS = (
    "--air-channel-multiplexing=L1",
    "--omit-ping-pong-transform=L1",
    # The int4 and bf16 K loops of mm_g share their channels and buffers.
    "--fuse-sequential-loops",
    # The device's plain lock scheme; the automatic lock race fix gives its
    # output buffers chain locks, which cost BDs and time.
    "--use-lock-race-condition-fix-auto=false",
)
STACK_SIZE = 4096
#: This module's own inputs, beyond mlir-air's build it extends.
_SOURCES = (HERE / "gemma4_gemm.py", HERE / "fused_device.py", SCHEDULE, Path(__file__))


def fingerprint(air_build_dir):
    """Digest of mlir-air's build this one extends and of the files here that
    make the device."""
    h = hashlib.sha256()
    h.update(Path(air_build_dir).resolve().name.encode())
    h.update(json.dumps(AIRCC_ARGS).encode() + str(STACK_SIZE).encode())
    for p in _SOURCES:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build(air_build_dir, out):
    """Build the device into `out` from mlir-air's Gemma4 build at
    `air_build_dir`; `out` then loads like that build."""
    import torch
    from triton.backends.amd_triton_npu.config import config_context

    import airsrc
    import gemma4_gemm as G
    from fused_device import FusedDevice

    airsrc.add_air_paths("gemma4_e2b_q4nx")
    from air_examples.llms.shared.fused_prefill import device as D
    from air_examples.llms.shared.fused_prefill.build import op_name, seq_name
    from air_examples.llms.gemma4_e2b_q4nx.fused_prefill import spec

    ref, out = Path(air_build_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    heads = spec.ATTN_POINTS["base"][0]
    dev = FusedDevice("fused_prefill", str(out / "proj"))
    for op in D.attn_ops(spec.CFG):
        name = seq_name(op, (heads,))
        lch = D.build(spec.CFG, op, heads, 1, 0, 0, name=name, own=True)
        dev.add_air(name, str(lch.build(target="npu2")))
    with config_context(bf16_emulation=True):
        for k, n, act, wq in spec.gemm_shapes():
            a = torch.zeros(128, k, dtype=torch.bfloat16)
            c = torch.zeros(128, n, dtype=torch.bfloat16)
            w = torch.zeros(n // 64, k // (128 if wq else 32), 5120, dtype=torch.uint8)
            dev.add(
                op_name("g", (k, n, act, wq)),
                G.mm_g,
                (1,),
                G.args(a, w, c, n, k, act, wq),
                transform_script=str(SCHEDULE),
            )
    objects = sorted(str(p) for p in ref.glob("*.o"))
    xclbin, header = dev.compile(
        objects=objects,
        runtime_loop_tiling_sizes=(),
        bf16_emulation=True,
        extra_args=AIRCC_ARGS,
        stack_size=STACK_SIZE,
    )
    for p in ref.iterdir():
        if p.suffix in (".o", ".so") or p.name.startswith("lm."):
            shutil.copy(p, out / p.name)
    shutil.copy(xclbin, out / "fused_prefill.xclbin")
    shutil.copy(header, out / "fused_prefill.txn.h")
    man = json.loads((ref / "manifest.json").read_text())
    man["device"] = "triton"
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    shutil.rmtree(out / "proj", ignore_errors=True)


def triton_fused_prefill(air_build_dir, cache_root=None):
    """Build (or reuse) the device for mlir-air's build at `air_build_dir`.
    Returns ``build_dir`` and ``cached``, as `tl.extra.npu.fused_prefill`."""
    import fcntl

    from triton.language.extra.npu.fused_prefill import _complete, default_cache_root

    root = Path(cache_root or default_cache_root())
    root.mkdir(parents=True, exist_ok=True)
    current = root / f"gemma4-e2b-triton_{fingerprint(air_build_dir)}"
    with open(root / f".{current.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if current.is_dir() and _complete(current):
            return dict(build_dir=str(current), cached=True)
        tmp = Path(tempfile.mkdtemp(prefix=f"{current.name}.", dir=root))
        try:
            build(air_build_dir, tmp)
            if not _complete(tmp):
                raise RuntimeError(f"Triton fused prefill build incomplete: {tmp}")
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        if current.exists():
            shutil.rmtree(current)
        os.replace(tmp, current)
    return dict(build_dir=str(current), cached=False)
