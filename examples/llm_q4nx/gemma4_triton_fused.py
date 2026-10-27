# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4-E2B's fused one-device NPU prefill, every op from Triton.

mlir-air's fused prefill (`--prefill-engine air-fused`) runs every op of a
128-token chunk on one device. This builds the same device from Triton: every
GEMM from `gemma4_gemm.mm_g` and the attention ops from
`gemma4_attention.attn`, linked into one module by `fused_device.FusedDevice`.
The LM head is `gemma4_lm_gemv.gemv_q4` on a device of its own, on the buffers
of mlir-air's `lm_gemv`. The host library comes from mlir-air's builder, and the
manifest tells mlir-air's runtime that the GEMMs read bf16 operands and take
one chunk per dispatch, so the runtime drives both devices unchanged.
"""

import contextlib
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
LM_SCHEDULE = HERE / "transform_gemma4_lm_gemv_aie2p.mlir"
#: What aircc builds the LM head with, beyond the defaults.
LM_AIRCC_ARGS = ("--air-channel-multiplexing=L1", "--omit-ping-pong-transform=L1")
SPEC = "air_examples.llms.gemma4_e2b_q4nx.fused_prefill.spec"
#: This module's own inputs, beyond mlir-air's sources.
_SOURCES = (
    HERE / "gemma4_gemm.py",
    HERE / "gemma4_attention.py",
    HERE / "gemma4_lm_gemv.py",
    HERE / "fused_device.py",
    SCHEDULE,
    HERE / "transform_gemma4_attention_aie2p.mlir.in",
    LM_SCHEDULE,
    Path(__file__),
)


def _toolchain():
    """Peano and mlir-aie, as `tl.extra.npu.fused_prefill` resolves them."""
    from triton.backends.amd_triton_npu.driver import find_peano_root
    from triton.language.extra.npu.fused_prefill import _mlir_aie_root

    peano = find_peano_root()
    if not peano:
        raise RuntimeError("no Peano (llvm-aie) install found for the AIE kernels")
    return str(peano), _mlir_aie_root()


def fingerprint():
    """Digest of mlir-air's fused prefill sources and toolchain, of the files
    here that make the device, and of the Triton and NPU backend that compile
    it."""
    import triton
    import triton.backends.amd_triton_npu as backend
    from triton.language.extra.npu.fused_prefill import PrefillConfig
    from triton.language.extra.npu.fused_prefill import fingerprint as air_fingerprint

    import airsrc

    h = hashlib.sha256()
    config = PrefillConfig("gemma4-e2b")
    h.update(
        air_fingerprint(config, airsrc.fused_prefill_dir(), *_toolchain()).encode()
    )
    h.update(json.dumps([AIRCC_ARGS, LM_AIRCC_ARGS, STACK_SIZE]).encode())
    h.update(triton.__version__.encode())
    backend_dir = Path(backend.__file__).parent
    sources = sorted(p for p in backend_dir.rglob("*") if p.suffix in (".py", ".mlir"))
    for p in [*_SOURCES, *sources]:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


@contextlib.contextmanager
def _air_build_env():
    """mlir-air's builder reads the toolchain from the environment and leaves
    the working directory where it compiled."""
    peano, aie = _toolchain()
    saved = {
        k: os.environ.get(k) for k in ("PEANO_INSTALL_DIR", "MLIR_AIE_INSTALL_DIR")
    }
    cwd = os.getcwd()
    os.environ.update(PEANO_INSTALL_DIR=peano, MLIR_AIE_INSTALL_DIR=aie)
    try:
        yield
    finally:
        os.chdir(cwd)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _build_lm(out, lm_gemv):
    """The LM head as `lm.xclbin` and `lm.insts.bin` in `out`, for the buffers
    of mlir-air's `lm_gemv` module."""
    import torch
    from triton.backends.amd_triton_npu.config import config_context
    from triton.backends.amd_triton_npu.driver import _aircc_compile, _ttshared_to_air
    from triton.backends.amd_triton_npu.driver_scope import driver_scope

    import gemma4_lm_gemv as L

    G = lm_gemv
    w = torch.zeros(G.NC, G.TILES * G.KC * G.NR * G.PKT, dtype=torch.uint8)
    x = torch.zeros(G.K, dtype=torch.bfloat16)
    y = torch.zeros(G.V, dtype=torch.bfloat16)
    with driver_scope("npu"), config_context(compile_only=True):
        c = L.gemv_q4.warmup(*L.args(G, w, x, y), grid=(1, 1, 1))
    with config_context(transform_tiling_script=str(LM_SCHEDULE)):
        m = _ttshared_to_air(c.asm["ttsharedir"], 1, 1, 1)
    work = out / "work" / "lm"
    work.mkdir(parents=True, exist_ok=True)
    path = work / "lm.air.mlir"
    path.write_text(str(m))
    # aircc writes its project into the working directory
    os.chdir(work)
    art = _aircc_compile(
        str(path),
        "xclbin",
        "npu2",
        str(work),
        bf16_emulation=False,
        extra_args=LM_AIRCC_ARGS,
        runtime_loop_tiling_sizes=(),
    )
    shutil.copy(art["bin_path"], out / "lm.xclbin")
    shutil.copy(art["insts_path"], out / "lm.insts.bin")


def add_attention(dev, spec, heads):
    """Add the spec's attention ops to `dev`, from `gemma4_attention.attn`,
    with the interface of mlir-air's device: the GEMM's buffers, then each
    attention group's q, KV and o in config order; an attention op takes
    (nkv, q0, k0) at run time."""
    import torch

    import gemma4_attention as A
    from air_examples.llms.shared.fused_prefill import build as B
    from air_examples.llms.shared.fused_prefill import device as D

    for op, groups in D.attn_ops(spec.CFG).items():
        if len(groups) != 1:
            raise NotImplementedError(f"attention op {op} drives {groups}")
        (grp,) = groups
        g = spec.CFG.attn[grp]
        rounds = heads * (D.M // (A.NQT.value * g.lkp))
        q = torch.zeros(rounds * A.NQT.value * g.lkp, g.dh, dtype=torch.bfloat16)
        kv = torch.zeros(2 * g.lkp * g.dh, dtype=torch.bfloat16)
        dev.add(
            B.seq_name(op, (heads,)),
            A.attn,
            (1,),
            A.args(
                g,
                rounds,
                1,
                0,
                0,
                q,
                kv,
                torch.zeros_like(q),
                rows=D.M,
                window=(spec.CFG.windows or {}).get(op, g.window),
            ),
            transform_script=str(A.schedule(g.lkp, g.dh, g.dvt, dev.proj)),
            slot=3 + 3 * list(spec.CFG.attn).index(grp),
            runtime=("nk", "q0", "k0"),
            key=f"attn_{grp}",
            pin_l2=True,
            # K and V come into L2 through one channel, as the records of the
            # KV buffer hold them: its memtile has no DMA channel to spare.
            multiplex=("L2",),
        )


def build(out):
    """Build the device, its kernels and its LM head into `out`, which then
    loads like a build of mlir-air's."""
    import importlib

    import torch
    from triton.backends.amd_triton_npu.config import config_context

    import airsrc
    import gemma4_gemm as G
    from fused_device import FusedDevice

    airsrc.add_air_paths("gemma4_e2b_q4nx")
    from air_examples.llms.shared.fused_prefill import build as B
    from air_examples.llms.shared.fused_prefill import device as D
    from air_examples.llms.shared.fused_prefill import lm_gemv

    spec = importlib.import_module(SPEC)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with _air_build_env():
        B.build_kernels(out, spec.CFG)
        _build_lm(out, lm_gemv)
    heads = spec.ATTN_POINTS["base"][0]
    dev = FusedDevice("fused_prefill", str(out / "proj"))
    with config_context(bf16_emulation=True):
        add_attention(dev, spec, heads)
        for k, n, act, wq in spec.gemm_shapes():
            a = torch.zeros(128, k, dtype=torch.bfloat16)
            c = torch.zeros(128, n, dtype=torch.bfloat16)
            w = torch.zeros(n // 64, k // (128 if wq else 32), 5120, dtype=torch.uint8)
            # one chunk per dispatch: there are no ops for chunk pairs
            dev.add(
                B.op_name("g", (k, n, act, wq, 0)),
                G.mm_g,
                (1,),
                G.args(a, w, c, n, k, act, wq, D.k_pad(k)),
                transform_script=str(SCHEDULE),
            )
    xclbin, header = dev.compile(
        runtime_loop_tiling_sizes=(),
        bf16_emulation=True,
        extra_args=AIRCC_ARGS,
        stack_size=STACK_SIZE,
    )
    shutil.copy(xclbin, out / "fused_prefill.xclbin")
    shutil.copy(header, out / "fused_prefill.txn.h")
    manifest = dict(
        gemm=[B.op_name("g", (*s, 0)) for s in spec.gemm_shapes()],
        attn={a: B.seq_name(a, (heads,)) for a in D.attn_ops(spec.CFG)},
        heads=heads,
        lm="lm",
        gemm_format="bf16",
        gemm_pairs=False,
        device="triton",
        max_len=getattr(spec, "max_len", None),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for scratch in ("proj", "work"):
        shutil.rmtree(out / scratch, ignore_errors=True)


def triton_fused_prefill(cache_root=None):
    """Build (or reuse) the device. Returns ``build_dir`` and ``cached``, as
    `tl.extra.npu.fused_prefill`."""
    import fcntl

    from triton.language.extra.npu.fused_prefill import _complete, default_cache_root

    root = Path(cache_root or default_cache_root())
    root.mkdir(parents=True, exist_ok=True)
    current = root / f"gemma4-e2b-triton_{fingerprint()}"
    with open(root / f".{current.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if current.is_dir() and _complete(current):
            return dict(build_dir=str(current), cached=True)
        tmp = Path(tempfile.mkdtemp(prefix=f"{current.name}.", dir=root))
        try:
            build(tmp)
            if not _complete(tmp):
                raise RuntimeError(f"Triton fused prefill build incomplete: {tmp}")
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        if current.exists():
            shutil.rmtree(current)
        os.replace(tmp, current)
    return dict(build_dir=str(current), cached=False)
