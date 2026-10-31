# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4-E2B's fused one-device NPU prefill, every op from Triton.

Every op of a chunk runs on one device: each GEMM shape from
`gemma4_gemm.mm_g` and both attention ops from `gemma4_attention.attn`,
linked into one module by `FusedDevice`, a runtime sequence per op. The LM
head is `gemma4_lm_gemv.gemv_q4` on a device of its own. `gemma4_npu_prefill`
runs both on `npu_runner`, with the math between the ops on the host
(`host_kernels`).

The ops' names, buffers and run-time arguments are defined here, for both
the build and the runner:
- a GEMM `gemm_name(k, n, act, wq)` takes (A tiles, weight packets, C) as
  device buffers 0, 1 and 2;
- attention op `attn_name(op)` takes each of its groups' q, KV records and
  o, the group at buffer `ATTN_SLOT[group]`, and (nk, q0, k0) at run time.
"""

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import config as C
import gemma4_attention as A
import gemma4_gemm as G
import gemma4_lm_gemv as LG

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
#: The files that make the devices, beyond the NPU backend.
_SOURCES = (
    HERE / "gemma4_gemm.py",
    HERE / "gemma4_attention.py",
    HERE / "gemma4_lm_gemv.py",
    SCHEDULE,
    HERE / "transform_gemma4_attention_aie2p.mlir.in",
    LM_SCHEDULE,
    Path(__file__),
)
#: Backend modules that run a build but take no part in making it.
_RUNTIME_ONLY = ("fused_program.py", "npu_runner.py")
#: What a complete build holds.
FILES = tuple(
    f"{d}.{x}" for d in ("fused_prefill", "lm") for x in ("xclbin", "txn.h", "ops.json")
)

#: Rows of a chunk.
M = G.M
HEADS = C.N_Q_HEADS
#: The attention groups: each op, full and sliding-window attention, splits
#: its heads over two columns, whose cores run both ops.
GROUPS = {
    "f0": A.Attn(col=0, dh=C.DH_GLOBAL, lkp=16),
    "s0": A.Attn(col=0, dh=C.DH_SLIDING, lkp=32),
    "f1": A.Attn(col=7, dh=C.DH_GLOBAL, lkp=16),
    "s1": A.Attn(col=7, dh=C.DH_SLIDING, lkp=32),
}
#: Attention op -> the groups one dispatch drives.
OPS = {"f": ("f0", "f1"), "s": ("s0", "s1")}
WINDOWS = {"s": C.SLIDING_WINDOW}
#: Device buffer of each group's q; its KV records and o follow.
ATTN_SLOT = {g: 3 + 3 * i for i, g in enumerate(GROUPS)}
#: Device buffers: a GEMM's three, then each group's three.
NARGS = 3 + 3 * len(GROUPS)
V_CHUNKS = A.v_chunks(GROUPS.values())
#: Bytes of one LM head weight packet.
LM_PKT = LG.GN * LG.GK // 2 + 2 * 2 * (LG.GK // 32) * LG.GN

if C.N_KV_HEADS != 1:
    # More KV heads per column need their records max_blocks apart, so a
    # build fixes the longest prompt; not done here.
    raise NotImplementedError(f"{C.N_KV_HEADS} KV heads")


def gemm_name(k, n, act, wq):
    """The op of a GEMM k deep and n wide (padded to whole rounds)."""
    return f"g_{k}_{G.n_pad(n)}_{act}_{wq}"


def attn_name(op):
    return f"attn_{op}"


def attn_qr(g):
    """Rounds per head of a chunk: a round is one head's q rows over a
    column's cores."""
    return M // (A.NQT.value * g.lkp)


def gemm_shapes():
    """(k, n, act, wq) of every Gemma4 prefill GEMM: act 1 drains through
    GELU, wq 1 reads int4 weights (else bf16)."""
    d, s = C.D, set()
    for dh in (C.DH_SLIDING, C.DH_GLOBAL):
        dq = C.N_Q_HEADS * dh
        s |= {(d, dq, 0, 1), (d, dh * C.N_KV_HEADS, 0, 1), (dq, d, 0, 1)}
    for inter in (C.INTER, C.INTER_WIDE):
        s |= {(d, inter, 1, 1), (d, inter, 0, 1), (inter, d, 0, 1)}
    s |= {(d, C.PLI_D, 1, 0), (C.PLI_D, d, 0, 0), (d, C.N_LAYERS * C.PLI_D, 0, 0)}
    return sorted((k, G.n_pad(n), act, wq) for k, n, act, wq in s)


def _toolchain():
    """Peano and mlir-aie, as the NPU backend resolves them."""
    from triton.backends.amd_triton_npu.driver import find_peano_root
    from triton.language.extra.npu.fused_prefill import _mlir_aie_root

    peano = find_peano_root()
    if not peano:
        raise RuntimeError("no Peano (llvm-aie) install found for the AIE kernels")
    return str(peano), _mlir_aie_root()


def _versions():
    """The installed compiler stack, by distribution."""
    import importlib.metadata as md

    out = []
    for names in (
        ("mlir_air", "mlir-air"),
        ("mlir_aie_no_rtti", "mlir_aie"),
        ("llvm-aie",),
    ):
        for n in names:
            try:
                out.append(f"{n}=={md.version(n)}")
                break
            except md.PackageNotFoundError:
                pass
    return out


def fingerprint():
    """Digest of what the devices are made from: the files here, the NPU
    backend and Triton that compile them, and the compiler stack (its
    versions and where it is installed)."""
    import triton
    import triton.backends.amd_triton_npu as backend

    h = hashlib.sha256()
    h.update(json.dumps([AIRCC_ARGS, LM_AIRCC_ARGS, STACK_SIZE]).encode())
    h.update(json.dumps([triton.__version__, *_versions(), *_toolchain()]).encode())
    backend_dir = Path(backend.__file__).parent
    sources = sorted(
        p
        for p in backend_dir.rglob("*")
        if p.suffix in (".py", ".mlir") and p.name not in _RUNTIME_ONLY
    )
    for p in [*_SOURCES, HERE.parent / "gemma4_e2b_q4nx" / "config.py", *sources]:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


@contextlib.contextmanager
def _toolchain_env():
    """aircc reads the toolchain from the environment and leaves the working
    directory where it compiled."""
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


def build_lm(out):
    """The LM head, device `lm` in `out`: `gemma4_lm_gemv.gemv_q4` on the
    weight stream (`gemma4_lm_gemv.pack`), x bf16 [K] and the logits' output
    (`gemma4_lm_gemv.unpack_y`)."""
    import torch
    from triton.backends.amd_triton_npu.fused_device import FusedDevice

    w = torch.zeros(LG.NC, LG.TILES * LG.KC * LG.NR * LM_PKT, dtype=torch.uint8)
    x = torch.zeros(LG.K, dtype=torch.bfloat16)
    y = torch.zeros(LG.V, dtype=torch.bfloat16)
    dev = FusedDevice("lm", str(Path(out) / "work" / "lm"))
    dev.add("lm", LG.gemv_q4, (1,), LG.args(w, x, y), transform_script=str(LM_SCHEDULE))
    with _toolchain_env():
        dev.compile(
            runtime_loop_tiling_sizes=(),
            bf16_emulation=False,
            extra_args=LM_AIRCC_ARGS,
        )
    dev.save(out)


def attn_args(op, bufs, nk, q0, k0):
    """`gemma4_attention.attn`'s arguments for attention op `op` on `bufs`,
    each group's q, KV records and o in `GROUPS` order: the op runs only its
    own groups' rounds, so the other groups' buffers are not touched."""
    f, s = (GROUPS[OPS[o][0]] for o in ("f", "s"))
    hg = HEADS // len(OPS[op])
    rounds = {o: hg * attn_qr(GROUPS[OPS[o][0]]) * (o == op) for o in OPS}
    return A.args(
        f, s, rounds["f"], rounds["s"], nk, q0, k0, bufs,
        rows=M, window=WINDOWS.get("s"),
    )  # fmt: skip


def call_attention(op, bufs, nk, q0, k0):
    """Run attention op `op` (inside its `FusedProgram`)."""
    A.attn[(1,)](*attn_args(op, bufs, nk, q0, k0))


def add_attention(dev):
    """Add both attention ops to `dev`, from `gemma4_attention.attn`: each
    drives its groups' heads, (nk, q0, k0) at run time."""
    import torch

    hg = HEADS // len(OPS["f"])
    bufs = []
    for g in GROUPS.values():
        q = torch.zeros(hg * M, g.dh, dtype=torch.bfloat16)
        bufs += [q, torch.zeros(2 * g.lkp * g.dh, dtype=torch.bfloat16), q.clone()]
    script = A.schedule(*(GROUPS[OPS[o][0]] for o in ("f", "s")), dev.proj)
    for op in OPS:
        dev.add(
            attn_name(op),
            A.attn,
            (1,),
            attn_args(op, bufs, 1, 0, 0),
            transform_script=str(script),
            slot=ATTN_SLOT[next(iter(GROUPS))],
            runtime=("nk", "q0", "k0"),
            key="attn",
            pin_l2=True,
            # K and V come into L2 through one channel, as the records of the
            # KV buffer hold them: its memtile has no DMA channel to spare.
            multiplex=("L2",),
        )


def add_gemm(dev, k, n, act, wq):
    """Add the GEMM op for (k, n, act, wq), n padded."""
    import torch

    a = torch.zeros(M, k, dtype=torch.bfloat16)
    c = torch.zeros(M, n, dtype=torch.bfloat16)
    w = torch.zeros(n // 64, k // (128 if wq else 32), 5120, dtype=torch.uint8)
    dev.add(
        gemm_name(k, n, act, wq),
        G.mm_g,
        (1,),
        G.args(a, w, c, n, k, act, wq, G.k_pad(k)),
        transform_script=str(SCHEDULE),
    )


def compile_device(dev):
    """`dev` compiled as the prefill device is: (xclbin, TXN builder header)."""
    return dev.compile(
        runtime_loop_tiling_sizes=(),
        bf16_emulation=True,
        extra_args=AIRCC_ARGS,
        stack_size=STACK_SIZE,
    )


def build(out):
    """Build the device and its LM head into `out`."""
    from triton.backends.amd_triton_npu.config import config_context
    from triton.backends.amd_triton_npu.fused_device import FusedDevice

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    build_lm(out)
    dev = FusedDevice("fused_prefill", str(out / "proj"))
    with config_context(bf16_emulation=True):
        add_attention(dev)
        for shape in gemm_shapes():
            add_gemm(dev, *shape)
    compile_device(dev)
    dev.save(out)
    for scratch in ("proj", "work"):
        shutil.rmtree(out / scratch, ignore_errors=True)


def complete(build_dir):
    return all((Path(build_dir) / f).is_file() for f in FILES)


def triton_fused_prefill(cache_root=None):
    """Build (or reuse) the devices. Returns ``build_dir`` and ``cached``."""
    import fcntl

    from triton.language.extra.npu.fused_prefill import default_cache_root

    root = Path(cache_root or default_cache_root())
    root.mkdir(parents=True, exist_ok=True)
    current = root / f"gemma4-e2b-triton_{fingerprint()}"
    with open(root / f".{current.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if current.is_dir() and complete(current):
            return dict(build_dir=str(current), cached=True)
        tmp = Path(tempfile.mkdtemp(prefix=f"{current.name}.", dir=root))
        try:
            build(tmp)
            if not complete(tmp):
                raise RuntimeError(f"Triton fused prefill build incomplete: {tmp}")
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        if current.exists():
            shutil.rmtree(current)
        os.replace(tmp, current)
    return dict(build_dir=str(current), cached=False)
