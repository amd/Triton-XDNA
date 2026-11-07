# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""A dense Q4NX model's fused one-device NPU prefill, every op from Triton.

The device mlir-air's `shared/fused_prefill` builds for Llama, Qwen3, Phi-4
and Gemma3 (one model table, `models.MODELS`), built here from Triton
kernels: each GEMM shape from `gemma4_gemm.mm_g`, the LM head as GEMMs of at
most MAX_N columns, and attention from `dense_attention.attn` (or, for a
model with sliding-window layers, `gemma4_attention.attn`'s full and
sliding ops), linked into one module by `FusedDevice`, a runtime sequence
per op. `dense_npu_prefill` runs it, with the math between the ops on the
host.

The ops' names, buffers and run-time arguments are defined here, for both the
build and the runner:
- a GEMM `gemm_name(k, n, act, wq)` takes (A tiles, weight packets, C) as
  device buffers 0, 1 and 2;
- an attention op `attn_name(op)` takes each column's group's q, KV records
  and o, from buffer ATTN_SLOT on, and (nk, q0, k0) at run time. Each
  column's group holds half the KV heads, each head's records max_blocks
  apart, so a build holds prompts up to its `max_len`.
"""

import json
from pathlib import Path
from typing import NamedTuple

import dense_attention as DA
import gemma4_attention as A
import gemma4_gemm as G
import npu_device_cache as cache

HERE = Path(__file__).resolve().parent
SCHEDULE = HERE / "transform_gemma4_gemm_aie2p.mlir"
#: What aircc builds the device with, beyond the defaults: as Gemma4's.
AIRCC_ARGS = (
    "--air-channel-multiplexing=L1",
    "--omit-ping-pong-transform=L1",
    "--fuse-sequential-loops",
    "--use-lock-race-condition-fix-auto=false",
)
STACK_SIZE = 4096
#: The files that make the device, beyond the NPU backend.
_SOURCES = (
    HERE / "gemma4_gemm.py",
    HERE / "gemma4_attention.py",
    HERE / "dense_attention.py",
    SCHEDULE,
    HERE / "transform_gemma4_attention_aie2p.mlir.in",
    Path(__file__),
)
FILES = ("fused_prefill.xclbin", "fused_prefill.txn.h", "fused_prefill.ops.json")
FILES += ("manifest.json",)

#: Rows of a chunk.
M = G.M
#: Widest GEMM op: the LM head is split into ops of at most this many
#: columns, as mlir-air splits it.
MAX_N = 32 * G.NR
#: Device buffer of the first column's q; its KV records and o follow, then
#: the other column's.
ATTN_SLOT = 3


class Shape(NamedTuple):
    """What the device is built for, from the model's entry in mlir-air's
    table (`models.Desc`) and the longest prompt."""

    d: int
    dh: int
    heads: int
    kv_heads: int
    inter: int
    vocab: int
    tied: bool
    gemma: bool
    window: int
    lkp: int
    max_len: int

    @classmethod
    def of(cls, desc, max_len=None):
        return cls(
            desc.d, desc.dh, desc.heads, desc.kv_heads, desc.inter, desc.vocab,
            desc.tied, desc.gemma, desc.window, desc.lkp, max_len or desc.max_len,
        )  # fmt: skip


def check(desc):
    """Refuse a model the device cannot run as mlir-air's would."""
    why = [
        n
        for n, bad in (
            ("its family", desc.family != "dense"),
            ("its codec", desc.codec != "q4nx"),
            ("q / k / v biases", desc.qkv_bias),
            ("post-norms only", desc.post_norm),
            ("q / k norms over all heads", desc.qk_norm_rows),
            ("a padded head dim", bool(desc.attn_dh)),
            ("an odd number of KV heads", desc.kv_heads % 2),
        )
        if bad
    ]
    if why:
        raise NotImplementedError(f"{desc.repo}: {', '.join(why)}")


def n_split(n):
    """[n0, n1) column ranges of a GEMM n wide, each at most MAX_N."""
    return [(n0, min(n, n0 + MAX_N)) for n0 in range(0, n, MAX_N)]


def gemm_name(k, n, act, wq):
    """The op of a GEMM k deep and n wide (padded to whole rounds)."""
    return f"g_{k}_{G.n_pad(n)}_{act}_{wq}"


def gemm_shapes(s):
    """(k, n, act, wq) of every GEMM, n padded: act 1 drains through the
    model's activation (SiLU, or GELU on Gemma3), wq 1 reads int4 weights
    (else bf16)."""
    dq, dk = s.heads * s.dh, s.kv_heads * s.dh
    shapes = {
        (s.d, dq + 2 * dk, 0, 1),  # q | k | v
        (dq, s.d, 0, 1),  # o
        (s.d, s.inter, 1, 1),  # gate
        (s.d, s.inter, 0, 1),  # up
        (s.inter, s.d, 0, 1),  # down
    }
    shapes |= {(s.d, n1 - n0, 0, 0 if s.tied else 1) for n0, n1 in n_split(s.vocab)}
    return sorted((k, G.n_pad(n), act, wq) for k, n, act, wq in shapes)


def group(s):
    """A column's attention group: half the KV heads, records max_blocks
    apart (the cores read K blocks in pairs, so one past the last)."""
    return A.Attn(
        col=0,
        dh=s.dh,
        lkp=s.lkp,
        kv_heads=s.kv_heads // 2,
        max_blocks=-(-s.max_len // s.lkp) + 1,
    )


def attn_ops(s):
    """The attention ops: full and sliding-window ("f", "s") on a model with
    windows, else one ("a")."""
    return ("f", "s") if s.window else ("a",)


def attn_name(op):
    return f"attn_{op}"


def attn_rounds(s):
    """Rounds of a column's group per chunk: a round is one head's q rows
    over a column's cores."""
    return s.heads // 2 * M // (A.NQT.value * s.lkp)


def attn_args(s, op, bufs, nk, q0, k0):
    """The arguments of attention op `op` on `bufs`: each column's q, KV
    records and o. With windows, `gemma4_attention.attn` runs the op's
    groups' rounds and none of the other op's, which take the same
    buffers."""
    g, nr = group(s), attn_rounds(s)
    if not s.window:
        return DA.args(g, nr, nk, q0, k0, bufs, rows=M)
    c0, c1 = bufs[:3], bufs[3:]
    return A.args(
        g, g, nr * (op == "f"), nr * (op == "s"), nk, q0, k0,
        (*c0, *c0, *c1, *c1), rows=M, window=s.window,
    )  # fmt: skip


def call_attention(s, op, bufs, nk, q0, k0):
    """Run attention op `op` (inside its `FusedProgram`)."""
    kernel = A.attn if s.window else DA.attn
    kernel[(1,)](*attn_args(s, op, bufs, nk, q0, k0))


def add_attention(dev, s):
    import torch

    g = group(s)
    bufs = []
    for _ in range(2):
        q = torch.zeros(s.heads // 2 * M, s.dh, dtype=torch.bfloat16)
        kv = torch.zeros(g.kv_heads * g.max_blocks, 2 * g.lkp * g.dh)
        bufs += [q, kv.to(torch.bfloat16), q.clone()]
    if s.window:
        script = A.schedule(g, g, dev.proj)
    else:
        script = DA.schedule(g, dev.proj)
    for op in attn_ops(s):
        dev.add(
            attn_name(op),
            A.attn if s.window else DA.attn,
            (1,),
            attn_args(s, op, bufs, 2, 0, 0),
            transform_script=str(script),
            slot=ATTN_SLOT,
            runtime=("nk", "q0", "k0"),
            key="attn",
            pin_l2=True,
            # K and V come into L2 through one channel, as the records of the
            # KV buffer hold them: its memtile has no DMA channel to spare.
            multiplex=("L2",),
        )


def gemm_args(s, a, w, c, n, k, act, wq):
    return G.args(a, w, c, n, k, act, wq, G.k_pad(k), silu=not s.gemma)


def add_gemm(dev, s, k, n, act, wq):
    import torch

    a = torch.zeros(M, k, dtype=torch.bfloat16)
    c = torch.zeros(M, n, dtype=torch.bfloat16)
    w = torch.zeros(n // 64, k // (128 if wq else 32), 5120, dtype=torch.uint8)
    dev.add(
        gemm_name(k, n, act, wq),
        G.mm_g,
        (1,),
        gemm_args(s, a, w, c, n, k, act, wq),
        transform_script=str(SCHEDULE),
    )


def build(out, s):
    from triton.backends.amd_triton_npu.config import config_context
    from triton.backends.amd_triton_npu.fused_device import FusedDevice

    out = Path(out)
    dev = FusedDevice("fused_prefill", str(out / "proj"))
    with config_context(bf16_emulation=True):
        add_attention(dev, s)
        for shape in gemm_shapes(s):
            add_gemm(dev, s, *shape)
    with cache.toolchain_env():
        dev.compile(
            runtime_loop_tiling_sizes=(),
            bf16_emulation=True,
            extra_args=AIRCC_ARGS,
            stack_size=STACK_SIZE,
        )
    dev.save(out)
    (out / "manifest.json").write_text(json.dumps(dict(s._asdict())))
    import shutil

    shutil.rmtree(out / "proj", ignore_errors=True)


def triton_fused_prefill(name, desc, max_len=None, cache_root=None):
    """Build (or reuse) the device of model `name` (`desc`, its entry in
    mlir-air's table) for prompts up to `max_len`, default the table's.
    Returns `build_dir` and `cached`."""
    check(desc)
    s = Shape.of(desc, max_len)
    digest = cache.fingerprint(_SOURCES, [s, AIRCC_ARGS, STACK_SIZE])
    return cache.cached(
        f"{name}-triton", digest, FILES, lambda out: build(out, s), cache_root
    )
