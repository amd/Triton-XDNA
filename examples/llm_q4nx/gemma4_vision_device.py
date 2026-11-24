# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4's vision tower on this repository's devices: its GEMMs one NPU
device of Triton kernels, its attention another, and the work between them
Triton kernels on the CPU (`VisionTower`).

One `gemma4_gemm.mm_g` op per GEMM shape of the tower (q|k|v, o, gate with
GELU in its drain, up, down, the patch embedding and the projection into the
language space), linked into one device by `FusedDevice` and called on a
`FusedProgram` one 128-row chunk at a time. The weights are bf16, as the
checkpoint ships them.

`TritonOps` runs the GEMMs, a 128-row chunk at a time or a whole array, with
the device-ops interface of mlir-air's `VisionEncoder` (`gemma4_vision_air`):
build, gemm, attention, timings. Attention runs `gemma4_vision_attention`'s
Triton kernel on its own device, or with `attention="air"` mlir-air's
FlashAttention ELF, which isolates the GEMMs when comparing against
mlir-air's `NpuOps`.

E2B and E4B share every shape but the projection's, so the device carries
both widths and serves either model.
"""

import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import torch

import gemma4_gemm as G
import gemma4_vision_attention as VA
import gemma4_vision_q4nx as vw
import npu_device_cache as cache
import vision_host_kernels as K

HERE = Path(__file__).resolve().parent
SCHEDULE = HERE / "transform_gemma4_gemm_aie2p.mlir"
#: As the text prefill's device builds its GEMMs.
AIRCC_ARGS = (
    "--air-channel-multiplexing=L1",
    "--omit-ping-pong-transform=L1",
    "--fuse-sequential-loops",
    "--use-lock-race-condition-fix-auto=false",
)
STACK_SIZE = 4096
NAME = "vision"
FILES = tuple(f"{NAME}.{x}" for x in ("xclbin", "txn.h", "ops.json"))
#: The language widths the projection may have: E2B's and E4B's.
OUT_DIMS = (1536, 2560)
_SOURCES = (HERE / "gemma4_gemm.py", SCHEDULE, Path(__file__))


def gemm_shapes():
    """(k, n padded, act) of every GEMM of the tower; act 1 drains through
    GELU."""
    d, inter = vw.HIDDEN, vw.INTER
    s = {(d, 3 * d, 0), (d, d, 0), (d, inter, 1), (d, inter, 0), (inter, d, 0)}
    s |= {(d, n, 0) for n in OUT_DIMS}
    return sorted((k, G.n_pad(n), act) for k, n, act in s)


def gemm_name(k, n, act):
    return f"v_{k}_{G.n_pad(n)}_{act}"


def build(out):
    """The device, `vision` in `out`."""
    import torch
    from triton.backends.amd_triton_npu.config import config_context
    from triton.backends.amd_triton_npu.fused_device import FusedDevice

    out = Path(out)
    dev = FusedDevice(NAME, str(out / "proj"))
    with config_context(bf16_emulation=True):
        for k, n, act in gemm_shapes():
            a = torch.zeros(G.M, k, dtype=torch.bfloat16)
            c = torch.zeros(G.M, n, dtype=torch.bfloat16)
            w = torch.zeros(n // 64, k // 32, 5120, dtype=torch.uint8)
            dev.add(
                gemm_name(k, n, act),
                G.mm_g,
                (1,),
                G.args(a, w, c, n, k, act, 0, G.k_pad(k)),
                transform_script=str(SCHEDULE),
            )
    with cache.toolchain_env():
        dev.compile(
            runtime_loop_tiling_sizes=(),
            bf16_emulation=True,
            extra_args=AIRCC_ARGS,
            stack_size=STACK_SIZE,
        )
    dev.save(out)


def vision_device(cache_root=None):
    """Build (or reuse) the device. Returns `build_dir` and `cached`."""
    digest = cache.fingerprint(_SOURCES, [AIRCC_ARGS, STACK_SIZE, OUT_DIMS])
    return cache.cached("gemma4-vision-triton", digest, FILES, build, cache_root)


class TritonOps:
    """mlir-air `VisionEncoder` device ops: the GEMMs on this device, the
    attention on the Triton attention device (`attention="triton"`) or on
    mlir-air's FlashAttention ELF (`"air"`)."""

    def __init__(self, build_dir=None, verbose=False, attention="triton"):
        from triton.backends.amd_triton_npu.fused_program import FusedProgram

        if attention == "triton":
            prog = FusedProgram(VA.device()["build_dir"], VA.NAME)
            self._attn = _TritonAttention(VA.Attention(prog))
        elif attention == "air":
            import gemma4_vision_air

            enc = gemma4_vision_air._air_vision()
            self._attn = _attention_ops(enc)(verbose=verbose)
        else:
            raise ValueError(attention)
        built = vision_device() if build_dir is None else dict(build_dir=build_dir)
        self.prog = FusedProgram(built["build_dir"], NAME)
        self.sites, self.a, self.c = {}, {}, {}
        self.ra, self.rc = {}, {}
        self.op_t = {}

    def build(self, shape, m_soft, out_dim):
        if out_dim not in OUT_DIMS:
            raise ValueError(f"the device holds projections to {OUT_DIMS}")
        self._attn.build(shape, m_soft, out_dim)

    def contexts(self):
        """The devices holding hardware contexts: the GEMMs', and the Triton
        attention's (mlir-air's ELFs reopen theirs on their own)."""
        attn = getattr(self._attn, "attn", None)
        return [self.prog.device] + ([attn.prog.device] if attn else [])

    def release(self):
        """Close the devices' hardware contexts, keeping their buffers. True
        if any were open."""
        open_ = any(d.ctx is not None for d in self.contexts())
        for d in self.contexts():
            d.close()
        if not hasattr(self._attn, "attn"):
            import gemma4_vision_air

            open_ |= bool(gemma4_vision_air._release(self._attn))
        return open_

    def resume(self):
        """Reopen what `release` closed."""
        for d in self.contexts():
            if d.ctx is None:
                d.open()

    # ---- buffers

    def _weights(self, site, w):
        """`w` [k, n] bf16 as this site's packets, written once."""
        if site not in self.sites:
            k = w.shape[0]
            packed = G.pack_bf16(np.ascontiguousarray(np.asarray(w).T))
            kp = packed.shape[1] * G.k_pad(k) // k
            packed = np.pad(packed, ((0, 0), (0, kp - packed.shape[1]), (0, 0)))
            buf = self.prog.empty(packed.shape, np.uint8)
            buf.host(write=True)[...] = packed
            self.sites[site] = buf
        return self.sites[site]

    def _a(self, k, c):
        if (k, c) not in self.a:
            shape = (G.M // G.TM, G.k_pad(k) // G.TK, G.TM, G.TK)
            self.a[k, c] = self.prog.zeros(shape, bfloat16)
        return self.a[k, c]

    def _c(self, npd, c):
        if (npd, c) not in self.c:
            self.c[npd, c] = self.prog.empty((G.M, npd), bfloat16)
        return self.c[npd, c]

    # ---- one 128-row chunk at a time, for `VisionTower`. Each role
    # (the GEMM input or output it holds) has buffers of its own, so a chunk
    # of one never waits on a chunk of another.

    def tiles(self, role, k, c):
        """Chunk `c` of `role`'s input tiles, [M / TM, K / TK, TM, TK] for k
        columns, for a host kernel to write in place."""
        key = (role, c)
        if key not in self.ra:
            shape = (G.M // G.TM, G.k_pad(k) // G.TK, G.TM, G.TK)
            self.ra[key] = self.prog.zeros(shape, bfloat16)
        return self.ra[key]

    def put(self, role, rows, c):
        """Row-major `rows` [M, k] as chunk `c` of `role`'s input tiles."""
        k = rows.shape[1]
        key = (role, c)
        if key not in self.ra:
            shape = (G.M // G.TM, G.k_pad(k) // G.TK, G.TM, G.TK)
            self.ra[key] = self.prog.zeros(shape, bfloat16)
        tiles = rows.reshape(G.M // G.TM, G.TM, k // G.TK, G.TK)
        self.ra[key].host(write=True)[:, : k // G.TK] = tiles.transpose(0, 2, 1, 3)

    def start(self, op, src, dst, w, site, c):
        """Start chunk `c` of `src`'s input @ `w` into `dst`'s output."""
        k, n = w.shape
        npd = G.n_pad(n)
        if (dst, c) not in self.rc:
            self.rc[dst, c] = self.prog.empty((G.M, npd), bfloat16)
        G.mm_g[(1,)](
            *G.args(
                self.ra[src, c],
                self._weights(site, w),
                self.rc[dst, c],
                npd,
                k,
                int(op == "gate"),
                0,
                G.k_pad(k),
            )
        )

    def out(self, role, n, c):
        """Chunk `c` of `role`'s output, [M, n] bf16, once its GEMM is done.
        A view of the device buffer: read it before the next start into it."""
        return self.rc[role, c].host()[:, :n]

    # ---- the interface

    def gemm(self, op, x, w, site=None):
        """`x` [m, k] @ `w` [k, n], m a multiple of 128 -> [m, n] bf16."""
        t0 = time.perf_counter()
        m, k = x.shape
        n = w.shape[1]
        npd, act = G.n_pad(n), int(op == "gate")
        wb = self._weights(site or op, w)
        x = np.ascontiguousarray(x, bfloat16)
        chunks = range(m // G.M)
        with self.prog:
            for c in chunks:
                # A tiled [M / 32, K / 128, 32, 128]
                tiles = x[c * G.M : (c + 1) * G.M].reshape(
                    G.M // G.TM, G.TM, k // G.TK, G.TK
                )
                self._a(k, c).host(write=True)[:, : k // G.TK] = tiles.transpose(
                    0, 2, 1, 3
                )
                G.mm_g[(1,)](
                    *G.args(
                        self._a(k, c), wb, self._c(npd, c), npd, k, act, 0, G.k_pad(k)
                    )
                )
            y = np.empty((m, n), bfloat16)
            for c in chunks:
                y[c * G.M : (c + 1) * G.M] = self._c(npd, c).host()[:, :n]
        tag = (site or op).rstrip("0123456789")
        self.op_t[tag] = self.op_t.get(tag, 0.0) + time.perf_counter() - t0
        return y

    def attention(self, q, k, v, scale):
        return self._attn.attention(q, k, v, scale)

    def timings(self):
        return {**self.op_t, **self._attn.timings()}


class _TritonAttention:
    """`gemma4_vision_attention.Attention` as the encoder's attention ops."""

    def __init__(self, attn):
        self.attn = attn
        self.t = 0.0

    def build(self, shape, m_soft, out_dim):
        pass

    def attention(self, q, k, v, scale):
        t0 = time.perf_counter()
        out = self.attn.attention(q, k, v, scale)
        self.t += time.perf_counter() - t0
        return out

    def release(self):
        self.attn.prog.device.close()
        return True

    def timings(self):
        return {"attn": self.t}


def _attention_ops(enc):
    """mlir-air's `NpuOps`, building and running only its attention."""

    class AttentionOps(enc.NpuOps):
        def __init__(self, verbose=False):
            super().__init__(verbose=verbose)
            self.cache.max_contexts = 1

        def build(self, shape, m_soft, out_dim):
            import os

            stages = enc._fa_stages(shape.kv_rows)
            self.names["fa"] = f"fa_{shape.L_pad}x{shape.kv_rows}_s{stages}"
            if self.names["fa"] in self.cache.artifacts:
                return
            cwd = os.getcwd()
            build = self.dir / "build"
            build.mkdir(parents=True, exist_ok=True)
            os.chdir(build)
            try:
                self._compile_fa(shape, stages)
            finally:
                os.chdir(cwd)
                self.cache._save_manifest()

    return AttentionOps


class VisionTower:
    """Gemma4's vision tower on this repository's devices:
    `encode(pixels, pos_ids)` -> `[n_soft, out_dim]` f32 soft tokens.

    The GEMMs run on `TritonOps`' device and the attention on
    `gemma4_vision_attention`'s; everything between them is a Triton kernel
    on the CPU (`vision_host_kernels`), which writes the next GEMM's A tiles
    in place. The layers run 128 rows at a time: while the NPU works on one
    chunk's GEMM, the host runs the norms, clips and GLU of a chunk whose
    GEMM is done and starts its next GEMM. Every host op is row-wise; only the
    attention needs every row at once, and the next layer's q|k|v GEMMs start
    as soon as a chunk leaves this one.

    The math is mlir-air's tower's (`llms/gemma4_e2b_vision`): each GEMM a
    ClippableLinear, its output clamp applied where its output is read and
    its input clamp where its input is written, rounded to bf16 once.
    """

    def __init__(self, weights=None, model=None, build_dir=None, verbose=False):
        from triton.backends.amd_triton_npu import driver

        w = weights if weights is not None else vw.load_vision(model)
        self.fingerprint = w.get("fingerprint")
        self.ops = TritonOps(build_dir, verbose=verbose)
        self.att = self.ops._attn.attn
        self.out_dim = w["embedding_proj"].shape[1]
        self.layers = [_layer_weights(L) for L in w["layers"]]
        self.patch_w = np.ascontiguousarray(w["patch_embd"], bfloat16)
        self.proj_w = np.ascontiguousarray(w["embedding_proj"], bfloat16)
        self.table = np.ascontiguousarray(w["position_table"], np.float32)
        half = vw.HEAD_DIM // 2
        self.inv_freq = (
            1.0 / (vw.ROPE_THETA ** (np.arange(0, half, 2, dtype=np.float32) / half))
        ).astype(np.float32)
        # The tower holds hardware contexts, and the prefill and decode need
        # theirs. Registered so a refused launch can reclaim them.
        driver.context_releasers.append(self.release)

    def release(self):
        """Close the tower's hardware contexts, keeping its buffers; the next
        encode reopens them (`TritonOps.resume`). True if any were open."""
        return self.ops.release()

    def timings(self):
        return self.ops.timings()

    def encode(self, pixels, pos_ids):
        self.ops.resume()
        pos = np.asarray(pos_ids, np.int64)
        pad = (pos == -1).all(axis=-1)
        L, valid = len(pad), int((~pad).sum())
        assert not pad[:valid].any(), "padding patches must trail the image"
        rows = _round_up(L, 256)
        kv_rows = _round_up(valid, VA.LKP)
        if kv_rows > VA.MAX_BLOCKS * VA.LKP:
            raise ValueError(
                f"{valid} patches exceed the attention device's "
                f"{VA.MAX_BLOCKS * VA.LKP} keys; use a smaller soft-token budget"
            )
        pos_rows = np.full((rows, 2), -1, np.int32)
        pos_rows[:L] = pos
        d = vw.HIDDEN

        # the patch embedding, and each patch's x and y position embeddings
        x = np.empty((rows, d), bfloat16)
        K.pixel_rows(np.ascontiguousarray(pixels, np.float32), x, L)
        y = self.ops.gemm("o", x, self.patch_w, "patch")
        h = np.empty((rows, d), np.float32)
        K.add_positions(h, y, pos_rows, self.table, L, valid)

        cs, sn = K.rope_tables(pos_rows, self.inv_freq, vw.HEAD_DIM)
        bufs = dict(
            q=np.empty((rows, vw.N_HEADS * DK_AUG), bfloat16),
            k=np.empty((kv_rows, vw.N_HEADS * DK_AUG), bfloat16),
            v=np.empty((kv_rows, vw.N_HEADS * vw.HEAD_DIM), bfloat16),
        )
        for i in range(len(self.layers)):
            self._layer(i, h, cs, sn, bufs, rows, kv_rows, valid)

        members, n_soft = _pool_members(pos[:L], valid)
        m_soft = _round_up(n_soft, G.M)
        x = np.empty((m_soft, d), bfloat16)
        K.pool_norm(h, members, x, POOL_K * POOL_K, np.sqrt(d), vw.RMS_EPS)
        out = self.ops.gemm("proj", x, self.proj_w, "proj")[:n_soft]
        return torch.from_numpy(np.asarray(out, np.float32))

    def _layer(self, i, h, cs, sn, b, rows, kv_rows, valid):
        ops, att, P = self.ops, self.att, self.layers[i]
        c, nm, eps = P["clip"], P["norms"], vw.RMS_EPS
        d, inter = vw.HIDDEN, vw.INTER
        chunks = list(enumerate(range(0, rows, G.M)))

        def live(r0, n):
            return min(max(n - r0, 0), G.M)

        def to_qkv(j, r0, li):
            L = self.layers[li]
            K.rms_clip_tile(
                h[r0 : r0 + G.M], L["norms"]["norm1"], ops.tiles("x", d, j),
                live(r0, valid), d, eps, *L["clip"]["q"][:2],
            )  # fmt: skip
            ops.start("qkv", "x", "qkv", L["qkv"], f"qkv{li}", j)

        att.prepare(rows, kv_rows, valid)
        with ops.prog:
            if i == 0:
                for j, r0 in chunks:
                    to_qkv(j, r0, 0)
            for j, r0 in chunks:
                kv = live(r0, kv_rows)
                K.qkv_prep(
                    ops.out("qkv", 3 * d, j), cs[r0 : r0 + G.M], sn[r0 : r0 + G.M],
                    nm["q_norm"], nm["k_norm"], b["q"][r0 : r0 + G.M],
                    b["k"][r0 : r0 + kv], b["v"][r0 : r0 + kv], live(r0, valid),
                    eps, QSCALE, BIG, P["qkv_bounds"], vw.N_HEADS, vw.HEAD_DIM,
                )  # fmt: skip
                att.put_q(b["q"][r0 : r0 + G.M], r0)
                if kv:
                    att.put_kv(b["k"][r0 : r0 + kv], b["v"][r0 : r0 + kv], r0)
        t0 = time.perf_counter()
        att.run()
        attn = att.result()
        ops._attn.t += time.perf_counter() - t0
        with ops.prog:
            for j, r0 in chunks:
                K.clip_tile(
                    attn[r0 : r0 + G.M], ops.tiles("attn", d, j), d, *c["o"][:2]
                )
                ops.start("o", "attn", "o", P["o"], f"o{i}", j)
            for j, r0 in chunks:
                n = live(r0, valid)
                K.add_rms_clip(
                    h[r0 : r0 + G.M], ops.out("o", d, j), nm["post_attn_norm"],
                    n, eps, *c["o"][2:],
                )  # fmt: skip
                K.rms_clip_tile(
                    h[r0 : r0 + G.M], nm["norm2"], ops.tiles("x", d, j), n, d,
                    eps, *c["gate"][:2],
                )  # fmt: skip
                ops.start("gate", "x", "gate", P["gate"], f"gate{i}", j)
                ops.start("up", "x", "up", P["up"], f"up{i}", j)
            for j, r0 in chunks:
                K.glu_clip_tile(
                    ops.out("gate", inter, j), ops.out("up", inter, j),
                    ops.tiles("f", inter, j), inter, P["gate_hi"],
                    *c["up"][2:], *c["down"][:2],
                )  # fmt: skip
                ops.start("down", "f", "down", P["down"], f"down{i}", j)
            last = i + 1 == len(self.layers)
            for j, r0 in chunks:
                K.add_rms_clip(
                    h[r0 : r0 + G.M], ops.out("down", d, j), nm["ffn_post_norm"],
                    live(r0, valid), eps, *c["down"][2:],
                )  # fmt: skip
                if not last:
                    to_qkv(j, r0, i + 1)


#: The attention device's heads are widened from 64 to DK_AUG columns, the
#: extra ones carrying the padding keys' mask; it scales scores by
#: 1/sqrt(DK_AUG) where the model's scale is 1, so q is scaled up to match.
DK_AUG = 2 * vw.HEAD_DIM
QSCALE = float(np.sqrt(DK_AUG))
#: a padding key's score: low enough that softmax gives it nothing
BIG = 1.0e4
POOL_K = vw.POOLING_KERNEL_SIZE
#: GELU-tanh's value is within 2e-5 of 0 below this, so a gate whose output
#: clamp's lower bound is under it loses nothing to the bound's being dropped
GATE_LO_MAX = -4.4


def _round_up(x, m):
    return (x + m - 1) // m * m


def _gelu_tanh(x):
    """ACT2FN["gelu_pytorch_tanh"] of one value, in f32."""
    x = np.asarray(np.float32(x))
    return float(
        0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    )


def _layer_weights(L):
    """A layer's GEMM weights ([in, out] bf16, q|k|v fused), clamps and norms,
    as the tower takes them."""
    clip = {p: tuple(float(b) for b in L["clip"][p]) for p in L["clip"]}
    # q, k and v share one input, gate and up another: one input clamp each.
    assert clip["q"][:2] == clip["k"][:2] == clip["v"][:2], clip
    assert clip["gate"][:2] == clip["up"][:2], clip
    assert clip["gate"][2] <= GATE_LO_MAX, clip["gate"]
    return dict(
        qkv=np.ascontiguousarray(
            np.concatenate([np.asarray(L[p], bfloat16) for p in ("q", "k", "v")], 1)
        ),
        **{
            p: np.ascontiguousarray(L[p], bfloat16) for p in ("o", "gate", "up", "down")
        },
        clip=clip,
        qkv_bounds=tuple(b for p in ("q", "k", "v") for b in clip[p][2:]),
        gate_hi=_gelu_tanh(clip["gate"][3]),
        norms={
            n: np.ascontiguousarray(L[n], np.float32)
            for n in (
                "norm1",
                "post_attn_norm",
                "norm2",
                "ffn_post_norm",
                "q_norm",
                "k_norm",
            )
        },
    )


def _pool_members(pos, valid):
    """The pooler's slots: each POOL_K x POOL_K block of patch positions a
    slot, its real patches' rows (in order, -1 after) a row of `members`;
    and the number of slots any patch falls in, which are the soft tokens.
    Padding patches sit at (0, 0) and so add nothing to the slots there."""
    p = np.maximum(pos, 0)
    kern = p // POOL_K
    max_x = p[:, 0].max() + 1
    slot = kern[:, 0] + (max_x // POOL_K) * kern[:, 1]
    used = np.unique(slot)
    index = np.full(slot.max() + 1, -1, np.int64)
    index[used] = np.arange(len(used))
    members = np.full((len(used), POOL_K * POOL_K), -1, np.int32)
    fill = np.zeros(len(used), np.int64)
    for r in range(valid):
        s = index[slot[r]]
        members[s, fill[s]] = r
        fill[s] += 1
    return members, len(used)


def tower(weights=None, model=None, build_dir=None, verbose=False):
    """The tower on this repository's devices: `encode(pixels, pos_ids)`."""
    return VisionTower(weights, model, build_dir, verbose)
