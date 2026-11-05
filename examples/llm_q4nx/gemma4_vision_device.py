# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4's vision tower GEMMs as one NPU device of Triton kernels.

One `gemma4_gemm.mm_g` op per GEMM shape of the tower (q|k|v, o, gate with
GELU in its drain, up, down, the patch embedding and the projection into the
language space), linked into one device by `FusedDevice` and called on a
`FusedProgram` one 128-row chunk at a time. The weights are bf16, as the
checkpoint ships them.

`TritonOps` implements the device-ops interface of mlir-air's `VisionEncoder`
(`gemma4_vision_air`): build, gemm, attention, timings. The host pipeline is
mlir-air's. Attention runs `gemma4_vision_attention`'s Triton kernel on its
own device, or with `attention="air"` mlir-air's FlashAttention ELF, which
isolates the GEMMs when comparing against mlir-air's `NpuOps`.

E2B and E4B share every shape but the projection's, so the device carries
both widths and serves either model.
"""

import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import gemma4_gemm as G
import gemma4_vision_q4nx as vw
import npu_device_cache as cache

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

        import gemma4_vision_air
        import gemma4_vision_attention as VA

        enc = gemma4_vision_air._air_vision()
        self.kv_block = enc.NpuOps.kv_block
        if attention == "triton":
            prog = FusedProgram(VA.device()["build_dir"], VA.NAME)
            self._attn = _TritonAttention(VA.Attention(prog))
        elif attention == "air":
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

    def release(self):
        """Close the device's and the attention's hardware contexts."""
        import gemma4_vision_air

        self.prog.device.close()
        gemma4_vision_air._release(self._attn)
        return True

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

    # ---- one 128-row chunk at a time, for `pipelined_encoder`. Each role
    # (the GEMM input or output it holds) has buffers of its own, so a chunk
    # of one never waits on a chunk of another.

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


def pipelined_encoder(E):
    """mlir-air's `VisionEncoder` (module `E`) with its layers run 128 rows at
    a time: while the NPU works on one chunk's GEMM, the host runs the norms,
    clips and GLU of a chunk whose GEMM is done and starts its next GEMM.
    Every host op is row-wise; only the attention needs every row at once.
    The next layer's q|k|v GEMMs start as soon as a chunk leaves this layer."""
    H = E.H

    class PipelinedEncoder(E.VisionEncoder):
        def _layer(self, sh, i, h, cos, sin, b):
            ops, P = self.ops, self.layers[i]
            c, nm = P["clamp"], P["norms"]
            eps, valid = E.RMS_EPS, sh.L_valid
            d, inter, nqkv = h.shape[1], P["up"].shape[1], P["qkv"].shape[1]
            chunks = list(enumerate(range(0, sh.L_pad, G.M)))

            def live(r0, n):
                return min(max(n - r0, 0), G.M)

            def to_qkv(j, r0, li):
                L, x = self.layers[li], b["x"][r0 : r0 + G.M]
                H.rms_clip_bf16(
                    h[r0 : r0 + G.M],
                    live(r0, valid),
                    L["norms"]["input_ln"],
                    eps,
                    *L["clamp"]["q"][0],
                    x,
                )
                ops.put("x", x, j)
                ops.start("qkv", "x", "qkv", L["qkv"], f"qkv{li}", j)

            # The Triton attention takes its Q and K/V a chunk at a time, while
            # the q|k|v GEMMs of later chunks run; any other is given them whole.
            att = getattr(ops._attn, "attn", None)
            if att is not None:
                att.prepare(sh.L_pad, sh.kv_rows, valid)
            qscale = float(E.ATTN_SCALE * np.sqrt(E.DK_AUG))
            with ops.prog:
                if i == 0:
                    for j, r0 in chunks:
                        to_qkv(j, r0, 0)
                for j, r0 in chunks:
                    kv = live(r0, sh.kv_rows)
                    H.qkv_prep(
                        ops.out("qkv", nqkv, j),
                        live(r0, valid),
                        E.N_HEADS,
                        E.DH,
                        E.DK_AUG,
                        nm["q_norm"],
                        nm["k_norm"],
                        eps,
                        cos[r0 : r0 + G.M],
                        sin[r0 : r0 + G.M],
                        P["qkv_bounds"],
                        qscale,
                        E._BIG,
                        b["q"][r0 : r0 + G.M],
                        b["k"][r0 : r0 + kv],
                        b["v"][r0 : r0 + kv],
                    )
                    if att is not None:
                        att.put_q(b["q"][r0 : r0 + G.M], r0)
                        if kv:
                            att.put_kv(b["k"][r0 : r0 + kv], b["v"][r0 : r0 + kv], r0)
            if att is not None:
                t0 = time.perf_counter()
                att.run()
                attn = att.result()
                ops._attn.t += time.perf_counter() - t0
            else:
                attn = ops.attention(b["q"], b["k"], b["v"], 1.0 / np.sqrt(E.DK_AUG))
                attn = np.ascontiguousarray(attn, bfloat16)
            with ops.prog:
                for j, r0 in chunks:
                    a = attn[r0 : r0 + G.M]
                    H.clip_bf16(a, *c["o"][0])
                    ops.put("attn", a, j)
                    ops.start("o", "attn", "o", P["o"], f"o{i}", j)
                for j, r0 in chunks:
                    rows, x = h[r0 : r0 + G.M], b["x"][r0 : r0 + G.M]
                    n = live(r0, valid)
                    o = ops.out("o", d, j)
                    H.add_rms_clip(rows, o, n, nm["post_attn_ln"], eps, *c["o"][1])
                    H.rms_clip_bf16(rows, n, nm["pre_ffn_ln"], eps, *c["gate"][0], x)
                    ops.put("x", x, j)
                    ops.start("gate", "x", "gate", P["gate"], f"gate{i}", j)
                    ops.start("up", "x", "up", P["up"], f"up{i}", j)
                for j, r0 in chunks:
                    f = b["f"][r0 : r0 + G.M]
                    g, u = ops.out("gate", inter, j), ops.out("up", inter, j)
                    H.glu_clip(g, u, P["gate_hi"], *c["up"][1], *c["down"][0], f)
                    ops.put("f", f, j)
                    ops.start("down", "f", "down", P["down"], f"down{i}", j)
                last = i + 1 == self.n_layers
                for j, r0 in chunks:
                    y = ops.out("down", d, j)
                    H.add_rms_clip(
                        h[r0 : r0 + G.M],
                        y,
                        live(r0, valid),
                        nm["post_ffn_ln"],
                        eps,
                        *c["down"][1],
                    )
                    if not last:
                        to_qkv(j, r0, i + 1)

    return PipelinedEncoder


def tower(
    weights=None,
    model=None,
    build_dir=None,
    verbose=False,
    attention="triton",
    pipelined=True,
):
    """mlir-air's tower pipeline on `TritonOps`: `encode(pixels, pos_ids)`.
    `pipelined` runs the layers 128 rows at a time (`pipelined_encoder`)."""
    import gemma4_vision_air

    return gemma4_vision_air.AirVisionTower(
        weights,
        model,
        ops=TritonOps(build_dir, verbose=verbose, attention=attention),
        encoder=pipelined_encoder if pipelined else None,
    )
