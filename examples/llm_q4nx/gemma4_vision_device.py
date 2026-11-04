# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4's vision tower GEMMs as one NPU device of Triton kernels.

One `gemma4_gemm.mm_g` op per GEMM shape of the tower -- q|k|v, o, gate (GELU
in its drain), up, down, the patch embedding and the projection into the
language space -- linked into one device by `FusedDevice` and called on a
`FusedProgram`, a 128-row chunk at a time. The weights are bf16, as the
checkpoint ships them.

`TritonOps` is the device-ops interface of mlir-air's `VisionEncoder`
(`gemma4_vision_air`): build, gemm, attention, timings. With it the tower's
host pipeline is mlir-air's, its attention `gemma4_vision_attention`'s Triton
kernel on a device of its own -- or, with `attention="air"`, mlir-air's
FlashAttention ELF, so a comparison against mlir-air's own `NpuOps` measures
the GEMMs alone.

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


def tower(weights=None, model=None, build_dir=None, verbose=False, attention="triton"):
    """mlir-air's tower pipeline on `TritonOps`: `encode(pixels, pos_ids)`."""
    import gemma4_vision_air

    return gemma4_vision_air.AirVisionTower(
        weights, model, ops=TritonOps(build_dir, verbose=verbose, attention=attention)
    )
