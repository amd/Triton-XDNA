# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""`SuperkernelDecoder`: a Gemma4 model's decode as one Triton superkernel
dispatch a token (`wave.py`), behind the interface of mlir-air's
`FusedDecoder`, so that mlir-air's `generate()` drives it unchanged. The
model is the one mlir-air's Gemma4 modules are set up for (GEMMA4_VARIANT).

The first dispatch compiles the superkernel (cached by Triton afterwards) and
runs the token; its buffers then stay on the device, and each later token
writes only the token's inputs and reads back the logits.
"""

import os
import time

import numpy as np
import torch
from ml_dtypes import bfloat16

HERE = os.path.dirname(os.path.abspath(__file__))


def _bf16_view(t):
    """A bfloat16 tensor's memory as a numpy array of ml_dtypes bfloat16."""
    return t.view(torch.int16).numpy().view(bfloat16)


class SuperkernelDecoder:
    """The decode's buffers and its dispatch.

    `max_L` is the longest sequence it serves; the KV cache is sized to it
    rounded up to a power of two (at least 128), so that a few builds serve
    every length.
    """

    def __init__(self, model=None, max_L=None, verbose=True):
        import airsrc

        # the variant's package, whose modules are E2B's at its dims
        variant = os.environ.get("GEMMA4_VARIANT", "e2b")
        airsrc.add_air_paths(f"gemma4_{variant}_q4nx", "gemma4_e2b_q4nx")
        import gemma4_e2b_q4nx_weights as gw

        from . import geom as G
        from . import pack as K
        from .wave import EPS, slab_offsets

        self.gw, self.K, self.G = gw, K, G
        self.maxL = self.ATTN_MAXL = max(128, 1 << ((max_L or 1) - 1).bit_length())
        t0 = time.time()
        m = self.m = gw.Q4nxModel(model)
        nl = gw.NUM_LAYERS
        self.arms = tuple(K.arm_of(L) for L in range(nl)) + (0,) * G.NL
        kvsrc = tuple(gw.kv_source_layer(L) for L in range(nl)) + (0,) * G.NL
        W = []
        for L in range(nl):
            W.append(K.pack_wave(K.phase_mats(m, L), self.arms[L], 4))
        for c in range(G.NL):
            W.append(K.pack_wave([K.lm_mats(m, c)], 0, 1))
        g = m.globals()
        norms = [m.layer_norms(L) for L in range(nl)]
        rms = [
            K.bf(
                np.concatenate(
                    [n["input"], n["post_attn"], n["pre_ffn"], n["post_ffn"]]
                )
            )
            for n in norms
        ]
        ple = [m.layer_ple(L) for L in range(nl)]
        wa, wb = [], []
        for p in ple:
            g_, u_ = K.bf(p["inp_gate"]), K.bf(p["per_layer_projection"])
            n = gw.D // 64  # each stream's half of the up projection's rows
            wa += K.ple_blocks(g_[:128]) + [
                u_[i * 32 : (i + 1) * 32, :].T.reshape(-1) for i in range(n)
            ]
            wb += K.ple_blocks(g_[128:]) + [
                u_[(i + n) * 32 : (i + n + 1) * 32, :].T.reshape(-1) for i in range(n)
            ]
        # a KV slab each layer that keeps a cache (they come first), then the
        # scratch slab the others append to
        own = [L for L in range(nl) if gw.owns_kv(L)]
        assert own == list(range(len(own)))
        kvapp = tuple(L if gw.owns_kv(L) else len(own) for L in range(nl))
        kvapp += (len(own),) * G.NL
        self.SL = self.maxL * G.KVW
        PX = K.bf_tensor(np.zeros(nl * G.PXW))
        px = _bf16_view(PX)
        for L, n in enumerate(norms):
            px[L * G.PXW + gw.PLI_D : (L + 1) * G.PXW] = K.bf(
                np.concatenate(
                    [g["ple_proj_norm"], n["post_ple"], np.full(32, n["out_scale"])]
                )
            ).astype(bfloat16)
        D = gw.D
        self.args = [
            K.bf_tensor(np.zeros(2 * D)),  # X
            K.bf_tensor(np.concatenate(rms + [np.zeros(D), K.bf(g["final_norm"])])),
            torch.from_numpy(np.concatenate(W)),  # W
            K.bf_tensor(np.zeros(nl * 1536)),  # ROPE
            K.bf_tensor(np.zeros((len(own) + 1) * 2 * self.SL)),  # KV
            K.bf_tensor(np.zeros(D)),  # E
            PX,
            K.bf_tensor(
                np.concatenate(
                    [b for p in ple for b in K.ple_blocks(K.bf(p["model_proj"]))]
                )
            ),  # WP
            K.bf_tensor(np.concatenate(wa)),  # WA
            K.bf_tensor(np.concatenate(wb)),  # WB
            torch.zeros((G.NL + 1) * G.LM_ROWS, dtype=torch.bfloat16),  # LOG
        ]
        self.kw = dict(
            ARMS=self.arms,
            SLABS=slab_offsets(self.arms),
            KVSRC=kvsrc,
            KVAPP=kvapp,
            NDEC=nl,
            NW=nl + G.NL,
            S=os.path.join(HERE, "proj_aie2p.mlir"),
            SM=os.path.join(HERE, "attn_aie2p.mlir"),
            EPS=EPS,
            MAXL=self.maxL,
            SWA=gw.SLIDING_WINDOW,
        )
        self.res = None
        if verbose:
            print(
                f"[superkernel] model packed in {time.time() - t0:.0f} s, "
                f"KV for {self.maxL} positions"
            )

    # buffer arguments: what a token writes (X E ROPE PX) and reads (LOG)
    X, KV, E, ROPE, PX, LOG = 0, 4, 5, 3, 6, 10

    def _buf(self, i):
        return self.args[i] if self.res is None else self.res.tensors[i]

    def seed_kv(self, ks, vs, P):
        """The prefill's K and V: per layer (P, kv heads x head_dim), as
        mlir-air's `FusedDecoder.seed_kv` takes them. Each head is in a 512
        slot of the row; a narrow one laid out as the decode's RoPE pairs it:
        K [lo | 0 | hi | 0], V [v | 0]."""
        gw, DH, SL, NKV = self.gw, self.K.DH, self.SL, self.gw.N_KV_HEADS
        if P >= self.maxL:
            raise ValueError(f"P={P} does not fit a KV cache of {self.maxL}")
        a = _bf16_view(self._buf(self.KV))
        a[:] = 0
        for L in range(gw.NUM_LAYERS):
            if not gw.owns_kv(L):
                continue
            dh = gw.head_dim(L)
            k = np.asarray(ks[L], np.float32).reshape(P, NKV, dh)
            v = np.asarray(vs[L], np.float32).reshape(P, NKV, dh)
            kr = np.zeros((P, NKV, DH), np.float32)
            kr[..., : dh // 2] = k[..., : dh // 2]
            kr[..., DH // 2 : DH // 2 + dh // 2] = k[..., dh // 2 :]
            vr = np.zeros((P, NKV, DH), np.float32)
            vr[..., :dh] = v
            o, n = L * 2 * SL, P * NKV * DH
            a[o : o + n] = self.K.bf(kr).reshape(-1).astype(bfloat16)
            a[o + SL : o + SL + n] = self.K.bf(vr).reshape(-1).astype(bfloat16)
        if self.res is not None:
            from triton.backends.amd_triton_npu.npu_runner import sync_to

            sync_to(self.res.bos[self.KV])

    def dispatch(self, tok, p):
        """Token `tok` at position `p`: appends its K/V at row p and returns
        the softcapped logits [VOCAB]."""
        gw, K, m = self.gw, self.K, self.m
        if not 0 <= p < self.maxL:
            raise ValueError(f"position {p} is outside [0, {self.maxL - 1}]")
        x0 = K.bf(
            m.embed_rows("model.embed_tokens.weight", [tok]).reshape(-1)
            * gw.EMBED_SCALE
        )
        embs = m.embed_rows("model.per_layer_token_embd.weight", [tok]).reshape(
            gw.NUM_LAYERS, gw.PLI_D
        )
        _bf16_view(self._buf(self.X))[: gw.D] = x0.astype(bfloat16)
        self._buf(self.E)[:] = K.bf_tensor(x0)
        rope = _bf16_view(self._buf(self.ROPE))
        px = _bf16_view(self._buf(self.PX))
        PXW = self.G.PXW
        for L in range(gw.NUM_LAYERS):
            rope[L * 1536 : (L + 1) * 1536] = K.rope_w(m, L, p).astype(bfloat16)
            px[L * PXW : L * PXW + gw.PLI_D] = K.bf(
                embs[L] * gw.PLE_EMBED_SCALE
            ).astype(bfloat16)
        if self.res is None:
            self._first(p)
        else:
            self.res.run(p, to=(self.X, self.E, self.ROPE, self.PX), back=(self.LOG,))
        logits = self._buf(self.LOG).to(torch.float32).numpy()[: gw.VOCAB]
        cap = gw.FINAL_LOGIT_SOFTCAP
        return cap * np.tanh(logits / cap) if cap else logits

    def _first(self, p):
        """Builds the dispatch (or takes Triton's cached build), runs the
        token, and keeps the buffers on the device from then on."""
        from triton.backends.amd_triton_npu.config import config_context
        from triton.backends.amd_triton_npu.driver_scope import driver_scope

        from .wave import wave

        with driver_scope("npu"), config_context(bf16_emulation=False):
            ck = wave[(1,)](*self.args, p, **self.kw)
            self.res = ck.run.launch.resident(*self.args, p)
        # the device's copies are the ones read from here on
        self.args = None

    def close(self):
        if self.res is not None:
            self.res.device.close()
            self.res = None
