# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""`SuperkernelDecoder`: the selected model's (`spec`) decode as one Triton
superkernel dispatch a token (`wave.py`), behind the interface of mlir-air's `FusedDecoder`, so
that mlir-air's `generate()` drives it unchanged.

The first dispatch compiles the superkernel (cached by Triton afterwards) and
runs the token; its buffers then stay on the device, and each later token
writes only the token's inputs and reads back the logits.
"""

import os
import time

import numpy as np
import torch
from ml_dtypes import bfloat16
from q4_blocks import bf, bf_tensor

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "gemma4_superkernel")


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
        from . import geom as G
        from . import pack as K
        from .wave import slab_offsets

        qw = G.W
        self.qw, self.G = qw, G
        self.maxL = self.ATTN_MAXL = max(128, 1 << ((max_L or 1) - 1).bit_length())
        D = qw.D
        t0 = time.time()
        S = G.S
        if S.get("source") == "requant":
            m = qw.open_weight_source(model or S["default_source"])
        else:
            m = qw.Q4nxModel(model)
        nl = qw.NUM_LAYERS
        # arm 1 a layer, 2 a sliding one, 0 an LM wave
        self.arms = tuple(
            2 if G.SWA and not qw.is_global_layer(L) else 1 for L in range(nl)
        )
        self.arms += (0,) * G.NL
        W = [K.pack_wave(K.phase_mats(m, L), 1) for L in range(nl)]
        lm = K.lm_q4(m, qw.VOCAB)
        W += [K.pack_wave([K.lm_mats(lm, c)], 0) for c in range(G.NL)]
        rms = []
        for L in range(nl):
            rms += list(m.layer_rms(L))
        # the final norm as an LM wave's rmsW's last D
        final = [np.zeros(D * (G.NORMS // 2 - 1), np.float32)]
        final += [m.bf16("model.norm.weight")]
        self.embed = np.asarray(m.bf16("model.embed_tokens.weight"), bfloat16)
        # what a layer's rope weights take besides the position
        rw = m.layer_qkv_bias if G.ROPE == "bias" else m.layer_qk_norm
        self.rw = [rw(L) for L in range(nl)]
        self.RS = self.maxL * G.GW  # a group's KV region
        # the weight buffers: one, or LPW layers' a buffer and the LM head's
        # in W4 (`wave`)
        if G.LPW:
            wbo = tuple(L // G.LPW for L in range(nl)) + (4,) * G.NL
            assert wbo[nl - 1] < 4
        else:
            wbo = (0,) * (nl + G.NL)
        ws = {}
        for b, w in zip(wbo, W):
            ws.setdefault(b, []).append(w)
        ws = {b: torch.from_numpy(np.concatenate(v)) for b, v in ws.items()}
        self.args = [
            bf_tensor(np.zeros(D + G.NL * G.LM_ROWS)),  # XL
            bf_tensor(np.concatenate(rms + final)),  # RMS
            ws.pop(0),  # W
            bf_tensor(np.zeros(nl * G.ROPE_W)),  # ROPE
            bf_tensor(np.zeros((nl + 1) * 4 * self.RS)),  # KV
        ]
        self.wx = {f"W{b}": t for b, t in sorted(ws.items())}
        self.kw = dict(
            ARMS=self.arms,
            SLABS=slab_offsets(self.arms, wbo),
            KVAPP=tuple(range(nl)) + (nl,) * G.NL,
            NDEC=nl,
            NW=nl + G.NL,
            S=os.path.join(SCRIPTS, "proj_aie2p.mlir"),
            SM=os.path.join(SCRIPTS, "attn_aie2p.mlir"),
            MAXL=self.maxL,
        )
        if self.wx:
            self.kw.update(NWB=1 + len(self.wx), WBO=wbo)
        self.res = None
        if verbose:
            print(
                f"[superkernel] model packed in {time.time() - t0:.0f} s, "
                f"KV for {self.maxL} positions"
            )

    # buffer arguments: what a token writes (XL ROPE) and reads (XL)
    XL, ROPE, KV = 0, 3, 4

    def _buf(self, i):
        return self.args[i] if self.res is None else self.res.tensors[i]

    def seed_kv(self, fk, fv, P):
        """The prefill's roped K and V, [layers, P, kv heads x head dim], as
        mlir-air's `FusedDecoder.seed_kv` takes them, into each layer's slab:
        each group's columns in its region."""
        G, RS = self.G, self.RS
        if P >= self.maxL:
            raise ValueError(f"P={P} does not fit a KV cache of {self.maxL}")
        a = _bf16_view(self._buf(self.KV))
        a[:] = 0
        for L in range(self.qw.NUM_LAYERS):
            for g in range(2):
                cols = slice(g * G.GW, (g + 1) * G.GW)
                for r, src in ((g, fk), (2 + g, fv)):
                    o = (L * 4 + r) * RS
                    rows = np.asarray(src[L, :P, cols], np.float32)
                    a[o : o + P * G.GW] = bf(rows).reshape(-1).astype(bfloat16)
        if self.res is not None:
            from triton.backends.amd_triton_npu.npu_runner import sync_to

            sync_to(self.res.bos[self.KV])

    def dispatch(self, tok, p):
        """Token `tok` at position `p`: appends its K/V at row p and returns
        the logits [VOCAB]."""
        qw, G = self.qw, self.G
        if not 0 <= p < self.maxL:
            raise ValueError(f"position {p} is outside [0, {self.maxL - 1}]")
        xl = _bf16_view(self._buf(self.XL))
        xl[: qw.D] = self.embed[tok]
        rope = _bf16_view(self._buf(self.ROPE))
        for L in range(qw.NUM_LAYERS):
            rope[L * G.ROPE_W : (L + 1) * G.ROPE_W] = qw.rope_w_layer(p, L, *self.rw[L])
        if self.res is None:
            self._first(p)
        else:
            self.res.run(p, to=(self.XL, self.ROPE), back=(self.XL,))
        x = self._buf(self.XL)
        return x[qw.D : qw.D + qw.VOCAB].to(torch.float32).numpy()

    def _first(self, p):
        """Builds the dispatch (or takes Triton's cached build), runs the
        token, and keeps the buffers on the device from then on."""
        from triton.backends.amd_triton_npu.config import config_context
        from triton.backends.amd_triton_npu.driver_scope import driver_scope

        from .wave import wave

        with driver_scope("npu"), config_context(bf16_emulation=False):
            ck = wave[(1,)](*self.args, p, **self.kw, **self.wx)
            self.res = ck.run.launch.resident(*self.args, p, *self.wx.values())
        # the device's copies are the ones read from here on
        self.args = self.wx = None

    def close(self):
        if self.res is not None:
            self.res.device.close()
            self.res = None
