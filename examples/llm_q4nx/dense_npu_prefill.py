# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""A dense Q4NX model's prefill on the device `dense_triton_fused` builds,
as a Triton program: NPU kernels called on a `FusedProgram`'s buffers, the
math between them as host kernels (`host_kernels`, `dense_host_kernels`, on
triton-cpu).

The model is mlir-air's `DensePrefill`, step for step: its table entry, its
weight reader and the norm eps and RoPE it takes from the weights'
config.json (`dense.config`) are used as they are. A prompt runs in chunks of
M rows; each stage calls every chunk's device op before the host reads the
first one's output, so the host work of one chunk overlaps the device work of
the others.
"""

import numpy as np
from ml_dtypes import bfloat16

import dense_host_kernels as DH
import dense_triton_fused as T
import gemma4_attention as A
import gemma4_gemm as G
import host_kernels as H
from triton.backends.amd_triton_npu.fused_program import FusedProgram

HR = G.M // G.TM


def _air():
    """mlir-air's dense model table, weight reader and model config."""
    from dense_fused_prefill import _dense

    dense, models = _dense()
    from air_examples.llms.shared.fused_prefill import packing

    return dense, models, packing


class NpuPrefill:
    """`prefill(ids)` returns the last token's logits; `kv_view(L)` layer L's
    roped K and raw V [P, kv heads * dh] (float32 of bf16), the handoff
    mlir-air's `DensePrefill` gives."""

    EMBED = "model.embed_tokens.weight"
    FINAL_NORM = "model.norm.weight"

    def __init__(self, build_dir, desc, context=None):
        import json
        from pathlib import Path

        T.check(desc)
        man = json.loads((Path(build_dir) / "manifest.json").read_text())
        self.s = T.Shape(**man)
        if self.s != T.Shape.of(desc, self.s.max_len):
            raise ValueError(f"{build_dir} is built for {self.s}, not {desc}")
        self.desc, self.context = desc, context
        self.max_len = self.s.max_len
        self.fp = FusedProgram(build_dir, "fused_prefill")
        self.a, self.c, self.att = {}, {}, {}
        self.kv = []
        self.rope_cache = {}
        self.current_context_length = 0
        self.suspended = False

    def contexts(self):
        return [self.fp.device]

    def suspend(self):
        if not self.suspended:
            self.fp.device.close()
            self.suspended = True

    def resume(self):
        if self.suspended:
            self.fp.device.open()
            self.suspended = False

    # ---- weights

    def _site(self, site, k, n, packed, wq):
        kp = packed.shape[1] * G.k_pad(k) // k
        packed = np.pad(packed, ((0, 0), (0, kp - packed.shape[1]), (0, 0)))
        w = self.fp.empty(packed.shape, np.uint8)
        w.host(write=True)[...] = packed
        self.sites[site] = dict(k=k, n=n, npd=G.n_pad(n), wq=wq, w=w)

    def load_weights(self, model=None):
        dense, _, P = _air()
        e, s = self.desc, self.s
        path = dense.resolve(e, model)
        self.eps, self.inv_freq, self.rot, self.rope_scale = dense.config(
            e, path, self.context
        )
        b = self.bundle = P.open_weights(P.codec(e, path), path)
        self.sites, self.norms = {}, []
        dq, dk = s.heads * s.dh, s.kv_heads * s.dh
        for L in range(e.layers):
            # the reader's layer as mlir-air's DensePrefill reads it
            w = dense.DensePrefill.layer_tensors(self, b, L)
            qkv = [np.concatenate(a) for a in zip(*w["qkv"])]
            self._site(f"{L}.qkv", s.d, dq + 2 * dk, G.pack_q4(*qkv), 1)
            for name, rows, k in (
                ("o", s.d, dq),
                ("gate", s.inter, s.d),
                ("up", s.inter, s.d),
                ("down", s.d, s.inter),
            ):
                self._site(f"{L}.{name}", k, rows, G.pack_q4(*w[name]), 1)
            self.norms.append(w["norms"])
        self.final_norm = b.bf16(self.FINAL_NORM)
        self.lm = T.n_split(s.vocab)
        if s.tied:
            emb = b.bf16(self.EMBED)
        else:
            raw = b.q4nx("lm_head.weight", s.vocab, s.d)
        for i, (n0, n1) in enumerate(self.lm):
            if s.tied:
                self._site(f"lm.{i}", s.d, n1 - n0, G.pack_bf16(emb[n0:n1]), 0)
            else:
                q4 = G.pack_q4(*(a[n0:n1] for a in raw))
                self._site(f"lm.{i}", s.d, n1 - n0, q4, 1)
        # Per layer and column, its KV heads' records, max_blocks apart.
        g = T.group(s)
        rec = 2 * g.lkp * g.dh
        self.kvb = [
            [self.fp.zeros((g.kv_heads * g.max_blocks, rec), bfloat16) for _ in "01"]
            for _ in range(e.layers)
        ]

    # ---- device buffers, one set per chunk so the chunks can be in flight

    def am(self, k, c):
        """The A tiles of a K-deep GEMM of chunk c."""
        if (k, c) not in self.a:
            shape = (HR, G.k_pad(k) // G.TK, G.TM, G.TK)
            self.a[k, c] = self.fp.zeros(shape, bfloat16)
        return self.a[k, c]

    def out(self, role, npd, c):
        if (role, npd, c) not in self.c:
            self.c[role, npd, c] = self.fp.empty((G.M, npd), bfloat16)
        return self.c[role, npd, c]

    def gemm(self, site, c, act=0):
        """Call the GEMM of weights `site` on chunk c's A tiles; returns its
        output rows [M, npd] bf16."""
        st = self.sites[site]
        y = self.out(site.split(".")[-1], st["npd"], c)
        G.mm_g[(1,)](
            *T.gemm_args(self.s, self.am(st["k"], c), st["w"], y, st["npd"], st["k"], act, st["wq"])
        )  # fmt: skip
        return y

    def attn_bufs(self, c):
        """Each column's q and o of chunk c; its KV slot is filled per call."""
        if c not in self.att:
            hg = self.s.heads // 2
            bufs = []
            for _ in "01":
                bufs += [
                    self.fp.empty((hg, G.M, self.s.dh), bfloat16),
                    None,
                    self.fp.empty((hg, G.M, self.s.dh), bfloat16),
                ]
            self.att[c] = bufs
        return self.att[c]

    # ---- the model

    def _rope(self, op, r0):
        """cos and sin [M, rot / 2] of positions r0 .. r0 + M, as mlir-air's
        DensePrefill makes them: in float64, scaled, rounded to bf16."""
        if (op, r0) not in self.rope_cache:
            ang = np.arange(r0, r0 + G.M)[:, None] * self.inv_freq[op][None, :]
            self.rope_cache[(op, r0)] = tuple(
                np.ascontiguousarray(
                    (f(ang) * self.rope_scale).astype(bfloat16).astype(np.float32)
                )
                for f in (np.cos, np.sin)
            )
        return self.rope_cache[(op, r0)]

    def layer_op(self, L):
        _, models, _ = _air()
        return models.layer_op(self.desc, L)

    def prefill(self, ids):
        self.resume()
        ids = [int(t) for t in np.asarray(ids).reshape(-1)]
        n = len(ids)
        if n > self.max_len:
            raise ValueError(f"a {n}-token prompt exceeds max_len {self.max_len}")
        if any(t < 0 or t >= self.s.vocab for t in ids):
            raise ValueError(f"token ids outside [0, {self.s.vocab})")
        x = np.ascontiguousarray(self.bundle.rows(self.EMBED, ids), np.float32)
        chunks = [(r0, min(G.M, n - r0)) for r0 in range(0, n, G.M)]
        dk = self.s.kv_heads * self.s.dh
        self.kv = [
            (np.empty((n, dk), np.float32), np.empty((n, dk), np.float32))
            for _ in range(self.desc.layers)
        ]
        with self.fp:
            for L in range(self.desc.layers):
                self.layer(L, x, chunks)
            logits = self.head(x[n - 1 :])
        self.current_context_length = n
        return logits

    def head(self, x):
        """Logits [vocab] of the row x [1, d]: the final norm and the LM
        head."""
        H.rms_tile(x, self.final_norm, self.am(self.s.d, 0), 1, self.s.d, self.eps)
        ys = [self.gemm(f"lm.{i}", 0) for i in range(len(self.lm))]
        out = np.empty(self.s.vocab, np.float32)
        for (n0, n1), y in zip(self.lm, ys):
            out[n0:n1] = y.host()[0, : n1 - n0]
        return out

    def layer(self, L, x, chunks):
        s, nm, eps = self.s, self.norms[L], self.eps
        dq, dk = s.heads * s.dh, s.kv_heads * s.dh
        op = self.layer_op(L)
        rows = [x[r0 : r0 + t] for r0, t in chunks]
        qkv = []
        for c, (r0, t) in enumerate(chunks):
            H.rms_tile(rows[c], nm["input"], self.am(s.d, c), t, s.d, eps)
            qkv.append(self.gemm(f"{L}.qkv", c))
        att = []
        for c, (r0, t) in enumerate(chunks):
            rope = self._rope(op, r0)
            self.kv_write(L, r0, t, qkv[c], nm.get("k_norm"), rope)
            att.append(self.attention(L, op, c, r0, t, qkv[c], nm.get("q_norm"), rope))
        o = []
        for c, (r0, t) in enumerate(chunks):
            am, h0 = self.am(dq, c), 0
            for i in (0, 1):
                H.o_tile(att[c][3 * i + 2], am, t, h0)
                h0 += att[c][3 * i + 2].shape[0]
            o.append(self.gemm(f"{L}.o", c))
        gu = []
        post = nm.get("post_attn") if s.gemma else None
        pre = nm["pre_ffn"] if s.gemma else nm["post_attn"]
        for c, (r0, t) in enumerate(chunks):
            DH.add_rms(rows[c], o[c], post, t, eps, am=self.am(s.d, c), w2=pre)
            gu.append((self.gemm(f"{L}.gate", c, 1), self.gemm(f"{L}.up", c)))
        down = []
        for c, (r0, t) in enumerate(chunks):
            H.mul_tile(gu[c][0], gu[c][1], self.am(s.inter, c), t, s.inter)
            down.append(self.gemm(f"{L}.down", c))
        post = nm.get("post_ffn") if s.gemma else None
        for c, (r0, t) in enumerate(chunks):
            DH.add_rms(rows[c], down[c], post, t, eps)

    def kv_write(self, L, r0, t, qkv, k_norm, rope):
        """Layer L's KV heads of a chunk from the qkv GEMM's output rows, into
        its records and the kept cache."""
        s = self.s
        g = T.group(s)
        dq, dk = s.heads * s.dh, s.kv_heads * s.dh
        b0 = r0 // g.lkp
        kf, vf = (a[r0 : r0 + t] for a in self.kv[L])
        span = g.max_blocks * 2 * g.lkp * g.dh
        for col in (0, 1):
            h = col * g.kv_heads
            DH.kv_post(
                qkv, dq + h * s.dh, dq + dk + h * s.dh, k_norm, rope, kf, vf,
                self.kvb[L][col][b0:], span, g.kv_heads, t, g.lkp, A.dvt(s.dh),
                self.eps, h * s.dh, s.dh,
            )  # fmt: skip

    def attention(self, L, op, c, r0, t, qkv, q_norm, rope):
        """Chunk c's attention of layer L: its q from the qkv GEMM's output
        rows, normed, roped and scaled here. Returns its buffers."""
        s = self.s
        g = T.group(s)
        q0 = r0 // g.lkp
        nend = -(-(r0 + t) // g.lkp)
        k0 = max(0, q0 - s.window // g.lkp) if op == "s" else 0
        n = nend - k0
        bufs = self.attn_bufs(c)
        hg = s.heads // 2
        for i in (0, 1):
            DH.q_post(qkv, q_norm, rope, bufs[3 * i], t, i * hg, 1.0, self.eps)
            # the records the chunk attends to on, so writing later ones
            # does not wait for it
            bufs[3 * i + 1] = self.kvb[L][i][k0:]
        # the cores read K blocks in pairs; the one past the last is masked
        T.call_attention(s, op if s.window else "a", bufs, n + n % 2, q0, k0)
        return bufs

    def kv_view(self, L):
        return self.kv[L]

    def clear_context(self):
        self.current_context_length = 0
        self.kv = []
