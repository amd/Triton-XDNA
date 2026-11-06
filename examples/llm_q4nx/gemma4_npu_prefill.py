# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""A Gemma4 model's prefill on the devices `gemma4_triton_fused` builds, as a
Triton program: NPU kernels called on a `FusedProgram`'s buffers, the math
between them as host kernels (`host_kernels`, on triton-cpu), on weights read
from the bundle (`gemma4_weights`) and packed here.

A prompt runs in chunks of M rows. Each layer calls every chunk's GEMMs before
the host reads the first one's output, so the host work of one chunk overlaps
the device work of the others.
"""

import numpy as np
from ml_dtypes import bfloat16

import config as C
import gemma4_gemm as G
import gemma4_lm_gemv as LG
import gemma4_triton_fused as T
import host_kernels as H
from gemma4_weights import Gemma4Weights
from triton.backends.amd_triton_npu.fused_program import FusedProgram

HR = G.M // G.TM


def inv_freq(dh, sliding, freqs):
    """RoPE's inverse frequencies [dh / 2], f32."""
    half = dh // 2
    theta = C.ROPE_SLIDING_THETA if sliding else C.ROPE_GLOBAL_THETA
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / dh))
    if not sliding and freqs is not None:
        inv = inv / freqs[:half]
    return np.ascontiguousarray(inv, np.float32)


def op_of(L):
    """Layer L's attention op."""
    return "s" if C.is_sliding(L) else "f"


class NpuPrefill:
    """`build_dir` holds the `fused_prefill` and `lm` devices.
    `kv_store(L, r0, k, v)` takes each owning layer's K and V rows (float32
    [t, dh], normed and roped)."""

    def __init__(self, build_dir, max_len=2048, kv_store=None):
        self.max_len, self.kv_store = max_len, kv_store
        self.fp = FusedProgram(build_dir, "fused_prefill")
        self.lm = FusedProgram(build_dir, "lm")
        self.on_layer = None
        #: `tail(x, pli, n, L0)`: given, the layers from `L0` on -- those that
        #: write no cache -- and the head are left to it, for the last row
        #: alone: `x` [1, D] f32 entering layer `L0`, `pli` that row's per-layer
        #: inputs [N_LAYERS, PLI_D]. Returns the logits.
        self.tail = None
        self.a, self.c, self.att = {}, {}, {}
        self.suspended = False

    def contexts(self):
        return [self.fp.device, self.lm.device]

    def suspend(self):
        if not self.suspended:
            for d in self.contexts():
                d.close()
            self.suspended = True

    def resume(self):
        if self.suspended:
            for d in self.contexts():
                d.open()
            self.suspended = False

    # ---- weights

    def _site(self, site, k, n, packed, wq):
        kp = packed.shape[1] * G.k_pad(k) // k
        packed = np.pad(packed, ((0, 0), (0, kp - packed.shape[1]), (0, 0)))
        w = self.fp.empty(packed.shape, np.uint8)
        w.host(write=True)[...] = packed
        self.sites[site] = dict(k=k, npd=G.n_pad(n), wq=wq, w=w)

    def load_weights(self, model=None):
        w = self.w = Gemma4Weights(model)
        self.sites = {}
        for L in range(C.N_LAYERS):
            for name, (q, s, mn) in w.projections(L).items():
                self._site(
                    f"{L}.{name}", q.shape[1], q.shape[0], G.pack_q4(q, s, mn), 1
                )
            ple = w.ple(L)
            for name, m in (("ple_gate", ple["gate"]), ("ple_proj", ple["proj"])):
                self._site(f"{L}.{name}", m.shape[1], m.shape[0], G.pack_bf16(m), 0)
        mp = np.concatenate([w.ple(L)["model_proj"] for L in range(C.N_LAYERS)])
        self._site("mp", mp.shape[1], mp.shape[0], G.pack_bf16(mp), 0)
        self.norms = [w.norms(L) for L in range(C.N_LAYERS)]
        self.final_norm, self.ple_norm = w.final_norm(), w.ple_proj_norm()
        freqs = w.rope_freqs()
        self.inv = {
            C.DH_SLIDING: inv_freq(C.DH_SLIDING, True, freqs),
            C.DH_GLOBAL: inv_freq(C.DH_GLOBAL, False, freqs),
        }
        self.embed_tbl, self.ple_tbl = w.embed_table(), w.ple_embed_table()
        # Per owning layer, each KV head's records.
        self.kv = {}
        for L in range(C.N_LAYERS):
            if C.owns_kv(L):
                g = T.GROUPS[T.OPS[op_of(L)][0]]
                # the cores read K blocks in pairs, so one past the last
                nblk = -(-self.max_len // g.lkp) + 1
                self.kv[L] = [
                    self.fp.zeros((nblk, 2 * g.lkp * g.dh), bfloat16)
                    for _ in range(C.N_KV_HEADS)
                ]
        # The LM head device is LG.K deep; a wider model runs it on slices of
        # its K, the last zero-padded, and adds their outputs.
        self.lm_k = [(k0, min(k0 + LG.K, C.D)) for k0 in range(0, C.D, LG.K)]
        q, sc, mn = w.lm_head(0, C.VOCAB)
        self.lm_w, self.lm_x, self.lm_y = [], [], []
        for k0, k1 in self.lm_k:
            parts = [q, sc, mn] if C.D == LG.K else []
            for a, g in ((q, 1), (sc, 32), (mn, 32))[len(parts) :]:
                part = np.zeros((a.shape[0], LG.K // g), a.dtype)
                part[:, : (k1 - k0) // g] = a[:, k0 // g : k1 // g]
                parts.append(part)
            lw = LG.pack(*parts)
            del parts
            self.lm_w.append(self.lm.empty(lw.shape, np.uint8))
            self.lm_w[-1].host(write=True)[...] = lw
            self.lm_x.append(self.lm.zeros(LG.K, bfloat16))
            self.lm_y.append(self.lm.empty(LG.V, bfloat16))
        del q, sc, mn
        self.logits = np.empty(LG.V, np.float32)

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
        s = self.sites[site]
        y = self.out(site.split(".")[-1], s["npd"], c)
        G.mm_g[(1,)](
            *G.args(self.am(s["k"], c), s["w"], y, s["npd"], s["k"], act, s["wq"], G.k_pad(s["k"]))
        )  # fmt: skip
        return y

    def attn_bufs(self, op, c):
        """Per group, in `GROUPS` order: q and o for op's groups of chunk c,
        stand-ins for the other op's (which it does not touch); the KV slot
        is filled per call."""
        if (op, c) not in self.att:
            hg = C.N_Q_HEADS // len(T.OPS[op])
            bufs = []
            for name, g in T.GROUPS.items():
                if name in T.OPS[op]:
                    q = self.fp.empty((hg, G.M, g.dh), bfloat16)
                    o = self.fp.empty((hg, G.M, g.dh), bfloat16)
                else:
                    q = o = self.fp.empty((1, 2 * g.lkp * g.dh), bfloat16)
                bufs += [q, None, o]
            self.att[op, c] = bufs
        return self.att[op, c]

    def attention(self, op, kv, c, r0, t, q, q_norm, rope):
        """q: the q GEMM's output rows, normed, roped and scaled here.
        Returns the op's buffers."""
        g = T.GROUPS[T.OPS[op][0]]
        scale = float(np.sqrt(g.dh) * C.ATTN_SCALE)
        q0 = r0 // g.lkp
        nend = -(-(r0 + t) // g.lkp)
        w = T.WINDOWS.get(op)
        k0 = max(0, q0 - w // g.lkp) if w else 0
        n = nend - k0
        bufs = self.attn_bufs(op, c)
        h0 = 0
        for i, name in enumerate(T.GROUPS):
            if name in T.OPS[op]:
                H.q_post(q, q_norm, rope, bufs[3 * i], t, h0, scale, C.RMS_EPS)
                # The op's j-th group takes q heads h0.., KV head j's, or the
                # one KV head.
                j = T.OPS[op].index(name)
                h0 += bufs[3 * i].shape[0]
                # the records the chunk attends to, so writing later ones
                # does not wait for it
                bufs[3 * i + 1] = kv[min(j, len(kv) - 1)][k0 : k0 + n]
            else:
                bufs[3 * i + 1] = bufs[3 * i]
        # the cores read K blocks in pairs; the one past the last is masked
        T.call_attention(op, bufs, n + n % 2, q0, k0)
        return bufs

    def attn_out(self, op, bufs, t, am):
        """The op's output into the A tiles am of [t, heads * dh]."""
        h0 = 0
        for i, name in enumerate(T.GROUPS):
            if name in T.OPS[op]:
                H.o_tile(bufs[3 * i + 2], am, t, h0)
                h0 += bufs[3 * i + 2].shape[0]

    def kv_write(self, L, r0, t, k, v, k_norm, rope):
        """Layer L's KV heads from the k and v GEMMs' output rows; `kv_store`
        gets them as [t, heads * dh]."""
        g = T.GROUPS[T.OPS[op_of(L)][0]]
        b0, nb = r0 // g.lkp, -(-t // g.lkp)
        kf = np.empty((C.N_KV_HEADS, t, g.dh), np.float32)
        vf = np.empty((C.N_KV_HEADS, t, g.dh), np.float32)
        for h, rec in enumerate(self.kv[L]):
            H.kv_post(
                k, v, k_norm, rope, kf[h], vf[h], rec[b0 : b0 + nb], t,
                g.lkp, g.dh // T.V_CHUNKS, C.RMS_EPS, col0=h * g.dh,
            )  # fmt: skip
        if self.kv_store:
            self.kv_store(
                L,
                r0,
                kf.transpose(1, 0, 2).reshape(t, -1),
                vf.transpose(1, 0, 2).reshape(t, -1),
            )

    # ---- the model

    def prefill(self, ids, soft_embeds=None, soft_positions=None):
        """`soft_embeds` [k, D] (a vision tower's output) replace the token
        embeddings at `soft_positions`, where `ids` hold the pad token. The
        per-layer table is gathered with those ids; the per-layer projection
        reads the embeddings with the soft tokens in place."""
        self.resume()
        ids = np.asarray(ids).reshape(-1)
        n = len(ids)
        if n > self.max_len:
            raise ValueError(f"a {n}-token prompt exceeds max_len {self.max_len}")
        # the embedding gathers index the tables with these unchecked
        bad = ids[(ids < 0) | (ids >= C.VOCAB)]
        if bad.size:
            raise ValueError(f"token ids outside [0, {C.VOCAB}): {bad[:8].tolist()}")
        ids = np.ascontiguousarray(ids, np.int32)
        soft = None
        if soft_embeds is not None:
            pos = np.asarray(soft_positions, np.int64)
            if len(pos) != len(soft_embeds):
                raise ValueError(
                    f"{len(pos)} soft positions for {len(soft_embeds)} soft tokens"
                )
            soft = (pos, np.asarray(soft_embeds, np.float32))
        with self.fp:
            return self._prefill(ids, n, soft)

    def _prefill(self, ids, n, soft=None):
        eps = C.RMS_EPS
        x = np.empty((n, C.D), np.float32)
        H.embed(*self.embed_tbl, ids, x, C.EMBED_SCALE)
        if soft is not None:
            # Unscaled: the soft tokens are already in the scaled space.
            x[soft[0]] = soft[1]
        tbl = np.empty((n, C.N_LAYERS * C.PLI_D), np.float32)
        H.embed(*self.ple_tbl, ids, tbl)
        chunks = [(r0, min(G.M, n - r0)) for r0 in range(0, n, G.M)]
        proj = []
        for c, (r0, t) in enumerate(chunks):
            H.rms_tile(x[r0:], None, self.am(C.D, c), t, C.D, eps, norm=False)
            proj.append(self.gemm("mp", c))
        pli = np.empty((n, C.N_LAYERS, C.PLI_D), np.float32)
        for c, (r0, t) in enumerate(chunks):
            H.ple(
                proj[c], self.ple_norm, tbl[r0:], pli[r0:], t,
                C.PLE_MODEL_PROJ_SCALE, C.PLE_INPUT_SCALE, eps,
            )  # fmt: skip
        ropes = {}
        for dh, inv in self.inv.items():
            ropes[dh] = []
            for r0, _ in chunks:
                cs = np.empty((G.M, dh // 2), np.float32)
                sn = np.empty((G.M, dh // 2), np.float32)
                H.rope_table(inv, cs, sn, r0)
                ropes[dh].append((cs, sn))
        # From the first layer after which none writes the cache, only the
        # last position's row reaches anything -- the logits -- so those layers
        # run its chunk alone. It still attends every earlier key, and the NPU
        # GEMMs take whole chunks, so this is one chunk a layer instead of all
        # of them.
        shared_from = next(
            L
            for L in range(C.N_LAYERS + 1)
            if not any(map(C.owns_kv, range(L, C.N_LAYERS)))
        )
        if self.tail is not None:
            for L in range(shared_from):
                self.layer(L, x, pli, chunks, ropes)
                if self.on_layer:
                    self.on_layer(L, x, pli)
            return self.tail(x[-1:], pli[-1], n, shared_from)
        for L in range(C.N_LAYERS):
            if L < shared_from:
                self.layer(L, x, pli, chunks, ropes)
            else:
                self.layer(
                    L, x, pli, chunks[-1:], {dh: r[-1:] for dh, r in ropes.items()}
                )
            if self.on_layer:
                self.on_layer(L, x, pli)
        if len(self.lm_k) == 1:
            H.rms_rows(x[-1:], self.final_norm, self.lm_x[0], eps)
        else:
            xn = np.empty(C.D, bfloat16)
            H.rms_rows(x[-1:], self.final_norm, xn, eps)
            for (k0, k1), xb in zip(self.lm_k, self.lm_x):
                xb.host(write=True)[: k1 - k0] = xn[k0:k1]
        with self.lm:
            for wb, xb, yb in zip(self.lm_w, self.lm_x, self.lm_y):
                LG.gemv_q4[(1,)](*LG.args(wb, xb, yb))
        y = self.lm_y[0]
        if len(self.lm_y) > 1:
            y = sum(np.asarray(yb.host(), np.float32) for yb in self.lm_y)
        tf = LG.TILES // LG.FLUSH
        H.softcap(y, self.logits, C.FINAL_LOGIT_SOFTCAP, LG.NR, tf, LG.FLUSH * LG.GN)
        return self.logits.copy()

    def layer(self, L, x, pli, chunks, ropes):
        nm = self.norms[L]
        eps = C.RMS_EPS
        dh = C.head_dim(L)
        dq = C.N_Q_HEADS * dh
        inter = C.mlp_inter(L)
        own = C.owns_kv(L)
        op = op_of(L)
        rows = [x[r0 : r0 + t] for r0, t in chunks]
        qkv = []
        for c, (r0, t) in enumerate(chunks):
            H.rms_tile(rows[c], nm["input"], self.am(C.D, c), t, C.D, eps)
            sites = ("q", "k", "v") if own else ("q",)
            qkv.append([self.gemm(f"{L}.{s}", c) for s in sites])
        kv = self.kv[C.kv_source_layer(L)]
        att = []
        for c, (r0, t) in enumerate(chunks):
            if own:
                _, k, v = qkv[c]
                self.kv_write(L, r0, t, k, v, nm["k_norm"], ropes[dh][c])
            q = qkv[c][0]
            att.append(self.attention(op, kv, c, r0, t, q, nm["q_norm"], ropes[dh][c]))
        o = []
        for c, (r0, t) in enumerate(chunks):
            self.attn_out(op, att[c], t, self.am(dq, c))
            o.append(self.gemm(f"{L}.o", c))
        gu = []
        for c, (r0, t) in enumerate(chunks):
            H.add_rms(
                rows[c], o[c], nm["post_attn"], t, eps,
                am=self.am(C.D, c), w2=nm["pre_ffn"],
            )  # fmt: skip
            gu.append((self.gemm(f"{L}.gate", c, 1), self.gemm(f"{L}.up", c)))
        down = []
        for c, (r0, t) in enumerate(chunks):
            H.mul_tile(gu[c][0], gu[c][1], self.am(inter, c), t, inter)
            down.append(self.gemm(f"{L}.down", c))
        pg = []
        for c, (r0, t) in enumerate(chunks):
            H.add_rms(rows[c], down[c], nm["post_ffn"], t, eps, am=self.am(C.D, c))
            pg.append(self.gemm(f"{L}.ple_gate", c, 1))
        pp = []
        for c, (r0, t) in enumerate(chunks):
            H.mul_tile(pg[c], pli[r0:, L], self.am(C.PLI_D, c), t, C.PLI_D)
            pp.append(self.gemm(f"{L}.ple_proj", c))
        for c, (r0, t) in enumerate(chunks):
            H.add_rms(rows[c], pp[c], nm["post_ple"], t, eps, scale=nm["out_scale"])
