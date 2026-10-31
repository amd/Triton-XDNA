# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4-E2B's prefill on the devices `gemma4_triton_fused` builds,
dispatched by `npu_runner` on weights it reads (`gemma4_weights`) and packs
itself.

A prompt runs in chunks of M rows. Each layer starts every chunk's GEMMs
before it waits on the first, so the host work of one chunk overlaps the
device work of the others. The math between the device's GEMMs and
attention runs on the host as Triton kernels (`host_kernels`, on
triton-cpu), each writing the layout the next device op reads.
"""

import numpy as np
from ml_dtypes import bfloat16

import config as C
import gemma4_gemm as G
import gemma4_lm_gemv as LG
import gemma4_triton_fused as T
import host_kernels as H
from gemma4_weights import Gemma4Weights
from triton.backends.amd_triton_npu.driver_scope import driver_scope
from triton.backends.amd_triton_npu.npu_runner import Device, sub_bo, sync_from, sync_to

HR = G.M // G.TM
#: per attention op, its groups (q's buffer slot, group) in kernel order
ATTN = {op: [(T.ATTN_SLOT[g], T.GROUPS[g]) for g in gs] for op, gs in T.OPS.items()}


def inv_freq(dh, sliding, freqs):
    """RoPE's inverse frequencies [dh / 2], f32."""
    half = dh // 2
    theta = C.ROPE_SLIDING_THETA if sliding else C.ROPE_GLOBAL_THETA
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / dh))
    if not sliding and freqs is not None:
        inv = inv / freqs[:half]
    return np.ascontiguousarray(inv, np.float32)


# ---- the prefill -----------------------------------------------------------


class NpuPrefill:
    """`build_dir` holds fused_prefill.xclbin / .txn.h and lm.xclbin /
    lm.insts.bin. `kv_store(L, r0, k, v)` takes each owning layer's K and V
    rows (float32 [t, dh], normed and roped)."""

    def __init__(self, build_dir, max_len=2048, kv_store=None):
        from pathlib import Path

        d = Path(build_dir)
        self.dir, self.max_len, self.kv_store = d, max_len, kv_store
        self.fp = Device(
            d / "fused_prefill.xclbin", T.NARGS, txn_h=d / "fused_prefill.txn.h"
        )
        self.lm_dev = Device(d / "lm.xclbin", 3, insts={"lm": d / "lm.insts.bin"})
        self.on_layer = None
        self.a, self.c, self.runs, self.att = {}, {}, {}, {}
        self.suspended = False

    def contexts(self):
        return [self.fp, self.lm_dev]

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
        bo = self.fp.bo(packed.nbytes, 1)
        bo.write(packed.tobytes(), 0)
        sync_to(bo)
        self.sites[site] = dict(k=k, n=n, npd=G.n_pad(n), wq=wq, w=bo)

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
        self.kv = {}
        for L in range(C.N_LAYERS):
            if not C.owns_kv(L):
                continue
            slot, g = ATTN["s" if C.is_sliding(L) else "f"][0]
            nblk = -(-self.max_len // g.lkp) + 1
            bo, m = self.fp.array(slot + 1, bfloat16, (nblk, 2 * g.lkp * g.dh))
            self.kv[L] = dict(bo=bo, m=m, subs={})
        lw = LG.pack(*w.lm_head(0, C.VOCAB))
        self.lm_w = self.lm_dev.bo(lw.nbytes, 0)
        self.lm_w.write(lw.tobytes(), 0)
        sync_to(self.lm_w)
        self.lm_x, self.lm_xm = self.lm_dev.array(1, bfloat16, (LG.K,))
        self.lm_y, self.lm_ym = self.lm_dev.array(2, bfloat16, (LG.V,))
        self.lm_run = self.lm_dev.launch(
            "lm", {0: self.lm_w, 1: self.lm_x, 2: self.lm_y}
        )
        self.logits = np.empty(LG.V, np.float32)

    # ---- GEMMs

    def a_buf(self, k, c):
        key = (k, c)
        if key not in self.a:
            self.a[key] = self.fp.array(
                0, bfloat16, (HR, G.k_pad(k) // G.TK, G.TM, G.TK)
            )
        return self.a[key]

    def am(self, k, c):
        """The A tiles a kernel writes for a K-deep GEMM of chunk c."""
        return self.a_buf(k, c)[1]

    def a_sync(self, k, c):
        sync_to(self.a_buf(k, c)[0])

    def gemm_start(self, site, c, act=0, role=None):
        s = self.sites[site]
        role = role or site.split(".")[-1]
        name = T.gemm_name(s["k"], s["n"], act, s["wq"])
        key = (name, role, c)
        if key not in self.runs:
            ck = (role, s["npd"], c)
            if ck not in self.c:
                self.c[ck] = self.fp.array(2, bfloat16, (G.M, s["npd"]))
            bos = {0: self.a_buf(s["k"], c)[0], 1: s["w"], 2: self.c[ck][0]}
            self.runs[key] = (self.fp.launch(name, bos), self.c[ck])
        run, (cb, cm) = self.runs[key]
        run.set_bo(1, s["w"])
        sync_to(cb, 64, 0)
        run.start()
        return run, cb, cm, s

    def gemm_wait(self, pend, t):
        """The output's rows [M, npd] bf16, valid in [:t, :n]."""
        run, cb, cm, s = pend
        run.wait()
        sync_from(cb, t * s["npd"] * 2, 0)
        return cm

    # ---- attention

    def attn_chunk(self, op, c):
        key = (op, c)
        if key not in self.att:
            groups, hg = ATTN[op], C.N_Q_HEADS // len(ATTN[op])
            bos, parts = {}, []
            for slot, g in groups:
                q, qm = self.fp.array(slot, bfloat16, (hg, G.M, g.dh))
                o, om = self.fp.array(slot + 2, bfloat16, (hg, G.M, g.dh))
                bos[slot], bos[slot + 2] = q, o
                parts.append(dict(q=q, qm=qm, o=o, om=om, slot=slot, hg=hg))
            run = self.fp.launch(T.attn_name(op), bos, rt=(1, 0, 0))
            self.att[key] = dict(run=run, parts=parts)
        return self.att[key]

    def attn_start(self, op, kv, c, r0, t, q, q_norm, rope):
        """q: the q GEMM's output rows, normed, roped and scaled here."""
        ch = self.attn_chunk(op, c)
        _, g = ATTN[op][0]
        scale = float(np.sqrt(g.dh) * C.ATTN_SCALE)
        q0 = r0 // g.lkp
        nend = -(-(r0 + t) // g.lkp)
        w = T.WINDOWS.get(op)
        k0 = max(0, q0 - w // g.lkp) if w else 0
        rec = 2 * g.lkp * g.dh * 2
        sub = kv["subs"].get(k0)
        if sub is None:
            sub = kv["subs"][k0] = sub_bo(kv["bo"], k0 * rec)
        h0 = 0
        for p in ch["parts"]:
            H.q_post(q, q_norm, rope, p["qm"], t, h0, scale, C.RMS_EPS)
            sync_to(p["q"])
            ch["run"].set_bo(p["slot"] + 1, sub)
            sync_to(p["o"], 64, 0)
            h0 += p["hg"]
        n = nend - k0
        ch["run"].set_rt(n + n % 2, q0, k0)
        ch["run"].start()
        return ch

    def attn_wait(self, ch, t, am):
        """The output into the A tiles am of [t, heads * dh]."""
        ch["run"].wait()
        h0 = 0
        for p in ch["parts"]:
            sync_from(p["o"])
            H.o_tile(p["om"], am, t, h0)
            h0 += p["hg"]

    def kv_write(self, L, r0, t, k, v, k_norm, rope):
        """Layer L's KV head from the k and v GEMMs' output rows."""
        _, g = ATTN["s" if C.is_sliding(L) else "f"][0]
        kv = self.kv[L]
        b0 = r0 // g.lkp
        kf = np.empty((t, g.dh), np.float32)
        vf = np.empty((t, g.dh), np.float32)
        H.kv_post(
            k, v, k_norm, rope, kf, vf, kv["m"][b0:], t,
            g.lkp, g.dh // T.V_CHUNKS, C.RMS_EPS,
        )  # fmt: skip
        nb = kv["m"].shape[1] * 2
        sync_to(kv["bo"], -(-t // g.lkp) * nb, b0 * nb)
        if self.kv_store:
            self.kv_store(L, r0, kf, vf)

    # ---- the model

    def prefill(self, ids):
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
        with driver_scope("cpu"):
            return self._prefill(ids, n)

    def _prefill(self, ids, n):
        eps = C.RMS_EPS
        x = np.empty((n, C.D), np.float32)
        H.embed(*self.embed_tbl, ids, x, C.EMBED_SCALE)
        tbl = np.empty((n, C.N_LAYERS * C.PLI_D), np.float32)
        H.embed(*self.ple_tbl, ids, tbl)
        chunks = [(r0, min(G.M, n - r0)) for r0 in range(0, n, G.M)]
        pend = []
        for c, (r0, t) in enumerate(chunks):
            H.rms_tile(x[r0:], None, self.am(C.D, c), t, C.D, eps, norm=False)
            self.a_sync(C.D, c)
            pend.append(self.gemm_start("mp", c))
        pli = np.empty((n, C.N_LAYERS, C.PLI_D), np.float32)
        for c, (r0, t) in enumerate(chunks):
            H.ple(
                self.gemm_wait(pend[c], t), self.ple_norm, tbl[r0:], pli[r0:], t,
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
        for L in range(C.N_LAYERS):
            self.layer(L, x, pli, chunks, ropes)
            if self.on_layer:
                self.on_layer(L, x, pli)
        H.rms_rows(x[-1:], self.final_norm, self.lm_xm, eps)
        sync_to(self.lm_x)
        sync_to(self.lm_y, 64, 0)
        self.lm_run.start()
        self.lm_run.wait()
        sync_from(self.lm_y)
        _, tf, nr, fg = self.lm_ym.reshape(LG.NC, -1, LG.NR, LG.FLUSH * LG.GN).shape
        H.softcap(self.lm_ym, self.logits, C.FINAL_LOGIT_SOFTCAP, nr, tf, fg)
        return self.logits.copy()

    def layer(self, L, x, pli, chunks, ropes):
        nm = self.norms[L]
        eps = C.RMS_EPS
        dh = C.head_dim(L)
        dq = C.N_Q_HEADS * dh
        inter = C.mlp_inter(L)
        own = C.owns_kv(L)
        op = "s" if C.is_sliding(L) else "f"
        rows = [x[r0 : r0 + t] for r0, t in chunks]
        pend = []
        for c, (r0, t) in enumerate(chunks):
            H.rms_tile(rows[c], nm["input"], self.am(C.D, c), t, C.D, eps)
            self.a_sync(C.D, c)
            ps = [self.gemm_start(f"{L}.q", c)]
            if own:
                ps += [self.gemm_start(f"{L}.k", c), self.gemm_start(f"{L}.v", c)]
            pend.append(ps)
        kv = self.kv[C.kv_source_layer(L)]
        att = []
        for c, (r0, t) in enumerate(chunks):
            q = self.gemm_wait(pend[c][0], t)
            if own:
                k = self.gemm_wait(pend[c][1], t)
                v = self.gemm_wait(pend[c][2], t)
                self.kv_write(L, r0, t, k, v, nm["k_norm"], ropes[dh][c])
            att.append(self.attn_start(op, kv, c, r0, t, q, nm["q_norm"], ropes[dh][c]))
        pend = []
        for c, (r0, t) in enumerate(chunks):
            self.attn_wait(att[c], t, self.am(dq, c))
            self.a_sync(dq, c)
            pend.append(self.gemm_start(f"{L}.o", c))
        gu = []
        for c, (r0, t) in enumerate(chunks):
            o = self.gemm_wait(pend[c], t)
            H.add_rms(
                rows[c], o, nm["post_attn"], t, eps,
                am=self.am(C.D, c), w2=nm["pre_ffn"],
            )  # fmt: skip
            self.a_sync(C.D, c)
            gu.append(
                (self.gemm_start(f"{L}.gate", c, 1), self.gemm_start(f"{L}.up", c))
            )
        pend = []
        for c, (r0, t) in enumerate(chunks):
            g = self.gemm_wait(gu[c][0], t)
            u = self.gemm_wait(gu[c][1], t)
            H.mul_tile(g, u, self.am(inter, c), t, inter)
            self.a_sync(inter, c)
            pend.append(self.gemm_start(f"{L}.down", c))
        pg = []
        for c, (r0, t) in enumerate(chunks):
            d = self.gemm_wait(pend[c], t)
            H.add_rms(rows[c], d, nm["post_ffn"], t, eps, am=self.am(C.D, c))
            self.a_sync(C.D, c)
            pg.append(self.gemm_start(f"{L}.ple_gate", c, 1))
        pend = []
        for c, (r0, t) in enumerate(chunks):
            g = self.gemm_wait(pg[c], t)
            H.mul_tile(g, pli[r0:, L], self.am(C.PLI_D, c), t, C.PLI_D)
            self.a_sync(C.PLI_D, c)
            pend.append(self.gemm_start(f"{L}.ple_proj", c))
        for c, (r0, t) in enumerate(chunks):
            p = self.gemm_wait(pend[c], t)
            H.add_rms(rows[c], p, nm["post_ple"], t, eps, scale=nm["out_scale"])
