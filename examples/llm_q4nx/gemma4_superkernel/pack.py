# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The model's weights in the layouts the decode superkernel streams them in:
Q4 blocks of 32 rows by 256 columns ([scale | min | nibbles], 5120 bytes),
in each phase's round order, only the blocks a layer's arm has; the PLE
projections as bf16 blocks; the per-position RoPE and norm vector.
"""

import numpy as np

import gemma4_e2b_q4nx_weights as gw

from q4_blocks import _blocks, _cat, _rows, bf, pack_blocks

from .geom import DH, I2A, I2P, INTER, J2P, KV_REPL, LM_ROWS, LMR, MASKA, NBA, NQ


def arm_of(L):
    """air-FLM's arm_of_layer: 1 wide/shared/full, 3 narrow/own/full, 5
    wide/shared/sliding, 7 narrow/own/sliding."""
    return (
        (5 if gw.is_sliding(L) else 1)
        if not gw.owns_kv(L)
        else (7 if gw.is_sliding(L) else 3)
    )


def has_w(arm, p, i, j):
    k = p * 8 + arm
    return i < I2A.value[k] and j < NBA.value[k] and (j & MASKA.value[k]) == 0


def _ocols(dh):
    """For each x column of o-proj (the o-gather's order, per CU [half][head]
    [256]) the model's column, or -1 where a sliding head's half is empty."""
    out = np.full(NQ * DH, -1, np.int64)
    for cu in range(2):
        for b in range(2):
            for a in range(4):
                h = cu * 4 + a
                if dh == DH:
                    src = h * DH + b * 256 + np.arange(256)
                elif b == 0:
                    src = h * dh + np.arange(256)
                else:
                    continue
                k = cu * 2048 + b * 1024 + a * 256
                out[k : k + 256] = src
    return out


def _take_cols(t, cols, G=32):
    """(q, scale, min) [M, K] -> x-space columns `cols` (-1: zero)."""
    q, sc, mn = t
    z = cols < 0
    c = np.where(z, 0, cols)
    q2 = q[:, c]
    q2[:, z] = 0
    g = c.reshape(-1, G)[:, 0] // G  # each 32 x columns are one model group
    zg = z.reshape(-1, G)[:, 0]
    sc2, mn2 = sc[:, g], mn[:, g]
    sc2[:, zg] = 0
    mn2[:, zg] = 0
    return q2, sc2, mn2


def phase_mats(m, L):
    """The layer's 4 phase matrices as wave4's cores take them, x-space
    columns and build-max rows: (q uint8 [M, K], scale, min [M, K / 32])."""
    w = m.layer_q4nx(L)
    dh = gw.head_dim(L)
    if gw.owns_kv(L) and KV_REPL:
        rep = lambda t: tuple(np.concatenate([a, a]) for a in t)  # the 2 CUs' copies
        qkv = _cat(w["q"], rep(w["k"]), rep(w["v"]))
    elif gw.owns_kv(L):
        qkv = _cat(w["q"], w["k"], w["v"])
    else:
        qkv = w["q"]
    ph0 = _rows(qkv, I2P[0] * 512)
    ph1 = _take_cols(w["o"], _ocols(dh))
    inter = w["up"][0].shape[0]
    up, gate = w["up"], w["gate"]
    il = tuple(
        np.concatenate(
            [
                (a if h == 0 else b)[s * 512 : (s + 1) * 512]
                for s in range(inter // 512)
                for h in (0, 1)
            ]
        )
        for a, b in zip(up, gate)
    )
    ph2 = _rows(il, I2P[2] * 512)
    dn = w["down"]
    pad = INTER - inter
    ph3 = tuple(
        np.concatenate(
            [a, np.zeros((a.shape[0], pad // (1 if i == 0 else 32)), a.dtype)], 1
        )
        for i, a in enumerate(dn)
    )
    return [ph0, ph1, ph2, ph3]


def pack_wave(mats, arm, phases):
    """A wave's weight slab: [stream][phase][fill], the arm's blocks only."""
    # rounds of 512 rows: an LM wave's, a layer's phases' build max
    rounds = [LMR] if arm == 0 else list(I2P)
    blocks = [_blocks(mats[p], rounds[p] * 16, J2P[p] * 2) for p in range(phases)]
    chunks = []
    for x in range(4):
        for k in range(2):
            for p in range(phases):
                q, sc, mn = blocks[p]
                idx = [
                    (i * 16 + x * 4 + 2 * k + c, j)
                    for i in range(rounds[p])
                    for j in range(J2P[p] * 2)
                    for c in range(2)
                    if has_w(arm, p, i, j)
                ]
                rbi = np.array([a for a, _ in idx])
                cbi = np.array([b for _, b in idx])
                chunks.append(
                    pack_blocks(q[rbi, cbi], bf(sc[rbi, cbi]), bf(mn[rbi, cbi]))
                )
    return np.concatenate(chunks).reshape(-1)


def lm_mats(m, c):
    r0 = c * LM_ROWS
    r1 = min(r0 + LM_ROWS, gw.VOCAB)
    t = m.lm_head_q4nx(r0, r1)
    return _rows(t, LM_ROWS)


def ple_blocks(Wm):
    """A bf16 matrix [rows, cols] as wave4's PLE cores stream it: blocks of 32
    rows x 256 columns, each laid out [256][32]."""
    rb, cb = Wm.shape[0] // 32, Wm.shape[1] // 256
    return [
        Wm[i * 32 : (i + 1) * 32, j * 256 : (j + 1) * 256].T.reshape(-1)
        for i in range(rb)
        for j in range(cb)
    ]


def rope_w(m, L, pos):
    """[cos 256 | sin 256 | q_norm 512 | k_norm 512] (air-FLM's layer_rope_w
    and qk-norm placement, with the 1/sqrt(2) a padded head's norm needs)."""
    cos, sin, dh = gw.rope_lut(pos, L, rope_freqs=m.rope_freqs())
    c, s = np.ones(256, np.float32), np.zeros(256, np.float32)
    c[: dh // 2], s[: dh // 2] = cos, sin
    nm = m.layer_norms(L)
    fix = 1.0 if dh == DH else float(np.sqrt(dh / DH))
    p = np.concatenate([np.arange(dh // 2), DH // 2 + np.arange(dh // 2)])
    qn, kn = np.zeros(DH, np.float32), np.zeros(DH, np.float32)
    qn[p] = np.asarray(nm["q_norm"], np.float32) * fix
    kn[p] = np.asarray(nm["k_norm"], np.float32) * fix
    return bf(np.concatenate([c, s, qn, kn]))
