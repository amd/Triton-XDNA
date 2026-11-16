# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The model's weights in the layouts the decode superkernel streams them in:
Q4 blocks of 32 rows by 256 columns ([scale | min | nibbles], 5120 bytes),
in each phase's round order, only the blocks a layer's arm has; the PLE
projections as bf16 blocks; the per-position RoPE and norm vector.
"""

import numpy as np
import torch

import gemma4_e2b_q4nx_weights as gw

from .wave import I2A, I2P, J2P, MASKA, NBA

DH = 512  # every head in a 512 slot: the full layers' head dim
NQ = gw.N_Q_HEADS
LM_ROWS = 13824  # vocab rows an LM-head wave computes
NL = (gw.VOCAB + LM_ROWS - 1) // LM_ROWS  # LM-head waves


def bf(a):
    """float32 values rounded to bfloat16 (to nearest even), as float32."""
    u = np.asarray(a, np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def bf_tensor(a):
    """`a` as a bfloat16 tensor."""
    return torch.from_numpy(bf(a)).to(torch.bfloat16)


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


def _rows(t, n):
    """(q, scale, min) padded with zero rows to n."""
    return tuple(
        np.concatenate([a, np.zeros((n - a.shape[0],) + a.shape[1:], a.dtype)])
        for a in t
    )


def _cat(*mats):
    return tuple(np.concatenate(p, 0) for p in zip(*mats))


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
    if gw.owns_kv(L):
        rep = lambda t: tuple(np.concatenate([a, a]) for a in t)  # the 2 CUs' copies
        qkv = _cat(w["q"], rep(w["k"]), rep(w["v"]))
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
    pad = 12288 - inter
    ph3 = tuple(
        np.concatenate(
            [a, np.zeros((a.shape[0], pad // (1 if i == 0 else 32)), a.dtype)], 1
        )
        for i, a in enumerate(dn)
    )
    return [ph0, ph1, ph2, ph3]


def _blocks(t, rb, cb):
    """(q [rb*32, cb*256], sc, mn) -> q [rb, cb, 32, 256], sc/mn [rb, cb, 8, 32]."""
    q, sc, mn = t
    q = q.reshape(rb, 32, cb, 256).transpose(0, 2, 1, 3)
    f = lambda a: a.reshape(rb, 32, cb, 8).transpose(0, 2, 3, 1)  # noqa: E731
    return q, f(sc), f(mn)


def pack_blocks(q, sc, mn):
    """q [n, 32, 256] of 0..15, sc/mn [n, 8, 32] -> [n, 2560] int16."""
    n = q.shape[0]
    s16 = lambda a: (bf(a).view(np.uint32) >> 16).astype(np.uint16)
    out = np.zeros((n, 2560), np.uint16)
    out[:, :256] = s16(sc).reshape(n, -1)
    out[:, 256:512] = s16(mn).reshape(n, -1)
    qq = q.reshape(n, 32, 8, 8, 4).astype(np.uint16)  # [n, row, g, j, s]
    words = (
        qq[..., 0] | qq[..., 1] << 4 | qq[..., 2] << 8 | qq[..., 3] << 12
    )  # [n, row, g, j]
    out[:, 512:] = words.transpose(0, 2, 3, 1).reshape(n, -1)
    return out.view(np.int16)


def pack_wave(mats, arm, phases):
    """A wave's weight slab: [stream][phase][fill], the arm's blocks only."""
    # rounds of 512 rows: an LM wave's 27, a layer's phases' build max
    rounds = [27] if arm == 0 else list(I2P)
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
