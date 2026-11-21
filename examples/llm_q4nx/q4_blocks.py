# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Weights in the layout the decode superkernels stream: Q4 blocks of 32 rows
by 256 columns, each [scale | min | nibbles] in 2560 int16 (5120 bytes), and
the bf16 rounding the packers share."""

import numpy as np
import torch


def bf(a):
    """float32 values rounded to bfloat16 (to nearest even), as float32."""
    u = np.asarray(a, np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def bf_tensor(a):
    """`a` as a bfloat16 tensor."""
    return torch.from_numpy(bf(a)).to(torch.bfloat16)


def _rows(t, n):
    """(q, scale, min) padded with zero rows to n."""
    return tuple(
        np.concatenate([a, np.zeros((n - a.shape[0],) + a.shape[1:], a.dtype)])
        for a in t
    )


def _cat(*mats):
    return tuple(np.concatenate(p, 0) for p in zip(*mats))


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
