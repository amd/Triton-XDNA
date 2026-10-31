# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Reader for FastFlowLM's model.q4nx bundles (safetensors, Codec B).

An int4 matrix [rows, K] is stored block-major as int16 blocks of
ROW_BLOCK rows by COL_BLOCK columns, block (i, j) at index i * (K /
COL_BLOCK) + j. A block holds the bf16 scales of its N_GROUPS column groups
for each row, then their mins, then the nibbles: byte k of column c in row
group g holds rows g * PARALLEL + 2k (low nibble) and g * PARALLEL + 2k + 1
(high nibble). A weight is q * scale + min.
"""

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROW_BLOCK = 32
COL_BLOCK = 256
GROUP = 32
PARALLEL = 16
N_GROUPS = COL_BLOCK // GROUP
#: int16 words of one block: scales, mins, then the nibbles
BLOCK_I16 = 2 * N_GROUPS * ROW_BLOCK + ROW_BLOCK * COL_BLOCK // 4


def resolve(model, repo):
    """`model` as a local model.q4nx path: a file, a directory holding one,
    or an HF repo id; `repo` when `model` is None."""
    model = model or os.environ.get("Q4NX_MODEL_SOURCE") or repo
    if os.path.isfile(model):
        return Path(model)
    if os.path.isdir(model) and os.path.isfile(os.path.join(model, "model.q4nx")):
        return Path(model) / "model.q4nx"
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model, "model.q4nx"))


def unpack(i16, nbi, nbj):
    """Blocks [nbi * nbj, BLOCK_I16] int16 -> (q uint8 [nbi * ROW_BLOCK,
    nbj * COL_BLOCK], scale and min float32 [nbi * ROW_BLOCK, nbj *
    N_GROUPS]) in row-major order."""
    b = np.ascontiguousarray(i16).reshape(nbi, nbj, BLOCK_I16)
    per = N_GROUPS * ROW_BLOCK

    def groups(a):
        a = np.ascontiguousarray(a).view(bfloat16).astype(np.float32)
        a = a.reshape(nbi, nbj, N_GROUPS, ROW_BLOCK).transpose(0, 3, 1, 2)
        return np.ascontiguousarray(a).reshape(nbi * ROW_BLOCK, nbj * N_GROUPS)

    qb = np.ascontiguousarray(b[..., 2 * per :]).view(np.uint8)
    g = ROW_BLOCK // PARALLEL
    qb = qb.reshape(nbi, nbj, g, COL_BLOCK, PARALLEL // 2).transpose(0, 1, 2, 4, 3)
    q = np.stack((qb & 0xF, qb >> 4), axis=4)  # [i, j, g, k, low/high, col]
    q = q.reshape(nbi, nbj, ROW_BLOCK, COL_BLOCK).transpose(0, 2, 1, 3)
    q = np.ascontiguousarray(q).reshape(nbi * ROW_BLOCK, nbj * COL_BLOCK)
    return q, groups(b[..., :per]), groups(b[..., per : 2 * per])


class Bundle:
    """One model.q4nx file, read by tensor name."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.hdr = json.loads(f.read(n))
        self.off = 8 + n
        if "__metadata__" in self.hdr:
            raise ValueError(f"{self.path}: a Codec A bundle; this reads Codec B")

    def fingerprint(self):
        """A hash of the header: a re-exported bundle gets a new one."""
        text = json.dumps(self.hdr, sort_keys=True).encode()
        return hashlib.sha256(text).hexdigest()[:16]

    def has(self, name):
        return name in self.hdr

    def shape(self, name):
        return self.hdr[name]["shape"]

    def raw(self, name, dtype, count=-1, at=0):
        """`count` items of `dtype` from item `at` of tensor `name`; all of
        it by default."""
        s, e = self.hdr[name]["data_offsets"]
        size = np.dtype(dtype).itemsize
        if count < 0:
            count = (e - s) // size - at
        return np.fromfile(
            self.path, dtype=dtype, count=count, offset=self.off + s + at * size
        )

    def bf16(self, name):
        """A bf16 tensor as float32, in its declared shape."""
        a = self.raw(name, bfloat16).astype(np.float32)
        return a.reshape(self.shape(name))

    def int8_table(self, name):
        """An int8 table [V, W] and its group-32 f32 scales [V, W / 32]
        (tensor `name + ".scale"`), mapped from the file."""
        V, W = self.shape(name)
        if self.shape(name + ".scale") != [V, W // 32]:
            raise ValueError(f"{name}: no group-32 scale table")

        def mapped(n, dtype, shape):
            s = self.off + self.hdr[n]["data_offsets"][0]
            return np.memmap(self.path, dtype, "r", offset=s, shape=tuple(shape))

        return mapped(name, np.int8, (V, W)), mapped(
            name + ".scale", np.float32, (V, W // 32)
        )

    def int4(self, name, rows, K, r0=0):
        """Rows [r0, r0 + rows) of an int4 matrix of depth K as (q, scale,
        min). r0 and rows are whole row blocks, which are contiguous."""
        if r0 % ROW_BLOCK or rows % ROW_BLOCK or K % COL_BLOCK:
            raise ValueError(f"{name}: rows [{r0}, {r0 + rows}) of depth {K}")
        nbi, nbj = rows // ROW_BLOCK, K // COL_BLOCK
        have = self.shape(name)[0]
        if (r0 // ROW_BLOCK + nbi) * nbj > have or have % nbj:
            raise ValueError(f"{name}: {have} blocks do not hold [{rows}, {K}]")
        at = r0 // ROW_BLOCK * nbj * BLOCK_I16
        i16 = self.raw(name, np.int16, nbi * nbj * BLOCK_I16, at)
        return unpack(i16, nbi, nbj)

    def int4_rows(self, name, K):
        """The number of rows an int4 matrix of depth K holds."""
        return self.shape(name)[0] // (K // COL_BLOCK) * ROW_BLOCK
