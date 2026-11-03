# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Reader for Gemma4's vision tower weights (`vision_weight.q4nx`).

The vision weights are **not in `model.q4nx`** -- that bundle has 709 keys and
none of them is a vision tensor. They ship as a separate 337 MB file in the same
repo, `vision_weight.q4nx`, and this module is the only thing here that reads
it.

Two things make it unlike the text bundle's reader:

* **Nothing is quantized.** Every tensor is BF16, projections included. There is
  no Codec A or Codec B here, so no dequantization step and no group scales.
* **Everything is tile-packed** for FLM's vision MM engine. A projection stored
  as `[12, 3, 16384]` is a `[768, 768]` matrix rearranged, not a 3-D tensor.

The packing is FLM's own, from `FLM_Q4NX_Converter/q4nx/model_converter.py:702`
(`_multi_modal_mm_weight_rearrange`), applied to an `[out, in]` = `[N, K]` HF
weight::

    rearrange(W, "(N mm_n) (K mm_k) -> N K (mm_n mm_k)", mm_n=64, mm_k=256)

`mm_n`/`mm_k` are `VISION_MM_TILE_N`/`VISION_MM_TILE_K` from the model's
`vision_config`. The converter pads both dims up to `max(mm_n, mm_k)` first, but
every vision dim here (768, 3072, 10240, and the language widths 1536 and 2560)
is already a multiple of 256, so
no padding is in the file and `unpack` does not try to strip any -- it asserts
the shapes work out exactly instead.

The rearrange is a pure reshape-and-transpose, so it inverts exactly; `unpack`
is bitwise lossless and `test_gemma4_vision_q4nx.py` checks that by re-packing
what it reads.

Returned projections are **`[K, out]`**, transposed out of the file's `[N, K]`.
That is what every forward in `llm_q4nx/` takes (`y = x @ W`), and matches the
note in `gemma4_e2b_q4nx/config.py::load_q4nx` -- doing the transpose once, in
the reader, keeps the tower reading like its text sibling.
"""

import json
import os
import struct
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

#: The repo the text bundle comes from, when the caller names none. Same repo,
#: different file. E2B and E4B ship the same tower; only the projection into
#: the language space differs in width.
MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Gemma4-E2B-IT-NPU2")

#: The file inside it.
VISION_FILE = "vision_weight.q4nx"

# ---------------------------------------------------------------- dims
# Mirrors config.json's `vision_config`. Restated rather than parsed because the
# shape assertions below need them before any model directory is resolved, and a
# silent disagreement with the checkpoint is exactly what those assertions are
# for.
HIDDEN = 768
N_LAYERS = 16
N_HEADS = 12
HEAD_DIM = 64
INTER = 3072
PATCH = 16
POSITION_EMBEDDING_SIZE = 10240
POOLING_KERNEL_SIZE = 3
ROPE_THETA = 100.0
RMS_EPS = 1e-6

#: The MM tile the converter packed against. Not a free parameter: a wrong pair
#: reshapes into a same-sized matrix of nonsense, which is why `unpack` checks
#: the resulting dims against what the caller expects rather than trusting them.
MM_TILE_N = 64
MM_TILE_K = 256

#: Per-projection clip ranges ship as scalars next to each weight. The key stems
#: differ from the weight names (`attn_out` vs `out_proj`, `ffn.gate` vs
#: `ffn.gate_proj`), so the mapping is written out rather than derived.
_CLIP_STEM = {
    "q": "vision_attn.q",
    "k": "vision_attn.k",
    "v": "vision_attn.v",
    "o": "attn_out",
    "gate": "ffn.gate",
    "up": "ffn.up",
    "down": "ffn.down",
}


def _proj_keys():
    """name in the returned layer dict -> (key in the file, expected (K, out))."""
    return {
        "q": ("vision_attn.q_proj.weight", (HIDDEN, N_HEADS * HEAD_DIM)),
        "k": ("vision_attn.k_proj.weight", (HIDDEN, N_HEADS * HEAD_DIM)),
        "v": ("vision_attn.v_proj.weight", (HIDDEN, N_HEADS * HEAD_DIM)),
        "o": ("vision_attn.out_proj.weight", (N_HEADS * HEAD_DIM, HIDDEN)),
        "gate": ("ffn.gate_proj.weight", (HIDDEN, INTER)),
        "up": ("ffn.up_proj.weight", (HIDDEN, INTER)),
        "down": ("ffn.down_proj.weight", (INTER, HIDDEN)),
    }


#: name in the returned layer dict -> key in the file. All `[HIDDEN]` but the
#: two per-head norms, which are `[HEAD_DIM]`.
_NORM_KEYS = {
    "norm1": ("norm1.weight", HIDDEN),
    "post_attn_norm": ("post_attn_norm.weight", HIDDEN),
    "norm2": ("norm2.weight", HIDDEN),
    "ffn_post_norm": ("ffn_post_norm.weight", HIDDEN),
    "q_norm": ("vision_attn.q_norm.weight", HEAD_DIM),
    "k_norm": ("vision_attn.k_norm.weight", HEAD_DIM),
}


def as_bf16(a):
    """A numpy array as ml_dtypes bfloat16, without a copy when it already is."""
    a = np.asarray(a)
    return a if a.dtype == bfloat16 else a.astype(bfloat16)


def to_torch(a, dtype=None):
    """numpy -> torch. bfloat16 is reinterpreted, not converted: torch cannot
    borrow an ml_dtypes buffer, but the two types are the same two bytes."""
    import torch

    a = np.ascontiguousarray(a)
    if a.dtype == bfloat16:
        t = torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16)
    else:
        t = torch.from_numpy(a)
    return t if dtype is None else t.to(dtype)


def resolve_vision_weights(model=None):
    """Resolve `model` to a local `vision_weight.q4nx` path.

    The same three cases `gemma4_e2b_q4nx_weights.resolve_q4nx_model` accepts --
    a direct file, a directory holding the file, or an HF repo id -- so a caller
    that passes nothing downloads the file rather than reading whatever happens
    to be in someone's home directory.
    """
    model = model or MODEL_DEFAULT
    if os.path.isfile(model):
        return model
    if os.path.isdir(model):
        p = os.path.join(model, VISION_FILE)
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(f"{model} holds no {VISION_FILE}")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model, VISION_FILE)


def unpack(t, K, out, mm_n=MM_TILE_N, mm_k=MM_TILE_K):
    """Tile-packed `[N/mm_n, K/mm_k, mm_n*mm_k]` -> plain `[K, out]`.

    The exact inverse of the converter's rearrange, then a transpose: the file
    holds `[N=out, K=in]` and every forward here wants `[in, out]`.
    """
    n_blk, k_blk, blk = t.shape
    if blk != mm_n * mm_k:
        raise ValueError(f"tile is {blk}, not {mm_n}*{mm_k}={mm_n * mm_k}")
    if (n_blk * mm_n, k_blk * mm_k) != (out, K):
        raise ValueError(
            f"packed {t.shape} unpacks to [{n_blk * mm_n}, {k_blk * mm_k}], "
            f"but this tensor should be [{out}, {K}]"
        )
    # (N mm_n) (K mm_k) -> N K (mm_n mm_k) reversed: split the tile back into
    # its two axes, then interleave them with the block axes.
    w = t.reshape(n_blk, k_blk, mm_n, mm_k)
    w = w.transpose(0, 2, 1, 3).reshape(out, K)  # [N, K], the HF layout
    return np.ascontiguousarray(w.T)  # [K, out]


def pack(w, mm_n=MM_TILE_N, mm_k=MM_TILE_K):
    """`[K, out]` -> the file's tile-packed layout. Inverse of `unpack`.

    Only the round-trip test needs this, but it lives next to `unpack` so the
    two cannot drift.
    """
    n, k = w.shape[1], w.shape[0]
    t = np.ascontiguousarray(w.T).reshape(n // mm_n, mm_n, k // mm_k, mm_k)
    return t.transpose(0, 2, 1, 3).reshape(n // mm_n, k // mm_k, mm_n * mm_k)


class VisionWeights:
    """Reader for `vision_weight.q4nx`.

    Opens the safetensors header once and reads tensors on demand; `load()`
    pulls the whole tower, which is 337 MB and comfortably fits in host memory.
    """

    def __init__(self, path=None):
        self.path = Path(resolve_vision_weights(path))
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self._hdr = json.loads(f.read(n))
            self._data_off = 8 + n
        self._hdr.pop("__metadata__", None)

    def fingerprint(self):
        """Short digest of the header, as the text configs do it.

        Of the header alone, never the payload: this is printed next to a
        result, and hashing 337 MB to label one would be out of proportion.
        """
        import hashlib

        return hashlib.sha256(
            json.dumps(self._hdr, sort_keys=True).encode()
        ).hexdigest()[:16]

    @property
    def out_hidden(self):
        """The language model's hidden size: the projection's output width.

        Read off the packed shape rather than restated, because it is the one
        dim that differs between models (1536 on E2B, 2560 on E4B).
        """
        n_blk = self._hdr["model.vision.embedding_projection.weight"]["shape"][0]
        return n_blk * MM_TILE_N

    def has(self, name):
        return name in self._hdr

    def raw(self, name):
        """The tensor as stored: BF16, in the header's own shape."""
        e = self._hdr[name]
        if e["dtype"] != "BF16":
            raise ValueError(f"{name} is {e['dtype']}, and this reader is BF16-only")
        s, t = e["data_offsets"]
        a = np.fromfile(
            self.path,
            dtype=np.int16,
            count=(t - s) // 2,
            offset=self._data_off + s,
        )
        return a.view(bfloat16).reshape(e["shape"])

    def f32(self, name):
        return self.raw(name).astype(np.float32)

    def scalar(self, name):
        """A `[1]` clip bound as a float.

        `+/-inf` survives the BF16 round trip, so a projection the exporter
        chose not to clip comes back as an unbounded clamp rather than as a
        special case anyone has to test for.
        """
        v = self.f32(name)
        if v.size != 1:
            raise ValueError(f"{name} is {v.shape}, expected a scalar")
        return float(v.reshape(-1)[0])

    def proj(self, name, K, out):
        """A tile-packed projection as `[K, out]` bfloat16."""
        return unpack(self.raw(name), K, out)

    def load(self):
        """The whole tower.

        Returns a dict with

            layers : list of 16 dicts, each with
                q, k, v, o, gate, up, down   : bfloat16 [K, out]
                norm1, post_attn_norm,
                norm2, ffn_post_norm         : float32 [768]
                q_norm, k_norm               : float32 [64]
                clip                         : {proj: (in_lo, in_hi, out_lo, out_hi)}
            patch_embd        : bfloat16 [768, 768]   (3*16^2 -> hidden)
            position_table    : bfloat16 [2, 10240, 768]
            embedding_proj    : bfloat16 [768, out_hidden]  (-> language hidden)
            fingerprint       : str

        The norms come back f32 and the projections bf16, matching what the text
        loader returns and what `_rms_norm`/`_matmul` each expect.
        """
        projs = _proj_keys()
        layers = []
        for L in range(N_LAYERS):
            p = f"model.vision.{L}."
            w = {
                nm: self.proj(p + key, K, out) for nm, (key, (K, out)) in projs.items()
            }
            for nm, (key, dim) in _NORM_KEYS.items():
                v = self.f32(p + key)
                if v.shape != (dim,):
                    raise ValueError(f"{p + key} is {v.shape}, expected ({dim},)")
                w[nm] = v
            w["clip"] = {
                nm: (
                    self.scalar(f"{p}{stem}_input_min"),
                    self.scalar(f"{p}{stem}_input_max"),
                    self.scalar(f"{p}{stem}_output_min"),
                    self.scalar(f"{p}{stem}_output_max"),
                )
                for nm, stem in _CLIP_STEM.items()
            }
            layers.append(w)

        # The position table is the one tensor stored per-dimension: [2, ...],
        # x first then y. The converter packs `weights[k].T`, i.e. it calls the
        # hidden size N and the table length K -- so `unpack`, which returns
        # [K, out], hands back [10240, 768], the HF layout, with no transpose
        # here. Passing (HIDDEN, POSITION_EMBEDDING_SIZE) instead would trip the
        # shape check rather than silently transpose.
        tbl = self.raw("model.vision.patch_embedder.position_embedding_table")
        if tbl.shape[0] != 2:
            raise ValueError(f"position table is {tbl.shape}, expected a leading 2")
        table = np.stack(
            [unpack(tbl[i], POSITION_EMBEDDING_SIZE, HIDDEN) for i in range(2)]
        )

        return dict(
            layers=layers,
            # 3 channels x 16x16 pixels = 768, which coincides with the hidden
            # size here. They are not the same number twice by design; the
            # assertion names both.
            patch_embd=self.proj(
                "model.vision.patch_embd.weight", 3 * PATCH * PATCH, HIDDEN
            ),
            position_table=np.ascontiguousarray(table),
            embedding_proj=self.proj(
                "model.vision.embedding_projection.weight", HIDDEN, self.out_hidden
            ),
            fingerprint=self.fingerprint(),
        )


def load_vision(model=None):
    """`VisionWeights(model).load()`, for callers that want one line."""
    return VisionWeights(model).load()
