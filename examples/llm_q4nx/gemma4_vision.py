# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4's vision tower: patches in, language-space soft tokens out.

This is the piece that lets the `gemma4_*_flash` examples take an image at
all. E2B and E4B share the tower; only the final projection's width differs,
and it is read off the weights. The text half of the model is already here
and gated (`examples/gemma4_*_q4nx/`); nothing below touches it.

The tower is a 16-layer ViT at hidden 768, 12 heads of 64, and it borrows the
Gemma block the text side already implements -- the four-norm sandwich, the
GELU-tanh GLU, the per-head q/k norms. Four things are its own, and each is
quiet when it is wrong:

* **Attention is not causal.** Patches see each other in both directions; the
  only mask is padding. `LlamaPrefill._attention` is causal by construction and
  `gpu_kernels.attn_prefill` takes a causal window, so `_attention` here is a
  separate implementation rather than a flag on that one.
* **RoPE is two-dimensional.** Lanes 0-31 of every head rotate on the patch's x
  coordinate and lanes 32-63 on its y, each with its own 16 frequencies at base
  100. Getting the split backwards costs nothing structural -- the shapes all
  still work -- and scrambles where the model thinks everything is.
* **Every projection is clipped on both sides.** `clamp(input) -> matmul ->
  clamp(output)`, with the bounds shipped per layer per projection in the
  weight file. This is model math (`Gemma4ClippableLinear`, and `vision_config`
  sets `use_clipped_linears: true`), not a quantization detail to drop on a
  bf16 path. Dropping it still produces plausible features.
* **v is normalized without a scale.** q and k get their learned per-head norm;
  v gets an RMSNorm with no weight at all. `_head_norm(..., weight=None)`
  already models exactly that shape.

The attention scale is **1.0**, not `head_dim**-0.5` -- the same surprise the
text side carries (`config.ATTN_SCALE`), and just as quiet: the softmax still
normalizes, it merely runs at the wrong temperature.

The reference is `transformers`' own `Gemma4VisionModel`
(`transformers/models/gemma4/modeling_gemma4.py`), which is what
`examples/gemma4_e2b_flash/vision.py` diffs this against stage by stage. The
deployed NPU implementation is FastFlowLM's
`FLM_DLL/detail/gemma4e_flash/gemma4e_image.cpp`; the two agree, and where this
file departs from FLM it is noted inline.

Placement: `_matmul`, `_rms_norm` and `_geglu` are the host object's, so they
go to the NPU under the same `--ops` flags the text prefill uses. Attention and
RoPE have no NPU kernel here and stay on the host or the iGPU, exactly as they
do for text.

**Do not run this with `--ops all`.** `rms_norm` is already outside
`Gemma4Prefill.DEFAULT_OPS` on the text side, for dispatch cost; here it also
does not complete. The per-head q/k/v norms fold the head axis into the row
axis, so at the default resolution `_head_norm` hands the kernel a `[2520 * 12,
64]` = `[30240, 64]` problem -- 30k rows of 64 lanes -- and the launch ends in
`ERT_CMD_STATE_TIMEOUT` and a TDR. The GEMMs at the same sequence length are
fine, so this is that kernel's shape, not the sequence length. The default op
set (`matmul`, `geglu`) is what this was gated at.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import gemma4_vision_q4nx as vw

HIDDEN = vw.HIDDEN
N_LAYERS = vw.N_LAYERS
N_HEADS = vw.N_HEADS
HEAD_DIM = vw.HEAD_DIM
INTER = vw.INTER
POOLING_K = vw.POOLING_KERNEL_SIZE
ROPE_THETA = vw.ROPE_THETA
RMS_EPS = vw.RMS_EPS

#: `Gemma4VisionAttention.scaling`. Not `HEAD_DIM**-0.5`; see the module note.
ATTN_SCALE = 1.0

#: The projections worth making NPU-resident, by the name they carry in a layer
#: dict. Every one is a GEMM over the full patch sequence; the norms are
#: kilobytes and stay where they are. Mirrors `LlamaPrefill.RESIDENT_ATTRS`.
#: What `_to_torch` leaves for the device, after fusing. `q|k|v` and
#: `gate|up` are concatenated rather than dispatched separately -- they read
#: the same activation and, crucially, the same *input* clip bound, which the
#: weight file happens to give them in all 16 layers (checked, not assumed).
#: Their output bounds differ, so those are applied per slice afterwards.
#:
#: Worth it because a wider GEMM is not merely one dispatch instead of three:
#: at the tower's K=768, three N=768 dispatches measured 3.63 ms against one
#: N=2304 at 2.20 ms (1.65x), and two N=3072 at 5.98 ms against one N=6144 at
#: 5.12 ms (1.17x). This mirrors what `LlamaPrefill.NPU_WEIGHTS` already does
#: on the text side, where the weights arrive concatenated to begin with.
RESIDENT_PROJS = ("qkv", "o", "gate_up", "down")

#: The fusions, as `(fused name, parts)`. `_attn_block` and the MLP split the
#: result back on these parts' widths.
_FUSIONS = (("qkv", ("q", "k", "v")), ("gate_up", ("gate", "up")))


def rope_tables(pos_ids, dtype=torch.float32):
    """Per-patch RoPE tables for the two spatial dimensions.

    `pos_ids` is `[P, 2]`, the (x, y) grid coordinate of each patch, with
    padding patches at `(-1, -1)`. Returns `(lut_x, lut_y)`, each `[P, 32]` in
    the layout `_rope` takes: 16 cosines then 16 sines.

    The frequencies are `Gemma4VisionRotaryEmbedding.compute_default_rope_parameters`:
    **the spatial dim, not the head dim**, sets the exponent denominator --
    `100 ** (-[0, 2, ..., 30] / 32)` -- so x and y get identical frequency
    ranges rather than two halves of one range. The HF source flags this as the
    thing that differs from the obvious reading, and it is worth flagging
    twice: the wrong denominator gives a table of the right shape.

    Negative coordinates are clamped to 0 the way HF clamps them. Padding
    patches are masked out of attention anyway, so what they rotate by does not
    reach a result; clamping only keeps the table finite.
    """
    spatial = HEAD_DIM // 2  # 32
    inv = 1.0 / (
        ROPE_THETA ** (torch.arange(0, spatial, 2, dtype=torch.float32) / spatial)
    )  # [16]
    pos = torch.as_tensor(np.asarray(pos_ids), dtype=torch.float32).clamp(min=0)
    luts = []
    for d in range(2):
        f = pos[:, d, None] * inv[None, :]  # [P, 16]
        luts.append(torch.cat([f.cos(), f.sin()], dim=-1).to(dtype))
    return luts[0], luts[1]


def pool_indices(pos_ids, valid, k=POOLING_K):
    """Which pooled slot each patch falls into, and how many slots there are.

    `Gemma4VisionPooler._avg_pool_by_positions`: the grid is divided into `k x k`
    tiles and every patch in a tile averages into one soft token. The slot is

        kernel_idx = (x // k) + (max_x // k) * (y // k)

    with `max_x` one past the largest x over the *valid* patches. Returns
    `(idx [P], n_slots)`.

    Padding patches clamp to (0, 0) and so land in slot 0. That is HF's
    behaviour, not an oversight to correct: their hidden states are zeroed
    before pooling, so they add nothing to the sum, and the divisor is a fixed
    `k*k` rather than a per-slot count. FLM computes the same thing as a
    scatter-add for the same reason -- the one-hot matrix HF writes is `[P,
    n_slots]` and almost entirely zero.
    """
    pos = np.asarray(pos_ids)
    clamped = np.maximum(pos, 0)
    max_x = int(clamped[np.asarray(valid), 0].max()) + 1
    kx = clamped[:, 0] // k
    ky = clamped[:, 1] // k
    n_slots = pos.shape[0] // (k * k)
    return kx + (max_x // k) * ky, n_slots


class Gemma4VisionTower:
    """The 16-layer vision encoder, plus the projection into language space.

    Constructed around a `host` -- a `Gemma4Prefill` -- whose `_matmul`,
    `_rms_norm`, `_geglu` and `_head_norm` it calls. Holding the driver rather
    than re-deriving it is what makes `--backend npu --ops ...` mean the same
    thing for both towers, and NPU residency is per driver instance in any
    case (`kernels.resident_weight`).
    """

    def __init__(self, host, weights=None, model=None):
        self.host = host
        self._w = self._to_torch(
            weights if weights is not None else vw.load_vision(model)
        )
        self.fingerprint = self._w.get("fingerprint")
        self._resident = False

    @staticmethod
    def _to_torch(w):
        """numpy out of the reader -> the tensors the operators take.

        Projections stay bf16 and are reinterpreted rather than converted (`_t`
        views the same two bytes); norms and clip bounds go to f32, which is
        what `_rms_norm` and `clamp` want. Done once here because `_matmul`
        distinguishes a torch tensor from an NPU-resident handle by asking
        whether it has `.to`, and a numpy array answers no -- so a raw array
        reaches the torch path as a confusing "this weight is resident" error.
        """
        bf = torch.bfloat16
        out = dict(w)
        out["patch_embd"] = vw.to_torch(vw.as_bf16(w["patch_embd"]), bf)
        out["embedding_proj"] = vw.to_torch(vw.as_bf16(w["embedding_proj"]), bf)
        # Gathered by row, so it stays numpy: `_patch_embed` indexes it with
        # the patch positions and converts only the rows it takes.
        out["position_table"] = np.asarray(w["position_table"])

        def _projs(lw):
            """The device-side projections for one layer, fused.

            Concatenating on the output axis is exactly what a separate
            dispatch per part would have computed -- same `x`, same shared
            input clamp, and the parts land in disjoint column ranges.
            """
            t = {
                nm: vw.as_bf16(np.asarray(lw[nm]))
                for nm in ("q", "k", "v", "o", "gate", "up", "down")
            }
            for fused, parts in _FUSIONS:
                lo = {tuple(lw["clip"][n][:2]) for n in parts}
                if len(lo) != 1:
                    raise ValueError(
                        f"{fused} parts disagree on their input clip {lo}; "
                        "fusing would apply one part's bound to another's"
                    )
                t[fused] = np.concatenate([t.pop(n) for n in parts], axis=1)
            return {nm: vw.to_torch(t[nm], bf) for nm in RESIDENT_PROJS}

        out["layers"] = [
            {
                **_projs(lw),
                **{
                    nm: vw.to_torch(np.asarray(lw[nm], dtype=np.float32))
                    for nm in (
                        "norm1",
                        "post_attn_norm",
                        "norm2",
                        "ffn_post_norm",
                        "q_norm",
                        "k_norm",
                    )
                },
                "clip": lw["clip"],
            }
            for lw in w["layers"]
        ]
        return out

    # ---- placement, delegated ----

    @property
    def timer(self):
        return self.host.timer

    def _matmul(self, x, w, **mm):
        return self.host._matmul(x, w, **mm)

    def _rms_norm(self, x, weight):
        return self.host._rms_norm(x, weight, RMS_EPS)

    def _geglu(self, gate, up):
        # The patch count is fixed by the image tier, so one dispatch per layer.
        return self.host._geglu(gate, up, whole=True)

    def _head_norm(self, x, weight):
        return self.host._head_norm(x, weight, N_HEADS, HEAD_DIM)

    def _rope(self, x, lut):
        return self.host._rope(x, lut, N_HEADS)

    # ---- operators of its own ----

    def _project(self, x, w, epilogue, **mm):
        """`epilogue(x @ w)`, with the epilogue applied where the NPU result
        lies rather than to a copy of it (`kernels.triton_matmul`)."""
        if self.host._device("matmul") == "npu":
            return self._matmul(x, w, epilogue=epilogue, **mm)
        return epilogue(self._matmul(x, w, **mm))

    def _clipped(self, x, w, clip, **mm):
        """`Gemma4ClippableLinear`: clamp in, project, clamp out.

        `clip` is the `(in_lo, in_hi, out_lo, out_hi)` tuple the weight file
        ships for this projection. Unclipped projections would carry
        `+/-inf` bounds and clamp to themselves, so there is no unclipped case
        to branch on.
        """
        lo_i, hi_i, lo_o, hi_o = clip
        return self._project(
            x.clamp(lo_i, hi_i), w, lambda y: y.clamp(lo_o, hi_o), **mm
        )

    def _clipped_split(self, x, w, clip, parts, widths, **mm):
        """One clipped projection standing in for several.

        The input bound is shared -- `_to_torch` refuses to fuse parts that
        disagree on it -- so the clamp happens once, before the single GEMM.
        The output bounds are not shared, so each slice gets its own.
        """
        lo_i, hi_i = clip[parts[0]][:2]

        def split(y):
            out, at = [], 0
            for name, width in zip(parts, widths):
                lo_o, hi_o = clip[name][2:]
                out.append(y[:, at : at + width].clamp(lo_o, hi_o))
                at += width
            return out

        return self._project(x.clamp(lo_i, hi_i), w, split, **mm)

    def _attention(self, q, k, v, valid):
        """Bidirectional MHA over the patches. q/k/v: [P, N_HEADS*HEAD_DIM].

        Not `LlamaPrefill._attention`: that one is causal by construction, and
        a ViT that only let a patch see the patches above and to its left would
        still produce a fluent caption.

        `valid` is a bool `[P]`: True for a real patch, False for the padding
        the image processor added to reach `max_patches`. Masked on the KEY axis
        only. A padded query row is left to attend whatever it likes -- its
        output is dropped by the pooler -- where masking both axes would leave
        those rows with an all `-inf` softmax and NaNs that then spread through
        the residual stream.

        At the default resolution P is 2520, so the score tensor is
        `[12, 2520, 2520]` -- about 290 MiB in f32, per layer. That is why this
        takes the iGPU when there is one.
        """
        with self.timer.track("vision_attention"):
            P = q.shape[0]
            npu = self._npu_attention(P, valid)
            if npu is not None:
                return npu(q, k, v, ATTN_SCALE)
            mask = torch.where(
                torch.as_tensor(np.asarray(valid), dtype=torch.bool),
                0.0,
                float("-inf"),
            )
            dev = "cuda" if self.host._device("attention") == "gpu" else None
            qh = q.reshape(P, N_HEADS, HEAD_DIM).transpose(0, 1)  # [H, P, dh]
            kh = k.reshape(P, N_HEADS, HEAD_DIM).transpose(0, 1)
            vh = v.reshape(P, N_HEADS, HEAD_DIM).transpose(0, 1)
            if dev is not None:
                qh, kh, vh, mask = (t.to(dev) for t in (qh, kh, vh, mask))
            # `scaled_dot_product_attention` rather than the explicit softmax
            # the text path writes: there the score matrix is small and the
            # explicit form documents the mask, here it is the dominant
            # allocation and SDPA's fused kernels avoid materializing it.
            # [1, 1, P] broadcasts over head and query for 3-D q/k/v.
            # No mask where every key is real, which is how `encode` calls it.
            out = torch.nn.functional.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                attn_mask=None if bool(np.all(valid)) else mask.view(1, 1, P),
                scale=ATTN_SCALE,
            )
            return out.transpose(0, 1).reshape(P, N_HEADS * HEAD_DIM).cpu().float()

    def _npu_attention(self, P, valid):
        """The NPU attention for `P` rows, or None where it does not apply.

        Taken when attention is placed on the NPU and no iGPU is in play, for
        an all-real batch of patches (which is how `encode` calls it) of a
        count the kernel supports.
        """
        host = self.host
        if host._device("attention") != "npu" or not bool(np.all(valid)):
            return None
        from vision_attn_npu import NPUVisionAttention

        if not NPUVisionAttention.supports(P, HEAD_DIM):
            return None
        cache = self.__dict__.setdefault("_npu_attn", {})
        if P not in cache:
            cache[P] = NPUVisionAttention(P, N_HEADS, HEAD_DIM)
        return cache[P]

    def _patch_embed(self, pixels, pos_ids, valid):
        """Pixels and positions -> `[P, 768]`, before the first layer.

        `Gemma4VisionPatchEmbedder`: rescale to [-1, 1], project the flattened
        `3*16*16 = 768` patch, and add a position embedding.

        The position embedding is written upstream -- and on the device -- as
        `one_hot(x, 10240) @ table[0] + one_hot(y, 10240) @ table[1]`, which is
        two `[P, 10240] x [10240, 768]` GEMMs. It **is** a row gather, and this
        does the gather: 10240 columns of which one is set is the definition of
        indexing. FLM materializes the one-hot buffer because its MM engine is
        the only multiplier it has there, which is not a constraint here.

        Padding rows are zeroed rather than left to gather row 0, matching
        `torch.where(padding_positions, 0.0, ...)` upstream.
        """
        with self.timer.track("patch_embed"):
            px = torch.as_tensor(np.asarray(pixels), dtype=torch.float32)
            h = self._matmul(2.0 * (px - 0.5), self._w["patch_embd"])
            tbl = self._w["position_table"]
            pos = np.maximum(np.asarray(pos_ids), 0)
            pe = vw.to_torch(tbl[0][pos[:, 0]], torch.float32) + vw.to_torch(
                tbl[1][pos[:, 1]], torch.float32
            )
            keep = torch.as_tensor(np.asarray(valid), dtype=torch.bool)[:, None]
            return h + torch.where(keep, pe, 0.0)

    def _layer(self, h, L, lut_x, lut_y, valid):
        """One encoder layer. h: [P, 768] -> [P, 768].

        The Gemma sandwich, identical in shape to the text side's:

            residual; norm1 -> attn -> post_attn_norm -> +residual
            residual; norm2 -> mlp  -> ffn_post_norm  -> +residual
        """
        w = self._w["layers"][L]
        c = w["clip"]

        residual = h
        x = self._rms_norm(h, w["norm1"])
        x = self._attn_block(x, L, w, c, lut_x, lut_y, valid)
        x = self._rms_norm(x, w["post_attn_norm"])
        h = residual + x

        residual = h
        x = self._rms_norm(h, w["norm2"])
        gate, up = self._clipped_split(
            x,
            w["gate_up"],
            c,
            ("gate", "up"),
            (INTER, INTER),
            stage_key=f"vision_gate_up_L{L}",
        )
        x = self._clipped(
            self._geglu(gate, up),
            w["down"],
            c["down"],
            stage_key=f"vision_down_L{L}",
        )
        x = self._rms_norm(x, w["ffn_post_norm"])
        return residual + x

    def _attn_block(self, x, L, w, c, lut_x, lut_y, valid):
        """q/k/v -> norms -> 2-D RoPE -> bidirectional attention -> o."""
        qp, kp, vp = self._clipped_split(
            x,
            w["qkv"],
            c,
            ("q", "k", "v"),
            (N_HEADS * HEAD_DIM,) * 3,
            stage_key=f"vision_qkv_L{L}",
        )
        q = self._head_norm(qp, w["q_norm"])
        k = self._head_norm(kp, w["k_norm"])
        # No weight: the reference normalizes v with no learned scale at all.
        v = self._head_norm(vp, None)
        q = self._rope2d(q, lut_x, lut_y)
        k = self._rope2d(k, lut_x, lut_y)
        a = self._attention(q, k, v, valid)
        return self._clipped(a, w["o"], c["o"], stage_key=f"vision_o_L{L}")

    def _rope2d(self, x, lut_x, lut_y):
        """Rotate lanes 0-31 of each head on x and lanes 32-63 on y.

        `apply_multidimensional_rope` splits the 64-lane head into `ndim = 2`
        pieces of 32 and hands each to the ordinary `apply_rotary_pos_emb`,
        which is precisely `LlamaPrefill._rope` at `DH = 32` -- pairs `(i,
        i+16)` within the piece, not `(i, i+32)` across the head. Reusing that
        operator rather than writing the rotation again is also what puts this
        on the iGPU under `hetero` for free.
        """
        P = x.shape[0]
        v = x.reshape(P, N_HEADS, HEAD_DIM)
        half = HEAD_DIM // 2
        a = self._rope(v[..., :half].reshape(P, N_HEADS * half), lut_x)
        b = self._rope(v[..., half:].reshape(P, N_HEADS * half), lut_y)
        return torch.cat(
            [a.reshape(P, N_HEADS, half), b.reshape(P, N_HEADS, half)], dim=-1
        ).reshape(P, N_HEADS * HEAD_DIM)

    def _pool(self, h, pos_ids, valid):
        """Average `k x k` patch tiles into soft tokens, then scale.

        Returns `(pooled [n_slots, 768], slot_valid [n_slots])`. The divisor is
        a fixed `k*k` even where a tile is partly padding, because the padded
        rows are zeroed first -- HF's `weights / k_squared`, not a per-slot
        count.

        The `sqrt(768)` at the end is `Gemma4VisionPooler.root_hidden_size`.
        FLM's comment calls it overflow protection; it is in the reference
        either way, and the RMSNorm that follows is scale-invariant, so leaving
        it out would be invisible until the projection that is not.
        """
        with self.timer.track("vision_pool"):
            idx, n_slots = pool_indices(pos_ids, valid)
            keep = torch.as_tensor(np.asarray(valid), dtype=torch.bool)
            h = torch.where(keep[:, None], h, 0.0)
            ti = torch.as_tensor(np.asarray(idx), dtype=torch.long)
            out = torch.zeros((n_slots, HIDDEN), dtype=torch.float32)
            out.index_add_(0, ti, h.to(torch.float32))
            out /= POOLING_K * POOLING_K
            # Which slots any patch reached at all. Padding lands in slot 0, so
            # slot 0 is always live -- the same as upstream, whose mask is
            # taken over the one-hot weights and not over validity.
            hit = torch.zeros(n_slots, dtype=torch.bool)
            hit[ti] = True
            return out * math.sqrt(HIDDEN), hit

    # ---- entry points ----

    def make_npu_resident(self):
        """Pad and stage every projection on the device, once.

        The same trade the text prefill makes in `LlamaPrefill.make_npu_resident`:
        the unpadded host copy is freed, so the torch path cannot run afterwards
        and `_matmul` says so by name if it is asked to.
        """
        if self._resident:
            return
        import kernels

        for w in self._w["layers"]:
            for nm in RESIDENT_PROJS:
                # Padded to the K the chain contracts (`_matmul` passes a
                # `stage_key` for every one of these).
                w[nm] = kernels.resident_weight(w[nm], exact=True)
        self._resident = True

    def encode(self, pixels, pos_ids, trace=None):
        """Patches in, language-space soft tokens out.

        `pixels` is `[P, 768]` (3 channels x 16 x 16, flattened) and `pos_ids`
        is `[P, 2]` with padding patches at `(-1, -1)` -- exactly what
        `Gemma4ImageProcessor` returns. Returns `[n_soft_tokens, out_hidden]`, ready
        to be scattered over the image placeholders in the prompt.

        `trace`, if given, is a dict this fills with the intermediate of every
        stage, keyed the way `gemma4_e2b_flash/vision.py` expects. That gate is
        the reason the tower is written to hand its stages out rather than to
        report a single end-to-end number: a wrong RoPE split and a wrong
        de-tile both show up as a bad caption, and only a per-stage diff says
        which.
        """
        pos = np.asarray(pos_ids)
        valid = ~(pos == -1).all(axis=-1)
        # The layers run on the real patches only. A padding patch reaches
        # nothing: attention masks it out as a key, every other operator is
        # row-wise, and the pooler zeroes it. So dropping its rows here gives
        # the real rows the same values, for less work. `full` puts them back
        # where the pooler and a trace expect them, with zeros in the padding.
        keep = np.flatnonzero(valid)
        lut_x, lut_y = (t[keep] for t in rope_tables(pos))
        ones = np.ones(len(keep), dtype=bool)

        def full(x):
            out = torch.zeros((len(valid), x.shape[1]), dtype=x.dtype)
            out[keep] = x
            return out

        h = self._patch_embed(np.asarray(pixels)[keep], pos[keep], ones)
        if trace is not None:
            trace["patch_embed"] = full(h)

        for L in range(N_LAYERS):
            h = self._layer(h, L, lut_x, lut_y, ones)
            if trace is not None:
                trace[f"layer{L}"] = full(h)

        pooled, hit = self._pool(full(h), pos, valid)
        if trace is not None:
            trace["pool"] = pooled.clone()

        # Slots no patch reached are dropped here, before the projection rather
        # than after, so the row count matches the number of image placeholders
        # in the prompt. Both are row-wise, so the order does not change a
        # value -- it is upstream's order (`Gemma4VisionModel.forward` strips,
        # then `get_image_features` embeds) and keeping it lets the two traces
        # be compared row for row.
        pooled = pooled[hit]

        # `Gemma4MultimodalEmbedder`: RMSNorm with NO learned scale, then the
        # projection into the language hidden size. The weightless norm is why
        # this is not just a `_matmul` -- `_rms_norm` takes a weight and there
        # is none, so it is written out.
        v = pooled.to(torch.float32)
        normed = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        out = self._matmul(normed, self._w["embedding_proj"])
        if trace is not None:
            trace["embed_vision"] = out.clone()
        return out


class Gemma4GpuVisionTower:
    """`Gemma4VisionTower` on the iGPU, every stage a Triton kernel.

    Weights are uploaded once, bf16 as the file ships them. What differs from
    `Gemma4VisionTower`, beyond the device:

    * Only the real patches are computed. Padding patches are masked out of
      every key and their rows are dropped after pooling, so they never reach
      a soft token; leaving them out up front saves their share of every GEMM
      and of the quadratic attention.
    * The clip bounds are applied inside the GEMMs (`gpu_kernels.matmul_clip`)
      rather than as separate passes, and gate|up writes the GeGLU directly.
    * Each norm's output is bf16, the dtype of the GEMM that reads it, and the
      attention takes fp16 operands, so the soft tokens are close to
      `Gemma4VisionTower`'s rather than equal.

    `trace` is not supported: the per-stage gate is the host tower's.
    """

    #: `matmul_clip` tiles per projection, swept on gfx1103 at the default
    #: image size with bf16 inputs.
    TILES = {
        "qkv": dict(block_m=64, block_n=256, block_k=64, num_warps=8),
        "o": dict(block_m=64, block_n=256, block_k=64, num_warps=8),
        "gate_up": dict(block_m=64, block_n=256, block_k=32, num_warps=8),
        "down": dict(block_m=64, block_n=256, block_k=64, num_warps=8),
    }

    def __init__(self, weights=None, model=None, device="cuda"):
        w = weights if weights is not None else vw.load_vision(model)
        self.dev = device
        self.fingerprint = w.get("fingerprint")

        def bf16(a):
            return (
                torch.from_numpy(vw.as_bf16(np.asarray(a)).view(np.int16))
                .view(torch.bfloat16)
                .to(device)
            )

        def f32(a):
            return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)

        self.patch_embd = bf16(w["patch_embd"])
        self.embedding_proj = bf16(w["embedding_proj"])
        self.out_hidden = self.embedding_proj.shape[1]
        self.position_table = f32(w["position_table"])  # [2, 10240, 768]
        unbounded = torch.full((HIDDEN,), float("inf"), device=device)
        self._no_clip = (-float("inf"), float("inf"), -unbounded, unbounded)
        self.layers = [self._layer_weights(lw, bf16, f32) for lw in w["layers"]]

    def _layer_weights(self, lw, bf16, f32):
        c = lw["clip"]

        def clip(parts, width):
            # Fused parts share their input bound (checked as `_to_torch` does)
            # and keep their own output bounds, as per-column vectors.
            if len({tuple(c[p][:2]) for p in parts}) != 1:
                raise ValueError(f"{parts} disagree on their input clip")
            lo = torch.cat([torch.full((width,), float(c[p][2])) for p in parts])
            hi = torch.cat([torch.full((width,), float(c[p][3])) for p in parts])
            in_lo, in_hi = c[parts[0]][:2]
            return float(in_lo), float(in_hi), lo.to(self.dev), hi.to(self.dev)

        L = {
            "qkv": torch.cat([bf16(lw[n]) for n in ("q", "k", "v")], 1).contiguous(),
            "o": bf16(lw["o"]),
            "gate_up": torch.cat([bf16(lw[n]) for n in ("gate", "up")], 1).contiguous(),
            "down": bf16(lw["down"]),
            "qkv_clip": clip(("q", "k", "v"), N_HEADS * HEAD_DIM),
            "o_clip": clip(("o",), HIDDEN),
            "gate_up_clip": clip(("gate", "up"), INTER),
            "down_clip": clip(("down",), HIDDEN),
        }
        for nm in ("norm1", "post_attn_norm", "norm2", "ffn_post_norm"):
            L[nm] = f32(lw[nm])
        L["q_norm"], L["k_norm"] = f32(lw["q_norm"]), f32(lw["k_norm"])
        return L

    def encode(self, pixels, pos_ids):
        """Patches in, language-space soft tokens out. See
        `Gemma4VisionTower.encode` for the arguments."""
        import gpu_kernels as G

        dev, bf = self.dev, torch.bfloat16
        pos = np.asarray(pos_ids)
        valid = ~(pos == -1).all(axis=-1)
        pv = pos[valid]
        lut_x, lut_y = (t.to(dev) for t in rope_tables(pv))

        # Patch embedding: rescale to [-1, 1], project, add the two position
        # rows -- a gather, as `Gemma4VisionTower._patch_embed` explains.
        px = 2.0 * (np.asarray(pixels, dtype=np.float32)[valid] - 0.5)
        px = torch.as_tensor(px, device=dev).to(bf)
        h = G.matmul_clip(px, self.patch_embd, *self._no_clip)
        at = torch.as_tensor(pv, device=dev).clamp(min=0)
        h = h + self.position_table[0][at[:, 0]] + self.position_table[1][at[:, 1]]

        x = G.rmsnorm(h, self.layers[0]["norm1"], RMS_EPS, out_dtype=bf)
        for i, L in enumerate(self.layers):
            t = self.TILES
            qkv = G.matmul_clip(x, L["qkv"], *L["qkv_clip"], **t["qkv"])
            q, k, v = G.vision_qkv_post(
                qkv, L["q_norm"], L["k_norm"], lut_x, lut_y, N_HEADS, RMS_EPS
            )
            a = G.attn_prefill_fa(
                q,
                k,
                v,
                N_HEADS,
                HEAD_DIM,
                scale=ATTN_SCALE,
                n_kv=N_HEADS,
                causal=False,
                out_dtype=bf,
            )
            a = G.matmul_clip(a, L["o"], *L["o_clip"], **t["o"])
            h, x = G.rmsnorm_residual(
                a, L["post_attn_norm"], RMS_EPS, h, next_norm=L["norm2"], next_dtype=bf
            )
            act = G.matmul_clip(
                x,
                L["gate_up"],
                *L["gate_up_clip"],
                glu=True,
                out_dtype=bf,
                **t["gate_up"],
            )
            d = G.matmul_clip(act, L["down"], *L["down_clip"], **t["down"])
            if i + 1 < len(self.layers):
                h, x = G.rmsnorm_residual(
                    d,
                    L["ffn_post_norm"],
                    RMS_EPS,
                    h,
                    next_norm=self.layers[i + 1]["norm1"],
                    next_dtype=bf,
                )
            else:
                h = G.rmsnorm_residual(d, L["ffn_post_norm"], RMS_EPS, h)

        # `_pool`, over the real rows only: the padding rows the host tower
        # zeroes before its scatter-add are simply absent here.
        idx, n_slots = pool_indices(pos, valid)
        slot = torch.as_tensor(np.asarray(idx)[valid], dtype=torch.long, device=dev)
        pooled = torch.zeros((n_slots, HIDDEN), dtype=torch.float32, device=dev)
        pooled.index_add_(0, slot, h)
        pooled *= math.sqrt(HIDDEN) / (POOLING_K * POOLING_K)
        hit = torch.zeros(n_slots, dtype=torch.bool, device=dev)
        hit[slot] = True
        # `Gemma4MultimodalEmbedder`: a weightless RMSNorm, then the projection.
        normed = G.rmsnorm(pooled[hit], None, RMS_EPS, out_dtype=bf)
        return G.matmul_clip(
            normed,
            self.embedding_proj,
            -float("inf"),
            float("inf"),
            torch.full((self.out_hidden,), -float("inf"), device=dev),
            torch.full((self.out_hidden,), float("inf"), device=dev),
        )
