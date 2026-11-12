# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4's vision tower: patches in, language-space soft tokens out.

E2B and E4B share the tower; only the final projection's width differs, and it
is read off the weights.

The tower is a 16-layer ViT at hidden 768, 12 heads of 64, built from the same
Gemma block as the text side (four-norm sandwich, GELU-tanh GLU, per-head q/k
norms). It differs from the text side in these ways:

* Attention is bidirectional; the only mask is padding.
  `LlamaPrefill._attention` and `gpu_kernels.attn_prefill` are causal, so
  `_attention` here is a separate implementation.
* RoPE is two-dimensional. Lanes 0-31 of every head rotate on the patch's x
  coordinate and lanes 32-63 on its y, each with 16 frequencies at base 100.
  Swapping the halves keeps every shape valid but scrambles positions.
* Every projection is clipped: `clamp(input) -> matmul -> clamp(output)`, with
  per-layer, per-projection bounds from the weight file. This is part of the
  model (`Gemma4ClippableLinear`, `use_clipped_linears: true`), so it applies
  on the bf16 path too.
* v gets an RMSNorm with no weight; q and k get their learned per-head norm.

The attention scale is 1.0, not `head_dim**-0.5`, as on the text side
(`config.ATTN_SCALE`).

The reference is `transformers`' `Gemma4VisionModel`
(`transformers/models/gemma4/modeling_gemma4.py`). FastFlowLM's NPU
implementation agrees with it; departures from FastFlowLM are noted inline.

Placement: `_matmul`, `_rms_norm` and `_geglu` belong to the host object, so
they go to the NPU under the same `--ops` flags as the text prefill. RoPE
stays on the host or the iGPU; attention is placed as `_attention` describes.

Do not run this with `--ops all`. The per-head q/k/v norms fold the head axis
into the row axis, so at the default resolution `_head_norm` gives the NPU
`rms_norm` kernel a `[2520 * 12, 64]` problem, which times out
(`ERT_CMD_STATE_TIMEOUT`). Use the default op set (`matmul`, `geglu`).
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

#: `Gemma4VisionAttention.scaling`; see the module docstring.
ATTN_SCALE = 1.0

#: The NPU-resident projections, after fusing, by their name in a layer dict.
#: `q|k|v` and `gate|up` are concatenated into one GEMM each: they read the
#: same activation and share an input clip bound (`_to_torch` checks this).
#: Their output bounds differ and are applied per slice. One wider GEMM is
#: faster than several narrow ones, as with `LlamaPrefill.NPU_WEIGHTS`.
RESIDENT_PROJS = ("qkv", "o", "gate_up", "down")

#: `(fused name, parts)`. `_attn_block` and the MLP split the result back on
#: the parts' widths.
_FUSIONS = (("qkv", ("q", "k", "v")), ("gate_up", ("gate", "up")))


def rope_tables(pos_ids, dtype=torch.float32):
    """Per-patch RoPE tables for the two spatial dimensions.

    `pos_ids` is `[P, 2]`, the (x, y) grid coordinate of each patch, with
    padding patches at `(-1, -1)`. Returns `(lut_x, lut_y)`, each `[P, 32]` in
    the layout `_rope` takes: 16 cosines then 16 sines.

    The frequencies follow
    `Gemma4VisionRotaryEmbedding.compute_default_rope_parameters`: the spatial
    dim (32), not the head dim, is the exponent denominator,
    `100 ** (-[0, 2, ..., 30] / 32)`, so x and y get identical frequency ranges.

    Negative coordinates are clamped to 0, as HF does. Padding patches are
    masked out of attention, so the clamp only keeps the table finite.
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

    Padding patches clamp to (0, 0) and land in slot 0, as in HF. Their hidden
    states are zeroed before pooling, so they add nothing, and the divisor is a
    fixed `k*k`. This is computed as a scatter-add instead of HF's `[P,
    n_slots]` one-hot matmul.
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

    Wraps a `host` (a `Gemma4Prefill`) whose `_matmul`, `_rms_norm`, `_geglu`
    and `_head_norm` it calls, so `--backend npu --ops ...` places both towers
    the same way and they share the host's driver for NPU residency
    (`kernels.resident_weight`).
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
        """Convert the reader's numpy arrays to the tensors the operators take.

        Projections stay bf16 (reinterpreted, not converted); norms go to f32.
        This must happen up front: `_matmul` tells a torch tensor from an
        NPU-resident handle by whether it has `.to`, so a numpy array would be
        mistaken for a resident weight.
        """
        bf = torch.bfloat16
        out = dict(w)
        out["patch_embd"] = vw.to_torch(vw.as_bf16(w["patch_embd"]), bf)
        out["embedding_proj"] = vw.to_torch(vw.as_bf16(w["embedding_proj"]), bf)
        # Stays numpy: `_patch_embed` gathers and converts only the rows it uses.
        out["position_table"] = np.asarray(w["position_table"])

        def _projs(lw):
            """One layer's projections, fused on the output axis. Equivalent
            to separate GEMMs given a shared input clamp."""
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
        # The patch count is fixed per image tier: one dispatch per layer.
        return self.host._geglu(gate, up, whole=True)

    def _head_norm(self, x, weight):
        return self.host._head_norm(x, weight, N_HEADS, HEAD_DIM)

    def _rope(self, x, lut):
        return self.host._rope(x, lut, N_HEADS)

    # ---- operators of its own ----

    def _project(self, x, w, epilogue, **mm):
        """`epilogue(x @ w)`. On the NPU the epilogue runs on the result in
        place (`kernels.triton_matmul`)."""
        if self.host._device("matmul") == "npu":
            return self._matmul(x, w, epilogue=epilogue, **mm)
        return epilogue(self._matmul(x, w, **mm))

    def _clipped(self, x, w, clip, **mm):
        """`Gemma4ClippableLinear`: clamp in, project, clamp out.

        `clip` is the `(in_lo, in_hi, out_lo, out_hi)` tuple from the weight
        file. An unclipped projection carries `+/-inf` bounds.
        """
        lo_i, hi_i, lo_o, hi_o = clip
        return self._project(
            x.clamp(lo_i, hi_i), w, lambda y: y.clamp(lo_o, hi_o), **mm
        )

    def _clipped_split(self, x, w, clip, parts, widths, **mm):
        """A fused clipped projection, split back into `parts`.

        The parts share an input bound (`_to_torch` checks this), so the input
        is clamped once; each output slice gets its own bound.
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

        `LlamaPrefill._attention` is causal, so it cannot be reused here.

        `valid` is a bool `[P]`, False for padding patches. Only keys are
        masked. Padded query rows are dropped by the pooler; masking them too
        would give an all `-inf` softmax and NaNs in the residual stream.

        At the default resolution the score tensor is `[12, 2520, 2520]` per
        layer, so this uses the iGPU when there is one.
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
            # SDPA avoids materializing the large score matrix. The [1, 1, P]
            # mask broadcasts over head and query; it is omitted when every key
            # is real, which is how `encode` calls it.
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

        Applies when attention is placed on the NPU, every patch is real, and
        the kernel supports `P`.
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

        Upstream writes the position embedding as
        `one_hot(x, 10240) @ table[0] + one_hot(y, 10240) @ table[1]`; this does
        the equivalent row gather. Padding rows get zero, matching upstream's
        `torch.where(padding_positions, 0.0, ...)`.
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

        The Gemma sandwich, as on the text side:

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
        # v's norm has no learned scale.
        v = self._head_norm(vp, None)
        q = self._rope2d(q, lut_x, lut_y)
        k = self._rope2d(k, lut_x, lut_y)
        a = self._attention(q, k, v, valid)
        return self._clipped(a, w["o"], c["o"], stage_key=f"vision_o_L{L}")

    def _rope2d(self, x, lut_x, lut_y):
        """Rotate lanes 0-31 of each head on x and lanes 32-63 on y.

        `apply_multidimensional_rope` splits the 64-lane head into two pieces
        of 32 and applies `apply_rotary_pos_emb` to each, which is
        `LlamaPrefill._rope` at `DH = 32`: pairs `(i, i+16)` within a piece.
        Reusing `_rope` also gives this the iGPU under `hetero`.
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
        a fixed `k*k` even for partly padded tiles, since padded rows are zeroed
        first (HF's `weights / k_squared`).

        The final `sqrt(768)` scale is `Gemma4VisionPooler.root_hidden_size`.
        """
        with self.timer.track("vision_pool"):
            idx, n_slots = pool_indices(pos_ids, valid)
            keep = torch.as_tensor(np.asarray(valid), dtype=torch.bool)
            h = torch.where(keep[:, None], h, 0.0)
            ti = torch.as_tensor(np.asarray(idx), dtype=torch.long)
            out = torch.zeros((n_slots, HIDDEN), dtype=torch.float32)
            out.index_add_(0, ti, h.to(torch.float32))
            out /= POOLING_K * POOLING_K
            # Slots any patch reached. Padding lands in slot 0, so slot 0 is
            # always live, as upstream (its mask comes from the one-hot weights).
            hit = torch.zeros(n_slots, dtype=torch.bool)
            hit[ti] = True
            return out * math.sqrt(HIDDEN), hit

    # ---- entry points ----

    def make_npu_resident(self):
        """Pad and stage every projection on the device, once.

        As in `LlamaPrefill.make_npu_resident`, the host copy is freed, so the
        torch path cannot run afterwards.
        """
        if self._resident:
            return
        import kernels

        for w in self._w["layers"]:
            for nm in RESIDENT_PROJS:
                # Padded to the contracted K (`_matmul` passes a `stage_key`
                # for each of these).
                w[nm] = kernels.resident_weight(w[nm], exact=True)
        self._resident = True

    def encode(self, pixels, pos_ids, trace=None):
        """Patches in, language-space soft tokens out.

        `pixels` is `[P, 768]` (3 channels x 16 x 16, flattened) and `pos_ids`
        is `[P, 2]` with padding patches at `(-1, -1)`, as `Gemma4ImageProcessor`
        returns them. Returns `[n_soft_tokens, out_hidden]`, to be scattered
        over the image placeholders in the prompt.

        `trace`, if given, is a dict filled with every stage's intermediate,
        under the same keys as `gemma4_vision_hf.encode`, for a per-stage diff.
        """
        pos = np.asarray(pos_ids)
        valid = ~(pos == -1).all(axis=-1)
        # Run the layers on the real patches only. Padding is masked as a key,
        # every other operator is row-wise, and the pooler zeroes it, so
        # dropping it does not change the real rows. `full` restores the
        # padded layout, with zeros, for the pooler and the trace.
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

        # Drop slots no patch reached, so the row count matches the image
        # placeholders. Done before the projection, in upstream's order, so
        # traces compare row for row.
        pooled = pooled[hit]

        # `Gemma4MultimodalEmbedder`: RMSNorm with no learned scale (which
        # `_rms_norm` does not support), then the projection into the language
        # hidden size.
        v = pooled.to(torch.float32)
        normed = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        out = self._matmul(normed, self._w["embedding_proj"])
        if trace is not None:
            trace["embed_vision"] = out.clone()
        return out


class Gemma4GpuVisionTower:
    """`Gemma4VisionTower` on the iGPU, every stage a Triton kernel.

    Weights are uploaded once as bf16. Differences from `Gemma4VisionTower`:

    * Only the real patches are computed.
    * Clip bounds are applied inside the GEMMs (`gpu_kernels.matmul_clip`),
      and the gate|up GEMM writes the GeGLU directly.
    * Norm outputs are bf16 and attention takes fp16 operands, so the soft
      tokens are close to `Gemma4VisionTower`'s but not equal.

    `trace` is not supported.
    """

    #: `matmul_clip` tiles per projection, tuned on gfx1152 (Radeon 860M)
    #: at 2520 patches (a 280-token budget) with bf16 inputs.
    TILES = {
        "qkv": dict(block_m=64, block_n=256, block_k=32, num_warps=4),
        "o": dict(block_m=64, block_n=128, block_k=64, num_warps=4, num_stages=2),
        "gate_up": dict(block_m=64, block_n=128, block_k=32, num_warps=4),
        "down": dict(block_m=64, block_n=128, block_k=64, num_warps=4, num_stages=2),
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
            # Fused parts share their input bound and keep their own output
            # bounds, as per-column vectors.
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

        # Patch embedding, as in `Gemma4VisionTower._patch_embed`.
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

        # `_pool`, over the real rows only.
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
