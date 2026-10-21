# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The one definition of the Gemma4 KV cache layout (E2B and E4B).

Three things read this cache -- the Triton prefill, mlir-air's fused NPU decode
and the Triton iGPU decode -- and they used to hold two layouts between them,
with a per-turn rearrangement in `FusedDecoder.seed_kv` bridging the gap. They
now share one, and this module is where it is written down. Nothing else should
compute an offset into the cache.

That is not tidiness. mlir-air's `seed_kv` says of this layout that "getting it
wrong does not raise -- it seeds plausible garbage", and its own history bears
that out: seeding without the padded-head interleave produced a correct first
token followed by fluent nonsense. A layout that four call sites each re-derive
is a layout that will eventually be re-derived wrongly in one of them.

The layout, as the decode template is built:

    slab[L]  =  [ K region: ATTN_MAXL x REGION_W ][ V region: same ][ copies ]

    REGION_W = N_KV x DH_A(512)

A device row holds the model's KV heads side by side, DH_A each. E2B has one
(MQA) head, held once: the decode reads it back once and fans it to both
attention compute units, and its append also writes each row a second time,
past the K and V regions, where nothing reads it back; a prefill leaves that
part zero. E4B has two heads, one per attention compute unit, and no copy. Every layer is built at
the widest layer's geometry, so a sliding layer's 256-wide head does not fill
its 512-wide slot. Where it sits depends on the region:

    K:  [ real_lo 128 | zeros 128 | real_hi 128 | zeros 128 ]
    V:  [ real 256                | zeros 256               ]

K is scattered because `fused_decode/kernels/rope.cc` pairs dimension i with
i + DH_A/2 = i+256 at the BUILD's head dim, and that scatter is what makes the
kernel's fixed pairing land on the real (i, i+128) pairs. Upstream records the
contiguous alternative as measured-wrong for K. V is not roped, and the decode
streams a sliding layer's o-proj weights for the front half of the slot only,
so its V is contiguous there. `FusedDecoder.seed_kv` is the reference for both.

Every layer's slab is live, including those of the layers that read a lower
layer's cache:
`DECODE_KV_SRC` only reaches the builder when a Makefile sets `KV_SRC`, and
neither this repo's `decode_build.py` nor mlir-air's own `compile-decode` does,
so the template on disk is identity-mapped -- layer L reads and appends to slab
L. Filling only the owning slabs decodes to garbage; that was measured.
"""

from config import N_KV_HEADS as N_KV

#: Copies of the K and V regions in one layer's slab (the builder's
#: `KV_COPIES`): a single head is replicated for the two attention CUs.
KV_COPIES = 2 if N_KV == 1 else 1
#: The build's head dim -- the full-attention layers' width. Sliding layers are
#: half this and are padded up to it, as the module docstring describes.
DH_A = 512
#: One device row: every KV head, DH_A apart.
REGION_W = N_KV * DH_A
#: Regions per layer: K then V.
N_REGIONS = 2

K_REGION, V_REGION = 0, 1


def region_stride(attn_maxl):
    """Elements from the base of one region to the base of the next."""
    return attn_maxl * REGION_W


def layer_elems(attn_maxl):
    """Elements in one layer's slab -- both regions, and their copies."""
    return KV_COPIES * N_REGIONS * region_stride(attn_maxl)


def slab_shape(n_layers, attn_maxl):
    return (n_layers, layer_elems(attn_maxl))


def lane_shift(dh, region):
    """How far the second half of a `dh`-wide head is moved in its slot.

    Zero for a full-width head in either region, and for V in both widths, so
    `lane` is the identity there. The Triton kernels take this as a constexpr
    and spell `lane` inline.
    """
    if region not in (K_REGION, V_REGION):
        raise ValueError(f"region must be K_REGION or V_REGION, got {region!r}")
    return DH_A // 2 - dh // 2 if region == K_REGION else 0


def lane(d, dh, region):
    """Where real dimension `d` of a `dh`-wide head sits in its DH_A slot.

    One expression for both head widths and both regions; the Triton kernels
    spell the same map as

        tl.where(d < dh // 2, d, d + lane_shift(dh, region))
    """
    half = dh // 2
    return d if d < half else d + lane_shift(dh, region)


def region_view(slab, layer_idx, region, attn_maxl):
    """The `[attn_maxl, REGION_W]` view of one layer's K or V region.

    Works on a numpy array or a torch tensor -- it is basic slicing and a
    reshape, so neither copies. `slab` is the `[n_layers, layer_elems]` buffer.
    """
    rs = region_stride(attn_maxl)
    flat = slab[layer_idx][region * rs : (region + 1) * rs]
    return flat.reshape(attn_maxl, REGION_W)


def _runs(dh, region):
    """(dst_start, src_start, width) for the real lanes of one padded head.

    Two contiguous runs, which is the whole reason this layout can be written
    with basic slices rather than an index array: `rows[:, perm] = src` with
    `perm` an index array is an element-wise scatter, measured 17x slower than
    the identical bytes written as slices.
    """
    half = dh // 2
    return ((0, 0, half), (half + lane_shift(dh, region), half, half))


def scatter_rows(dst, src, dh, region):
    """Write `src` `[n, N_KV * dh]` into `dst` `[n, REGION_W]`, each head padded
    into its own DH_A slot.

    `dst` is a `region_view` slice; `src` is the prefill's dense head. numpy and
    torch both take this unchanged, and both convert on assignment -- but pass
    `src` already in the destination dtype where it is written more than once,
    because otherwise the conversion happens per copy.

    The padded lanes are NOT zeroed here: they are zero because the whole slab
    is zeroed before a prefill writes into it, and nothing ever writes them.
    Zeroing them per row would be half again the write volume for no effect.
    """
    for h in range(N_KV):
        for d0, s0, w in _runs(dh, region):
            d0, s0 = d0 + h * DH_A, s0 + h * dh
            dst[:, d0 : d0 + w] = src[:, s0 : s0 + w]


def lane_index(dh, region):
    """Destination lanes for ONE row's real dimensions, in order.

    `scatter_rows` is the right shape for a prefill, which writes P rows at
    once and wants contiguous runs. A decode writes a single row per layer per
    token, where the cost is the number of operations rather than the bytes:
    the slice assignments become one kernel launch each. With

        dst[pos:pos + 1, lane_index(dh, region)] = row

    is one launch and lands the identical bytes. Derived from the same `_runs`
    so the two spellings cannot drift.
    """
    import numpy as np

    return np.concatenate([np.arange(d0, d0 + w) for d0, _s0, w in _runs(dh, region)])


def gather_rows(src, dh, region, out=None):
    """The inverse of `scatter_rows`: `[n, REGION_W]` -> `[n, N_KV * dh]`.

    For `kv_stack()` and `--compare-cpu`, which want the dense head back.
    """
    import numpy as np

    n = src.shape[0]
    if out is None:
        out = np.empty((n, N_KV * dh), dtype=np.float32)
    for h in range(N_KV):
        for d0, s0, w in _runs(dh, region):
            d0, s0 = d0 + h * DH_A, s0 + h * dh
            out[:, s0 : s0 + w] = src[:, d0 : d0 + w]
    return out
