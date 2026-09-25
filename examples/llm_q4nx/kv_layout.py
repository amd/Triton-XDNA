# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The one definition of Gemma4-E2B's KV cache layout.

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

    slab[L]  =  [ K region: ATTN_MAXL x REGION_W ][ V region: same ]

    REGION_W = N_ATTN_CU(2) * DH_A(512) = 1024

One device row holds the model's single MQA head TWICE, once per attention
compute unit. Every layer is built at the widest layer's geometry, so a sliding
layer's 256-wide head does not fill its 512-wide slot and does NOT sit
contiguously in it -- it is scattered as

    [ real_lo 128 | zeros 128 | real_hi 128 | zeros 128 ]

because `fused_decode/kernels/rope.cc` pairs dimension i with i + DH_A/2 = i+256
at the BUILD's head dim, and that scatter is what makes the kernel's fixed
pairing land on the real (i, i+128) pairs. Upstream records the contiguous
alternative as measured-wrong.

All 35 layer slabs are live, including the 20 that read a lower layer's cache:
`DECODE_KV_SRC` only reaches the builder when a Makefile sets `KV_SRC`, and
neither this repo's `decode_build.py` nor mlir-air's own `compile-decode` does,
so the template on disk is identity-mapped -- layer L reads and appends to slab
L. Filling only the 15 owning slabs decodes to garbage; that was measured.
"""

#: Attention compute units the decode template fans the KV head across.
N_ATTN_CU = 2
#: The build's head dim -- the full-attention layers' width. Sliding layers are
#: half this and are padded up to it by the interleave below.
DH_A = 512
#: One device row: the head once per CU.
REGION_W = N_ATTN_CU * DH_A
#: Regions per layer: K then V.
N_REGIONS = 2

K_REGION, V_REGION = 0, 1


def region_stride(attn_maxl):
    """Elements from the base of one region to the base of the next."""
    return attn_maxl * REGION_W


def layer_elems(attn_maxl):
    """Elements in one layer's slab -- both regions."""
    return N_REGIONS * region_stride(attn_maxl)


def slab_shape(n_layers, attn_maxl):
    return (n_layers, layer_elems(attn_maxl))


def lane(d, dh):
    """Where real dimension `d` of a `dh`-wide head sits in its DH_A slot.

    One expression for both head widths: at ``dh == DH_A`` the offset term is
    zero and this is the identity, so the full layers need no special case --
    which is the point, because a special case is a thing to get wrong. The
    Triton kernels spell the same map as

        tl.where(d < dh // 2, d, d + (DH_A // 2 - dh // 2))
    """
    half = dh // 2
    return d if d < half else d + (DH_A // 2 - half)


def region_view(slab, layer_idx, region, attn_maxl):
    """The `[attn_maxl, REGION_W]` view of one layer's K or V region.

    Works on a numpy array or a torch tensor -- it is basic slicing and a
    reshape, so neither copies. `slab` is the `[n_layers, layer_elems]` buffer.
    """
    rs = region_stride(attn_maxl)
    flat = slab[layer_idx][region * rs : (region + 1) * rs]
    return flat.reshape(attn_maxl, REGION_W)


def _runs(dh):
    """(dst_start, src_start, width) for the real lanes of one padded head.

    Two contiguous runs, which is the whole reason this layout can be written
    with basic slices rather than an index array: `rows[:, perm] = src` with
    `perm` an index array is an element-wise scatter, measured 17x slower than
    the identical bytes written as slices.
    """
    half = dh // 2
    return ((0, 0, half), (DH_A // 2, half, half))


def scatter_rows(dst, src, dh):
    """Write `src` `[n, dh]` into `dst` `[n, REGION_W]`, padded and duplicated.

    `dst` is a `region_view` slice; `src` is the prefill's dense head. numpy and
    torch both take this unchanged, and both convert on assignment -- but pass
    `src` already in the destination dtype where it is written more than once,
    because otherwise the conversion happens per copy.

    The padded lanes are NOT zeroed here: they are zero because the whole slab
    is zeroed before a prefill writes into it, and nothing ever writes them.
    Zeroing them per row would be half again the write volume for no effect.
    """
    for cu in range(N_ATTN_CU):
        base = cu * DH_A
        for d0, s0, w in _runs(dh):
            dst[:, base + d0 : base + d0 + w] = src[:, s0 : s0 + w]


def lane_index(dh):
    """Destination lanes for ONE row's real dimensions, all CU copies, in order.

    `scatter_rows` is the right shape for a prefill, which writes P rows at
    once and wants contiguous runs. A decode writes a single row per layer per
    token, where the cost is the number of operations rather than the bytes:
    the four slice assignments become four kernel launches. Paired with the
    source repeated once per CU,

        dst[pos:pos + 1, lane_index(dh)] = row.repeat(1, N_ATTN_CU)

    is one launch and lands the identical bytes. Derived from the same `_runs`
    so the two spellings cannot drift.
    """
    import numpy as np

    out = []
    for cu in range(N_ATTN_CU):
        base = cu * DH_A
        for d0, _s0, w in _runs(dh):
            out.append(np.arange(base + d0, base + d0 + w))
    return np.concatenate(out)


def gather_rows(src, dh, out=None):
    """The inverse of `scatter_rows`: `[n, REGION_W]` -> `[n, dh]`.

    For `kv_stack()` and `--compare-cpu`, which want the dense head back. Reads
    the first CU's copy; the second is identical by construction.
    """
    import numpy as np

    n = src.shape[0]
    if out is None:
        out = np.empty((n, dh), dtype=np.float32)
    for d0, s0, w in _runs(dh):
        out[:, s0 : s0 + w] = src[:, d0 : d0 + w]
    return out
