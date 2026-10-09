#!/usr/bin/env python3
"""Check `kv_layout` against mlir-air's own `seed_kv`, byte for byte.

This is the load-bearing test of the unified KV cache. `kv_layout` restates a
layout that mlir-air defines, and the failure mode of restating it wrongly is
not a crash -- it is a correct first token followed by fluent nonsense, because
the first token comes from the prefill and only the decode reads the cache.
So the check is equality of the produced buffer against upstream's own code,
not a tolerance on logits.

Upstream's `FusedDecoder.seed_kv` runs here as it is, on a decoder object
that holds only what it reads, with the geometry taken from mlir-air's decode
builder. A copy of the method would keep passing after upstream changed the
layout.

Needs no hardware, but does need mlir-air's sources and its Python bindings
for the builder.

Not picked up by `scripts/run_tests.py` -- `llm_q4nx` is excluded from the
sweep as a library, like `test_gpu_kernels.py`. Run it by hand. Exits 77 --
graded as a skip -- when mlir-air's sources are not present.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kv_layout as KVL

REGIONS = (KVL.K_REGION, KVL.V_REGION)


class _HostBO:
    """Where `seed_kv` uploads; the test reads `KV` instead."""

    def write(self, *a):
        pass

    def sync(self, *a):
        pass


def _upstream_slab(air, gw, fd, ks, vs, attn_maxl, bf16):
    """`FusedDecoder.seed_kv` itself, on a decoder with only what it reads."""
    n = gw.NUM_LAYERS
    dec = object.__new__(air.FusedDecoder)
    dec.np, dec.bf16, dec.gw = np, bf16, gw
    dec.xrt = SimpleNamespace(
        xclBOSyncDirection=SimpleNamespace(XCL_BO_SYNC_BO_TO_DEVICE=0)
    )
    dec.kvc = _HostBO()
    dec.UNI, dec.ATTN_MAXL = n, attn_maxl
    dec.DH_A, dec.REGION_W = fd.DH_A, fd.REGION_W
    # As `FusedDecoder.__init__` sets it.
    dec.SWA = {L for L in range(n) if fd.arm_of_layer(L) & fd.SWA_ARM_BIT}
    dec.KV = np.zeros((n, KVL.layer_elems(attn_maxl)), dtype=bf16)
    dec.seed_kv(ks, vs, ks[0].shape[0])
    return dec.KV


def _ours(ks, vs, head_dim, n_layers, attn_maxl, bf16):
    KV = np.zeros((n_layers, KVL.layer_elems(attn_maxl)), dtype=bf16)
    P = ks[0].shape[0]
    for L in range(n_layers):
        dh = head_dim(L)
        for reg, src in ((KVL.K_REGION, ks[L]), (KVL.V_REGION, vs[L])):
            dst = KVL.region_view(KV, L, reg, attn_maxl)[:P]
            KVL.scatter_rows(dst, np.asarray(src, np.float32).astype(bf16), dh, reg)
    return KV


def main():
    try:
        sys.path.insert(0, _air_llms())
        import gemma4_e2b_q4nx_inference as air
        import gemma4_e2b_q4nx_weights as gw
        from ml_dtypes import bfloat16
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: mlir-air's gemma4 sources are not importable: {e}")
        return 77

    rng = np.random.default_rng(0)
    ok = True
    head_dim = lambda L: gw.head_dim(gw.kv_source_layer(L))  # noqa: E731
    n_layers = gw.NUM_LAYERS
    kv_src = [gw.kv_source_layer(L) for L in range(n_layers)]
    # Both head widths, a context that is not a multiple of anything, and one
    # that is -- the interleave is per row, so a ragged P is the interesting
    # case for the reshape rather than for the scatter.
    for attn_maxl, P in ((128, 6), (256, 163)):
        fd = air._load_builder(n_layers, attn_maxl, kv_src)
        if (fd.DH_A, fd.REGION_W) != (KVL.DH_A, KVL.REGION_W):
            print(
                f"  builder geometry DH_A={fd.DH_A} REGION_W={fd.REGION_W}, "
                f"kv_layout has {KVL.DH_A} {KVL.REGION_W}  FAIL"
            )
            ok = False
            continue
        ks, vs = [], []
        for L in range(n_layers):
            ks.append(rng.standard_normal((P, head_dim(L)), dtype=np.float32))
            vs.append(rng.standard_normal((P, head_dim(L)), dtype=np.float32))

        want = _upstream_slab(air, gw, fd, ks, vs, attn_maxl, bfloat16)
        got = _ours(ks, vs, head_dim, n_layers, attn_maxl, bfloat16)
        for reg, name in ((KVL.K_REGION, "K"), (KVL.V_REGION, "V")):
            bad = [
                L
                for L in range(n_layers)
                if not np.array_equal(
                    KVL.region_view(want, L, reg, attn_maxl).view(np.uint16),
                    KVL.region_view(got, L, reg, attn_maxl).view(np.uint16),
                )
            ]
            print(
                f"  layers={n_layers} ATTN_MAXL={attn_maxl} P={P} {name}  "
                + ("identical" if not bad else f"DIFFER on layers {bad}")
            )
            ok &= not bad

        # And the round trip, since kv_stack() and --compare-cpu depend on it.
        for L in range(n_layers):
            for reg, src in ((KVL.K_REGION, ks[L]), (KVL.V_REGION, vs[L])):
                back = KVL.gather_rows(
                    KVL.region_view(got, L, reg, attn_maxl)[:P], head_dim(L), reg
                )
                if not np.array_equal(back, src.astype(bfloat16).astype(np.float32)):
                    print(f"  layer {L} region {reg}: no round trip  FAIL")
                    ok = False

    # `lane_index` is the decode's one-launch spelling of `scatter_rows`. The
    # two must land the same bytes or the prompt and the generated tokens go
    # into the cache differently -- which no gate would catch at one token.
    for dh in (256, 512):
        for reg in REGIONS:
            row = np.arange(1, dh + 1, dtype=np.float32).reshape(1, dh)
            a = np.zeros((1, KVL.REGION_W), np.float32)
            KVL.scatter_rows(a, row, dh, reg)
            b = np.zeros((1, KVL.REGION_W), np.float32)
            b[0, KVL.lane_index(dh, reg)] = np.tile(row, (1, KVL.N_ATTN_CU))
            if not np.array_equal(a, b):
                print(f"  lane_index disagrees with scatter_rows at dh={dh}  FAIL")
                ok = False
    print("  lane_index lands the same bytes as scatter_rows")

    # The lane map the Triton kernels spell inline must agree with this one.
    for dh in (256, 512):
        half = dh // 2
        for reg in REGIONS:
            shift = KVL.lane_shift(dh, reg)
            for d in range(dh):
                inline = d if d < half else d + shift
                if inline != KVL.lane(d, dh, reg):
                    print(f"  lane map disagrees at dh={dh} d={d}  FAIL")
                    ok = False
                    break
    print("  lane map matches the kernels' inline form")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _air_llms():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import airsrc

    root = airsrc.air_llms_root()
    airsrc.add_air_paths("gemma4_e2b_q4nx")
    return str(root / "gemma4_e2b_q4nx")


if __name__ == "__main__":
    sys.exit(main())
