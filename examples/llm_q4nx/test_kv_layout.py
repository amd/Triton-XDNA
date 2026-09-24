#!/usr/bin/env python3
"""Check `kv_layout` against mlir-air's own `seed_kv`, byte for byte.

This is the load-bearing test of the unified KV cache. `kv_layout` restates a
layout that mlir-air defines, and the failure mode of restating it wrongly is
not a crash -- it is a correct first token followed by fluent nonsense, because
the first token comes from the prefill and only the decode reads the cache.
So the check is equality of the produced buffer against upstream's own code,
not a tolerance on logits.

Needs no hardware: it builds the cache with both implementations in numpy and
compares. It does need mlir-air's sources, for `_head_perm`.

Not picked up by `scripts/run_tests.py` -- `llm_q4nx` is excluded from the
sweep as a library, like `test_gpu_kernels.py`. Run it by hand. Exits 77 --
graded as a skip -- when mlir-air's sources are not present.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kv_layout as KVL


def _upstream_slab(ks, vs, head_dim, n_layers, attn_maxl, bf16, head_perm):
    """`FusedDecoder.seed_kv`'s staging loop, verbatim but for the BO write."""
    KV = np.zeros((n_layers, KVL.layer_elems(attn_maxl)), dtype=bf16)
    RW = KVL.REGION_W
    RS = KVL.region_stride(attn_maxl)
    P = ks[0].shape[0]
    for L in range(n_layers):
        perm = head_perm(head_dim(L), KVL.DH_A)
        for reg, src in ((0, ks[L]), (1, vs[L])):
            src = np.asarray(src, np.float32).reshape(P, -1)
            rows = KV[L, reg * RS : reg * RS + P * RW].reshape(P, RW)
            rows[:, perm] = src.astype(bf16)
            rows[:, KVL.DH_A + perm] = src.astype(bf16)
    return KV


def _ours(ks, vs, head_dim, n_layers, attn_maxl, bf16):
    KV = np.zeros((n_layers, KVL.layer_elems(attn_maxl)), dtype=bf16)
    P = ks[0].shape[0]
    for L in range(n_layers):
        dh = head_dim(L)
        for reg, src in ((KVL.K_REGION, ks[L]), (KVL.V_REGION, vs[L])):
            dst = KVL.region_view(KV, L, reg, attn_maxl)[:P]
            KVL.scatter_rows(dst, np.asarray(src, np.float32).astype(bf16), dh)
    return KV


def main():
    try:
        sys.path.insert(0, _air_llms())
        import gemma4_e2b_q4nx_weights as gw
        from gemma4_e2b_q4nx_requant import _head_perm
        from ml_dtypes import bfloat16
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: mlir-air's gemma4 sources are not importable: {e}")
        return 77

    rng = np.random.default_rng(0)
    ok = True
    # Both head widths, a context that is not a multiple of anything, and one
    # that is -- the interleave is per row, so a ragged P is the interesting
    # case for the reshape rather than for the scatter.
    for n_layers, attn_maxl, P in ((gw.NUM_LAYERS, 128, 6), (gw.NUM_LAYERS, 256, 163)):
        ks, vs = [], []
        for L in range(n_layers):
            dh = gw.head_dim(gw.kv_source_layer(L))
            ks.append(rng.standard_normal((P, dh), dtype=np.float32))
            vs.append(rng.standard_normal((P, dh), dtype=np.float32))
        head_dim = lambda L: gw.head_dim(gw.kv_source_layer(L))  # noqa: E731

        want = _upstream_slab(
            ks, vs, head_dim, n_layers, attn_maxl, bfloat16, _head_perm
        )
        got = _ours(ks, vs, head_dim, n_layers, attn_maxl, bfloat16)
        same = np.array_equal(want.view(np.uint16), got.view(np.uint16))
        print(
            f"  layers={n_layers} ATTN_MAXL={attn_maxl} P={P}  "
            f"{'identical' if same else 'DIFFER'}"
        )
        ok &= same

        # And the round trip, since kv_stack() and --compare-cpu depend on it.
        for L in range(n_layers):
            dh = head_dim(L)
            back = KVL.gather_rows(
                KVL.region_view(got, L, KVL.K_REGION, attn_maxl)[:P], dh
            )
            if not np.array_equal(back, ks[L].astype(bfloat16).astype(np.float32)):
                print(f"  layer {L}: gather_rows did not round-trip  FAIL")
                ok = False
                break

    # `lane_index` is the decode's one-launch spelling of `scatter_rows`. The
    # two must land the same bytes or the prompt and the generated tokens go
    # into the cache differently -- which no gate would catch at one token.
    for dh in (256, 512):
        row = np.arange(1, dh + 1, dtype=np.float32).reshape(1, dh)
        a = np.zeros((1, KVL.REGION_W), np.float32)
        KVL.scatter_rows(a, row, dh)
        b = np.zeros((1, KVL.REGION_W), np.float32)
        b[0, KVL.lane_index(dh)] = np.tile(row, (1, KVL.N_ATTN_CU))
        if not np.array_equal(a, b):
            print(f"  lane_index disagrees with scatter_rows at dh={dh}  FAIL")
            ok = False
    print("  lane_index lands the same bytes as scatter_rows")

    # The lane map the Triton kernels spell inline must agree with this one.
    for dh in (256, 512):
        half = dh // 2
        for d in range(dh):
            inline = d if d < half else d + (KVL.DH_A // 2 - half)
            if inline != KVL.lane(d, dh):
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
