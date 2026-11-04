#!/usr/bin/env python3
"""Check `gemma4_vision_q4nx.unpack` against the packing it inverts.

A wrong `mm_n`/`mm_k` pair does not fail on its own: it yields a matrix of
the right shape with values in the wrong places. So the check is that
re-packing what was read is bitwise equal to the bytes in the file.

`pack` is the inverse of `unpack`, so their agreement alone only shows the
pair is self-consistent; comparing against the file's bytes makes it a real
check.

Needs `vision_weight.q4nx`. Exits 77 (graded as a skip) when it cannot be
resolved.

    python test_gemma4_vision_q4nx.py [model-dir-or-repo-id]
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gemma4_vision_q4nx as vw

SKIP_EXIT_CODE = 77

#: (key, K, out) for one tensor of each distinct packed geometry: square,
#: wide (K < out), tall (K > out), and the output projection.
_CASES = [
    ("model.vision.0.vision_attn.q_proj.weight", 768, 768),
    ("model.vision.0.ffn.gate_proj.weight", 768, 3072),
    ("model.vision.0.ffn.down_proj.weight", 3072, 768),
    ("model.vision.embedding_projection.weight", 768, 1536),
]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    src = argv[0] if argv else None
    try:
        r = vw.VisionWeights(src)
    except Exception as e:  # unresolvable file, no network, no local bundle
        print(f"[skip] no vision_weight.q4nx: {type(e).__name__}: {e}")
        return SKIP_EXIT_CODE

    print(f"[vision-weights] {r.path} ({r.fingerprint()})")
    bad = 0
    for key, K, out in _CASES:
        stored = r.raw(key)
        back = vw.pack(r.proj(key, K, out))
        # Compare as int16 so the check is bitwise, not a float comparison.
        ok = np.array_equal(stored.view(np.int16), back.view(np.int16))
        bad += not ok
        print(
            f"  {'ok  ' if ok else 'FAIL'} {tuple(stored.shape)} -> "
            f"[{K}, {out}]  {key}"
        )

    # The position table is packed per axis from a transposed input, so a
    # swapped-axes bug would not show in the shape.
    key = "model.vision.patch_embedder.position_embedding_table"
    stored = r.raw(key)
    un = [
        vw.unpack(stored[i], vw.POSITION_EMBEDDING_SIZE, vw.HIDDEN)
        for i in range(stored.shape[0])
    ]
    back = np.stack([vw.pack(u) for u in un])
    ok = np.array_equal(stored.view(np.int16), back.view(np.int16))
    bad += not ok
    print(
        f"  {'ok  ' if ok else 'FAIL'} {tuple(stored.shape)} -> 2x[10240, 768]  {key}"
    )

    # The round trips above pass for any self-consistent tile pair, so also
    # check that a swapped mm_n/mm_k is rejected.
    key, K, out = _CASES[0]
    try:
        vw.unpack(r.raw(key), K, out, mm_n=vw.MM_TILE_K, mm_k=vw.MM_TILE_N)
    except ValueError:
        print("  ok   swapped mm_n/mm_k is rejected")
    else:
        bad += 1
        print("  FAIL swapped mm_n/mm_k was accepted")

    print(f"[vision-weights] {'PASS' if not bad else f'FAIL ({bad})'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
