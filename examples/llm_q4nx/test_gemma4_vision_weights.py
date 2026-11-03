#!/usr/bin/env python3
"""Check `gemma4_vision_weights.unpack` against the packing it inverts.

The vision tower's weights are tile-packed for FLM's MM engine and we un-pack
them to read them. A wrong `mm_n`/`mm_k` pair does not fail: it produces a
matrix of exactly the right shape holding the right numbers in the wrong places,
and the tower built on it yields plausible-looking soft tokens and a confidently
wrong caption. So the check is that a re-pack of what we read is **bitwise**
equal to the bytes in the file, not a tolerance on anything downstream.

Re-packing with our own `pack` rather than with the converter's `einops`
rearrange is deliberate: `pack` is written as the literal inverse of `unpack`,
so agreement between them proves only that the pair is self-consistent. What
makes this a real test is the third party -- the file, whose bytes neither
function produced.

Needs the 337 MB `vision_weight.q4nx`. Exits 77 -- graded as a skip -- when it
is not resolvable, so this runs offline without failing.

    python test_gemma4_vision_weights.py [model-dir-or-repo-id]
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gemma4_vision_weights as vw

SKIP_EXIT_CODE = 77

#: (key, K, out) for one tensor of every distinct packed geometry in the file:
#: square, wide (K < out), tall (K > out), and the non-square output projection.
#: Checking all 659 would take minutes and prove nothing the four do not.
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
        # int16 rather than bfloat16: a bitwise check should not route through
        # a float comparison that could call two NaN encodings equal.
        ok = np.array_equal(stored.view(np.int16), back.view(np.int16))
        bad += not ok
        print(
            f"  {'ok  ' if ok else 'FAIL'} {tuple(stored.shape)} -> "
            f"[{K}, {out}]  {key}"
        )

    # The position table is the one tensor packed per-dimension, and the one
    # where the converter feeds a transpose -- so it is the one most likely to
    # come back with its axes swapped, which is invisible in the shape (both
    # axes would still be 10240 and 768, just the other way round).
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

    # A swapped mm_n/mm_k has to be rejected, not merely un-chosen. Without
    # this, the checks above pass for any self-consistent pair.
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
