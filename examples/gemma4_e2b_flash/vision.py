#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gate the vision tower against `transformers`, stage by stage.

    python vision.py --image /tmp/shapes.png
    python vision.py --image /tmp/shapes.png --backend npu --ops all

This is the correctness gate for everything in
`../llm_q4nx/gemma4_vision.py`. It runs our tower and upstream's
`Gemma4VisionModel` on the **same de-tiled weights** and prints a cosine and a
max-abs difference for every stage: the patch embedder, each of the 16 encoder
layers, the pooler, and the projection into language space.

Per stage rather than end to end, because the failures this is guarding against
are not distinguishable at the output. A transposed projection, a RoPE split
applied to the wrong lanes, a dropped clip bound and a pooler keyed off the
wrong axis all produce soft tokens of the right shape and a plausible caption.
A stage-by-stage diff says *which*, and the first stage that drops is the one
to read.

What it does not cover is the de-tiling itself -- both sides here read the
weights through the same loader. `../llm_q4nx/test_gemma4_vision_weights.py`
covers that, against the bytes in the file.

Tolerances: `--backend cpu` runs both sides in f32 and should sit at cosine
1.000000 everywhere; our `_matmul` rounds its inputs to bf16 to model what the
device does, so a few units in the last place of bf16 is the floor, not zero.
Under `--backend npu` the deltas widen to the NPU GEMM's own rounding. Either
way the thing to watch is a stage that falls off a cliff, not the last digit.

No image? Any PNG will do. The one the FLM notes use is two shapes on a light
background, which is also the best end-to-end prompt:

    from PIL import Image, ImageDraw
    im = Image.new('RGB', (448, 448), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.rectangle([60, 60, 200, 200], fill=(220, 30, 30))
    d.ellipse([250, 250, 400, 400], fill=(30, 80, 220))
    im.save('/tmp/shapes.png')
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

sys.path.insert(0, config.text._SHARED)

import gemma4_vision  # noqa: E402
import hf_reference  # noqa: E402
import preprocess  # noqa: E402

from gemma4_prefill import Gemma4Prefill  # noqa: E402

#: Cosine below this on any stage fails the gate. Generous on purpose: this is
#: a cliff detector, and a real error here is 0.9 or 0.3, not 0.99998.
COS_MIN = 0.999


def _cos(a, b):
    a = a.reshape(-1).to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    n = a.norm() * b.norm()
    return float(a.dot(b) / n) if n > 0 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--backend", choices=("cpu", "npu", "hetero"), default="cpu")
    ap.add_argument(
        "--ops",
        default=None,
        help="NPU ops to enable: 'all', or a comma list of matmul,rms_norm,geglu",
    )
    ap.add_argument("--model", default=None, help="model dir or HF repo id")
    ap.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="square the input to NxN before the processor. 256 is what "
        "the production hybrid pipeline normalizes to; 0 keeps the "
        "image native",
    )
    ap.add_argument(
        "--max-soft-tokens",
        type=int,
        default=None,
        help="image resolution budget: 70, 140, 280 (default), 560, 1120. "
        "FastFlowLM ships 70 -- see preprocess.processor()",
    )
    ap.add_argument(
        "--layers",
        type=int,
        default=config.VISION_N_LAYERS,
        help="stop after this many encoder layers (bisection)",
    )
    args = ap.parse_args(argv)

    pixels, pos_ids, n_soft = preprocess.load_image(
        args.image,
        max_soft_tokens=args.max_soft_tokens,
        image_size=args.image_size or None,
    )
    print(
        f"[vision] {args.image}: {pixels.shape[0]} patches "
        f"({int((pos_ids[:, 0] >= 0).sum())} real) -> {n_soft} soft tokens"
    )

    w = config.load_vision(args.model)
    print(f"[vision] weights {w['fingerprint']}")

    # The tower borrows its operators from a Gemma4Prefill, which is what makes
    # `--backend`/`--ops` mean the same thing here as on the text side. Its
    # weights are never loaded: nothing below touches the language model.
    host = Gemma4Prefill(backend=args.backend, ops=args.ops, n_layers=1, max_seq=64)
    tower = gemma4_vision.Gemma4VisionTower(host, weights=w)
    if args.layers != config.VISION_N_LAYERS:
        gemma4_vision.N_LAYERS = args.layers
        print(f"[vision] stopping after {args.layers} layers")

    ours, ref = {}, {}
    t0 = time.time()
    got = tower.encode(pixels, pos_ids, trace=ours)
    t_ours = time.time() - t0

    t0 = time.time()
    want = hf_reference.encode(pixels, pos_ids, weights=w, trace=ref)
    t_ref = time.time() - t0

    print(
        f"[vision] ours {t_ours:.1f}s (backend={args.backend} "
        f"ops={sorted(host.enabled)}), reference {t_ref:.1f}s"
    )
    if tuple(got.shape) != tuple(want.shape):
        print(f"[vision] FAIL shape {tuple(got.shape)} != {tuple(want.shape)}")
        return 1

    # Per-patch stages are scored on the REAL patches only, and the padding
    # rows are reported beside them rather than folded in.
    #
    # Not a tolerance being relaxed to make a number pass. The image processor
    # pads 2520 rows out of 2304 real ones, and what those 216 rows hold is
    # undefined on both sides: upstream masks them out of attention on both
    # axes, this tower masks only the key axis (`gemma4_vision._attention`
    # says why -- masking a query row too leaves it an all -inf softmax and
    # NaNs that spread). So they drift, by design, and they are zeroed before
    # the pooler reads anything. Scoring them would report a disagreement about
    # values that cannot reach an output, and would hide a real one behind it:
    # at layer 15 the padding rows sit at cosine 0.95 while every real patch is
    # at 0.999993.
    #
    # `pool` and `embed_vision` are scored whole -- by then the padding is gone.
    valid = torch.as_tensor(~(pos_ids == -1).all(axis=-1), dtype=torch.bool)
    stages = ["patch_embed"] + [f"layer{L}" for L in range(args.layers)]
    if args.layers == config.VISION_N_LAYERS:
        stages += ["pool", "embed_vision"]
    else:
        # The reference always runs all 16, so a truncated run's pooler is
        # reading a different tensor and the two are not comparable.
        print("[vision] truncated: skipping pool/embed_vision")

    worst, bad = 1.0, []
    for s in stages:
        if s not in ours or s not in ref:
            continue
        a, b = ours[s], ref[s].to(torch.float32)
        per_patch = a.shape[0] == valid.shape[0]
        av, bv = (a[valid], b[valid]) if per_patch else (a, b)
        c = _cos(av, bv)
        d = float((av - bv).abs().max())
        pad = f"  pad {_cos(a[~valid], b[~valid]):.6f}" if per_patch else ""
        worst = min(worst, c)
        if c < COS_MIN:
            bad.append(s)
        flag = "" if c >= COS_MIN else "  <-- "
        print(f"  {s:14s} cos {c:.6f}  maxabs {d:.5f}{pad}{flag}")

    print(f"[vision] worst cosine {worst:.6f} over {len(stages)} stages (real patches)")
    if bad:
        print(f"[vision] FAIL -- first bad stage: {bad[0]}")
        return 1
    print("[vision] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
