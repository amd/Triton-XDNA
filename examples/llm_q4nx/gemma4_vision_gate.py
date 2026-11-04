#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gate a Gemma4 vision tower against transformers' own, on the same weights.

    python gemma4_vision_gate.py --image /tmp/shapes.png --tower host
    python gemma4_vision_gate.py --image /tmp/shapes.png --tower air --repeat 3

`--tower`:

* `host` -- `gemma4_vision.Gemma4VisionTower`, its operators placed by
  `--backend`/`--ops` as the text prefill's are. Scored **stage by stage**, so
  a failure names the stage that broke.
* `air` -- mlir-air's NPU tower (`gemma4_vision_air`). Scored on its soft
  tokens; mlir-air's own gate is the same comparison.
* `triton` -- mlir-air's tower pipeline with its GEMMs on this repository's
  Triton device (`gemma4_vision_device`). Scored on its soft tokens.
* `igpu` -- `gemma4_vision.Gemma4GpuVisionTower`, every stage a Triton kernel
  on the iGPU. Scored on its soft tokens.

Per-patch stages are scored on the real patches only. The padding rows are
undefined on both sides -- upstream masks them out of attention on both axes,
the towers here only on the key axis -- and are zeroed before the pooler reads
anything, so they cannot reach an output.

`--model` takes the text bundle's repo or directory: the vision weights ship
beside it, so `FastFlowLM/Gemma4-E4B-IT-NPU2` gates E4B's tower (the same
tower, at E4B's projection width).

The host tower in f32 sits at cosine ~0.99999 per stage; the device towers
add their bf16 GEMMs and land at ~0.9995 on the soft tokens. The floors below
are cliff detectors: a real error is 0.9 or 0.3, not 0.999.
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gemma4_image  # noqa: E402
import gemma4_vision  # noqa: E402
import gemma4_vision_hf  # noqa: E402
import gemma4_vision_q4nx as vw  # noqa: E402

#: Per stage, for the host tower.
STAGE_COS_MIN = 0.999
#: On the soft tokens, for the device towers -- mlir-air's own floor.
SOFT_COS_MIN = 0.99


def _cos(a, b):
    a = a.reshape(-1).to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    n = a.norm() * b.norm()
    return float(a.dot(b) / n) if n > 0 else float("nan")


def make_tower(kind, weights, backend="cpu", ops=None):
    """A tower by `--tower` name, on loaded `weights`."""
    if kind == "host":
        from gemma4_prefill import Gemma4Prefill

        # The tower borrows its operators from a Gemma4Prefill, so that
        # `--backend`/`--ops` mean the same thing as on the text side. Its
        # language weights are never loaded.
        host = Gemma4Prefill(backend=backend, ops=ops, n_layers=1, max_seq=64)
        return gemma4_vision.Gemma4VisionTower(host, weights=weights)
    if kind == "air":
        import gemma4_vision_air

        return gemma4_vision_air.AirVisionTower(weights=weights)
    if kind == "triton":
        import gemma4_vision_device

        return gemma4_vision_device.tower(weights=weights)
    if kind == "igpu":
        return gemma4_vision.Gemma4GpuVisionTower(weights=weights)
    raise ValueError(kind)


def _sync(kind):
    if kind == "igpu":
        torch.cuda.synchronize()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True)
    ap.add_argument(
        "--tower", choices=("host", "air", "triton", "igpu"), default="host"
    )
    ap.add_argument("--backend", choices=("cpu", "npu", "hetero"), default="cpu")
    ap.add_argument("--ops", default=None, help="host tower: NPU ops")
    ap.add_argument("--model", default=None, help="text bundle repo or dir")
    ap.add_argument("--image-size", type=int, default=0, help="square to NxN first")
    ap.add_argument(
        "--max-soft-tokens",
        type=int,
        default=None,
        help=f"resolution budget: {', '.join(map(str, gemma4_image.SOFT_TOKEN_BUDGETS))}",
    )
    ap.add_argument("--repeat", type=int, default=1, help="timed encodes, warm")
    args = ap.parse_args(argv)

    pixels, pos_ids, n_soft = gemma4_image.load_image(
        args.image,
        max_soft_tokens=args.max_soft_tokens,
        image_size=args.image_size or None,
    )
    valid = torch.as_tensor(~(pos_ids == -1).all(axis=-1), dtype=torch.bool)
    print(
        f"[vision] {args.image}: {pixels.shape[0]} patches "
        f"({int(valid.sum())} real) -> {n_soft} soft tokens"
    )
    w = vw.load_vision(args.model)
    print(f"[vision] weights {w['fingerprint']}, out {w['embedding_proj'].shape[1]}")
    tower = make_tower(args.tower, w, args.backend, args.ops)

    ours = {} if args.tower == "host" else None
    t0 = time.perf_counter()
    got = (
        tower.encode(pixels, pos_ids, trace=ours)
        if ours is not None
        else (tower.encode(pixels, pos_ids))
    )
    _sync(args.tower)
    print(f"[vision] first encode {time.perf_counter() - t0:.2f}s (incl. builds)")
    times = []
    for _ in range(args.repeat - 1 if args.repeat > 1 else 0):
        t0 = time.perf_counter()
        got = tower.encode(pixels, pos_ids)
        _sync(args.tower)
        times.append(time.perf_counter() - t0)
    if times:
        print(f"[vision] encode {statistics.median(times) * 1e3:.1f} ms (median, warm)")
        if args.tower in ("air", "triton"):
            t = {k: f"{v / args.repeat * 1e3:.1f}" for k, v in tower.timings().items()}
            print(f"[vision] device ms/encode {t}")

    ref = {} if ours is not None else None
    want = gemma4_vision_hf.encode(pixels, pos_ids, weights=w, trace=ref)
    got = got.to("cpu", torch.float32)
    if tuple(got.shape) != tuple(want.shape):
        print(f"[vision] FAIL shape {tuple(got.shape)} != {tuple(want.shape)}")
        return 1

    if ours is None:
        c = _cos(got, want)
        print(f"[vision] soft tokens vs HF: cosine {c:.6f} (floor {SOFT_COS_MIN})")
        if c < SOFT_COS_MIN:
            print("[vision] FAIL")
            return 1
        print("[vision] PASS")
        return 0

    stages = ["patch_embed"] + [f"layer{L}" for L in range(vw.N_LAYERS)]
    stages += ["pool", "embed_vision"]
    worst, bad = 1.0, []
    for s in stages:
        if s not in ours or s not in ref:
            continue
        a, b = ours[s].to("cpu", torch.float32), ref[s].to(torch.float32)
        per_patch = a.shape[0] == valid.shape[0]
        av, bv = (a[valid], b[valid]) if per_patch else (a, b)
        c = _cos(av, bv)
        d = float((av - bv).abs().max())
        worst = min(worst, c)
        if c < STAGE_COS_MIN:
            bad.append(s)
        flag = "" if c >= STAGE_COS_MIN else "  <-- "
        print(f"  {s:14s} cos {c:.6f}  maxabs {d:.5f}{flag}")
    print(f"[vision] worst cosine {worst:.6f} over {len(stages)} stages")
    if bad:
        print(f"[vision] FAIL -- first bad stage: {bad[0]}")
        return 1
    print("[vision] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
