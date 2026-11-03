#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The production hybrid's Task 2 shape: a cached system prefix, a fresh turn.

    python bench_prefix_reuse.py --image /tmp/shapes.png --backend npu

The shape being reproduced, as the production pipeline describes it: a
**135-token prompt** whose leading **61 tokens are a system prefix**, where a
second turn reuses that prefix and recomputes only the **74-token suffix**, and
"compact scheduling" brings the computed rows down to **128**.

All four numbers fall out here without tuning, which is the point of running it:

* 61 -- the system prompt below renders to exactly that.
* 74 -- the image turn is 64 soft tokens plus its template.
* 135 -- the two together.
* 128 -- `LlamaPrefill.SEQ_BUCKET` is already 128, so a 74-row suffix pads to
  128 where the full 135 pads to 256. The 2x they get from compact scheduling
  is the same 2x, from a bucket this repository already had.

What this measures is PREFILL time only. The KV handed to the decode is the
full 135-row context either way, so prefix reuse does not shrink the handoff --
their note makes the same point and it is worth not losing.

Correctness is not this script's job: `../llm_q4nx/test_prefix_reuse.py` gates
reuse against a full prefill and requires bit-identical logits and KV. This
only times it.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

sys.path.insert(0, config.text._SHARED)

import gemma4_vision  # noqa: E402
import harness  # noqa: E402
import preprocess  # noqa: E402

from gemma4_prefill import Gemma4Prefill  # noqa: E402

#: Renders to exactly 61 tokens through this checkpoint's chat template. The
#: wording is filler; the length is the point, so do not "tidy" it without
#: re-checking the count -- `bench_prefix_reuse.py --show-lengths` prints it.
SYSTEM = (
    "You are a careful visual assistant answer only from the image do not "
    "speculate about anything you cannot see be concise state colours and "
    "positions explicitly when asked about shapes never invent objects that "
    "are absent and prefer short factual sentences over elaboration in every "
    "reply you give to the user regardless of."
)

QUESTIONS = (
    "Describe the shapes and their colors and positions.",
    "Which shape is closest to the top left corner, and what colour is it?",
)


def build(tokenizer, system, question, n_soft):
    """System turn + image turn -> (ids, image_mask, ple_ids, n_system)."""
    sys_msg = [{"role": "system", "content": [{"type": "text", "text": system}]}]
    rendered_sys = tokenizer.apply_chat_template(
        sys_msg, tokenize=False, add_generation_prompt=False
    )
    n_system = len(tokenizer.encode(rendered_sys, add_special_tokens=False))

    full_msg = sys_msg + [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    rendered = tokenizer.apply_chat_template(
        full_msg, tokenize=False, add_generation_prompt=True
    )
    tok = tokenizer.image_token
    rendered = rendered.replace(
        tok, f"{tokenizer.boi_token}{tok * n_soft}{tokenizer.eoi_token}"
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    mask = [i == config.IMAGE_TOKEN_ID for i in ids]
    ple = [config.PAD_TOKEN_ID if m else i for i, m in zip(ids, mask)]
    return ids, mask, ple, n_system


def _rows(n, bucket=128):
    return ((n + bucket - 1) // bucket) * bucket


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--backend", choices=("cpu", "npu", "hetero"), default="npu")
    ap.add_argument("--ops", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--max-soft-tokens", type=int, default=70)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--show-lengths", action="store_true", help="print and exit")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    import registry

    tk_dir = args.model or harness.tokenizer_dir(None, registry.spec(config.MODEL_NAME))
    tokenizer = AutoTokenizer.from_pretrained(tk_dir)

    pixels, pos_ids, n_soft = preprocess.load_image(
        args.image,
        max_soft_tokens=args.max_soft_tokens,
        image_size=args.image_size or None,
    )
    # Turn B gets a DIFFERENT image, and that is what makes this their Task 2
    # rather than a flattering variant of it. With the same image in both
    # turns the shared prefix runs past the system prompt and through all 64
    # soft tokens -- only the question differs -- so reuse would look far
    # better than the pipeline being reproduced. A different image puts the
    # whole image turn in the suffix and pins the reused prefix at the system
    # prompt, which is what "the 61-token system KV cache is reused" means.
    from PIL import Image, ImageOps

    with Image.open(args.image) as im:
        alt = f"/tmp/{os.path.basename(args.image)}.mirrored.png"
        ImageOps.mirror(im.convert("RGB")).save(alt)
    pixels_b, pos_ids_b, n_soft_b = preprocess.load_image(
        alt,
        max_soft_tokens=args.max_soft_tokens,
        image_size=args.image_size or None,
    )

    a = build(tokenizer, SYSTEM, QUESTIONS[0], n_soft)
    b = build(tokenizer, SYSTEM, QUESTIONS[1], n_soft_b)
    n_system = a[3]
    print(
        f"[bench] system {n_system} tok | turn A {len(a[0])} tok | "
        f"turn B {len(b[0])} tok | image {n_soft} soft tokens"
    )
    print(
        f"[bench] rows: full {_rows(len(b[0]))}  "
        f"reuse {_rows(len(b[0]) - n_system)}  (SEQ_BUCKET=128)"
    )
    if args.show_lengths:
        return 0

    m = Gemma4Prefill(
        backend=args.backend,
        ops=args.ops,
        n_layers=config.N_LAYERS,
        max_seq=args.max_seq,
        model=args.model,
        expect_model=config.MODEL_NAME,
    )
    # Reusing and full passes compared on the same (host) attention.
    m.npu_attention = False
    harness.load_prefill_weights(m, args.backend)
    tower = gemma4_vision.Gemma4VisionTower(m, model=args.model)
    soft = tower.encode(pixels, pos_ids)
    soft_b = tower.encode(pixels_b, pos_ids_b)

    full_t, reuse_t, p0s = [], [], []
    for _ in range(args.rounds):
        # Turn A warms the cache. Not timed: it is the first turn of a session
        # and has nothing to reuse by definition.
        m.clear_context()
        m.prefill(a[0], image_embeds=soft, image_mask=a[1], ple_ids=a[2])

        # Turn B twice over: once reusing A's system prefix, once from nothing.
        # `_reusable_prefix` is read BEFORE the reusing call, which is the only
        # moment it still reflects turn A -- `prefill` sets `_cached_ids` to
        # turn B on the way out, and asking afterwards would report the prompt
        # matching itself.
        # The GUARDED prefix, which is what `prefill` will act on. The raw
        # id match is longer and misleading: every soft token shares one
        # placeholder id, so ids alone say turn B's image matches turn A's.
        p0s.append(m._image_safe_prefix(m._reusable_prefix(b[0]), b[1], soft_b))
        t0 = time.time()
        m.prefill(b[0], image_embeds=soft_b, image_mask=b[1], ple_ids=b[2], reuse=True)
        reuse_t.append(time.time() - t0)

        m.clear_context()
        t0 = time.time()
        m.prefill(b[0], image_embeds=soft_b, image_mask=b[1], ple_ids=b[2])
        full_t.append(time.time() - t0)

    med = lambda xs: float(np.median(xs))  # noqa: E731
    print(
        f"[bench] reused prefix {p0s[0]} rows "
        f"(the system turn is {n_system}; the image turn is the suffix)"
    )
    print(
        f"[bench] full prefill   median {med(full_t):.3f}s  {[round(t,3) for t in full_t]}"
    )
    print(
        f"[bench] reuse prefill  median {med(reuse_t):.3f}s  {[round(t,3) for t in reuse_t]}"
    )
    print(
        f"[bench] saving         {1 - med(reuse_t) / med(full_t):+.1%} of prefill time"
    )
    print("[bench] the decode still receives the FULL context KV either way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
