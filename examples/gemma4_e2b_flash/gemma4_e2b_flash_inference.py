#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""End to end with an image: Triton vision tower + prefill, mlir-air decode.

    python gemma4_e2b_flash_inference.py --image /tmp/shapes.png \
        --text "Describe the shapes and their colors and positions."

    python gemma4_e2b_flash_inference.py --image /tmp/shapes.png --prefill-only

The language half is the text example's, unchanged: the same weights, the same
35-layer Triton prefill (`../llm_q4nx/gemma4_prefill.py`), and the same fused
Q4NX decode from mlir-air's per-layer-embedding engine. What this adds is
everything before the first token -- the image is encoded by
`../llm_q4nx/gemma4_vision.py` into soft tokens, and those are scattered over
the placeholders in the prompt.

**Why an image is the better end-to-end check.** A text prompt that prefills
wrongly produces obviously broken text, which is easy but weak: many bugs do
not produce it. A broken image path produces a *confident, fluent, wrong*
description, and the correct answer is unambiguous and checkable by eye. For
the two-shape test image the model must name both shapes, both colours and both
positions; anything less is a failure even if it reads well.

This is a separate driver rather than a flag on the shared `harness.main`
because the prompt is built differently (the placeholder run has to be sized
from the image), the prefill takes three extra arguments, and none of that
should be reachable from the text examples. The decode handoff, though, is the
shared one -- `harness.generate_via_kv_arrays`, the same path
`gemma4_e2b_q4nx` takes -- so the decode is not a second implementation.

Prompt length is dominated by the image: 256 soft tokens for a 448x448 input,
against a dozen for the text around it. The decode template on disk has to be
at least that long, so:

    make compile-decode DECODE_L=512

`--prefill-only` needs no decode build and is the right first run.

`--backend gpu` runs everything on the iGPU instead: the tower is
`Gemma4GpuVisionTower`, the prefill `Gemma4GpuDecode.prefill`, and the decode
the same decoder's. It needs neither an NPU nor a decode build.
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
import harness  # noqa: E402
import preprocess  # noqa: E402
import registry  # noqa: E402

from gemma4_prefill import Gemma4Prefill  # noqa: E402

#: What `--text` asks when nothing is given. Deliberately the question the FLM
#: notes use, so a result here is comparable with one from `flm run`.
DEFAULT_TEXT = "Describe the shapes and their colors and positions."


def build_prompt(tokenizer, text, n_soft, system=None):
    """Render the chat template and expand the image placeholder.

    Returns `(ids, image_mask, ple_ids)`.

    `system`, when given, adds a system turn ahead of the image turn. That is
    the prefix a `--reuse` run caches: it is the same across turns, where the
    image turn is not.

    The template emits ONE image token per image; the processor's job -- and so
    this function's -- is to grow it into `boi + image_token * n_soft + eoi`,
    which is `Gemma4Processor.replace_image_token` verbatim. The count is the
    image's, not a constant: a different aspect ratio pools to a different
    number of soft tokens, and a run of the wrong length silently misaligns
    every token after it.

    `ple_ids` carries the pad token at the placeholder positions. See
    `gemma4_prefill.Gemma4Prefill.prefill` for why that is not the same list.
    """
    msg = []
    if system:
        msg.append({"role": "system", "content": [{"type": "text", "text": system}]})
    msg.append(
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text}],
        }
    )
    rendered = tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    )
    tok = tokenizer.image_token
    if rendered.count(tok) != 1:
        raise RuntimeError(
            f"the chat template emitted {rendered.count(tok)} image tokens, "
            "expected exactly 1 for one image"
        )
    rendered = rendered.replace(
        tok, f"{tokenizer.boi_token}{tok * n_soft}{tokenizer.eoi_token}"
    )
    # The template already emits <bos>, so this must not add a second one --
    # unlike the text example's canonical prompt, where the <bos> is prepended
    # by hand because no template ran.
    ids = tokenizer.encode(rendered, add_special_tokens=False)

    mask = [i == config.IMAGE_TOKEN_ID for i in ids]
    if sum(mask) != n_soft:
        raise RuntimeError(
            f"{sum(mask)} image placeholders survived tokenization, expected "
            f"{n_soft} -- the tokenizer may be splitting the placeholder"
        )
    ple_ids = [config.PAD_TOKEN_ID if m else i for i, m in zip(ids, mask)]
    return ids, mask, ple_ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument(
        "--backend",
        choices=("cpu", "npu", "hetero", "gpu"),
        default="npu",
        help="where the vision tower and the prefill run. gpu: both on the "
        "iGPU in Triton (Gemma4GpuVisionTower, Gemma4GpuDecode.prefill), and "
        "the decode too",
    )
    ap.add_argument(
        "--decode",
        choices=("npu", "gpu"),
        default="npu",
        help="npu: mlir-air's fused decode, all 35 layers in one dispatch. "
        "gpu: Triton on the iGPU, token by token. Both read the SAME KV slab "
        "this prefill wrote -- see ../llm_q4nx/kv_layout.py",
    )
    ap.add_argument(
        "--prefill-engine",
        choices=("triton", "air-fused"),
        default="triton",
        help="triton: this repository's prefill on --backend. air-fused: "
        "mlir-air's one-device chunked NPU prefill, writing the same KV slab; "
        "the vision tower then runs on the iGPU",
    )
    ap.add_argument("--ops", default=None, help="NPU ops: 'all' or a comma list")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--max-seq", type=int, default=512)
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
        "--system",
        default=None,
        help="a system turn ahead of the image turn. This is the prefix "
        "--reuse caches",
    )
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="run a PRIOR turn first (same --system, mirrored image, different "
        "question) and reuse its KV, recomputing only this turn's suffix. The "
        "production hybrid's Task 2 cache path",
    )
    ap.add_argument(
        "--ignore-eos",
        action="store_true",
        help="generate the full --max-tokens instead of stopping at "
        "<end_of_turn>. What a fixed-length timing comparison needs. "
        "**GPU decode only** -- mlir-air's fused decode stops on EOS "
        "unconditionally and takes no argument for it",
    )
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument(
        "--prefill-only",
        action="store_true",
        help="stop after the prefill; needs no decode build",
    )
    args = ap.parse_args(argv)
    if args.backend == "gpu":
        args.decode = "gpu"
        if args.reuse:
            raise SystemExit("--reuse is the host prefill's; not with --backend gpu")
    fused = args.prefill_engine == "air-fused"
    if fused and args.backend == "gpu":
        raise SystemExit(
            "--prefill-engine air-fused runs on the NPU; not with --backend gpu"
        )
    if fused and args.reuse:
        # The fused prefill always runs the whole prompt.
        raise SystemExit("--reuse is the Triton prefill's; not with air-fused")

    spec = registry.spec(config.MODEL_NAME)
    from transformers import AutoTokenizer

    # The GPU decode needs no mlir-air driver and no decode template on disk --
    # it is Triton on the iGPU reading the slab the prefill wrote. Skipping the
    # import also means `--decode gpu` does not fail for want of a build.
    need_air = not args.prefill_only and args.decode == "npu"
    # The decode driver is imported first where it is wanted, because
    # `tokenizer_dir` prefers the tokenizer that driver was built against.
    air = harness.air_inference_module(spec) if need_air else None
    tk_dir = args.model or harness.tokenizer_dir(air, spec)
    tokenizer = AutoTokenizer.from_pretrained(tk_dir)

    pixels, pos_ids, n_soft = preprocess.load_image(
        args.image,
        max_soft_tokens=args.max_soft_tokens,
        image_size=args.image_size or None,
    )
    ids, mask, ple_ids = build_prompt(tokenizer, args.text, n_soft, args.system)
    print(
        f"[flash] {args.image}: {pixels.shape[0]} patches -> {n_soft} soft "
        f"tokens; prompt {len(ids)} tokens"
    )
    if args.reuse and not args.system:
        # Without a system turn the two turns share only <bos> and the turn
        # opening, so there is nothing worth reusing and the run would report
        # a saving it did not make.
        raise SystemExit("--reuse needs a --system prefix to cache")
    if len(ids) > args.max_seq:
        raise SystemExit(
            f"the prompt is {len(ids)} tokens and --max-seq is {args.max_seq}. "
            "Raise it, and rebuild the decode at least that long: "
            f"make compile-decode DECODE_L={args.max_seq}"
        )

    on_gpu = args.backend == "gpu"
    prefill_cls = harness.fused_prefill_cls(config) if fused else Gemma4Prefill
    m = prefill_cls(
        # The GPU path still needs the host object, for the weights its
        # decoder uploads and for the KV slab, but never runs its forward.
        backend="cpu" if on_gpu else args.backend,
        ops=args.ops,
        n_layers=config.N_LAYERS,
        max_seq=args.max_seq,
        model=args.model,
        expect_model=config.MODEL_NAME,
    )
    # A reusing suffix runs attention on the host; the turn it reuses must too.
    m.npu_attention = not args.reuse
    t0 = time.time()
    try:
        # `npu_resident=False` under `--decode gpu`: making the projections
        # NPU-resident pads them onto the device and frees the unpadded host
        # copy, and the GPU decode reads exactly that copy. Skipping this is
        # how `harness.main` does it too, and without it the run prefills
        # correctly and then dies in `gpu_kernels.gemv` on a `ResidentWeight`
        # that has no `.stride`.
        harness.load_prefill_weights(
            m,
            "cpu" if on_gpu or fused else args.backend,
            npu_resident=args.decode != "gpu",
        )
    except Exception as e:  # noqa: BLE001
        raise harness.ExampleUnavailable(f"cannot load the q4nx weights: {e}") from e
    if on_gpu:
        from gemma4_prefill import Gemma4GpuDecode

        # Cached where `harness.generate_on_gpu` looks, so the prefill and the
        # decode are one decoder: one copy of the weights, one slab.
        dec = m._gpu_decoder = Gemma4GpuDecode(m, max_L=m.max_seq)
        # The host never runs a projection on this path, and its bf16 copies
        # are most of the process's memory.
        for w in m._w:
            for k in [
                k for k, v in w.items() if torch.is_tensor(v) and v.numel() > 1 << 20
            ]:
                del w[k]
        tower = gemma4_vision.Gemma4GpuVisionTower(model=args.model)
    elif fused:
        tower = gemma4_vision.Gemma4GpuVisionTower(model=args.model)
    else:
        tower = gemma4_vision.Gemma4VisionTower(m, model=args.model)
    print(f"[flash] weights in {time.time() - t0:.1f}s (vision {tower.fingerprint})")

    t0 = time.time()
    soft = tower.encode(pixels, pos_ids)
    if on_gpu or fused:
        torch.cuda.synchronize()
    t_vision = time.time() - t0

    p0 = 0
    if args.reuse:
        # The PRIOR turn: the same system prefix, a different picture, a
        # different question. A different picture matters -- with the same one
        # the shared run would extend through all 64 soft tokens and the
        # saving would flatter itself. This is untimed: it is the earlier turn
        # of a session, and by definition had nothing to reuse.
        from PIL import Image, ImageOps

        alt = f"/tmp/{os.path.basename(args.image)}.prior.png"
        with Image.open(args.image) as im:
            ImageOps.mirror(im.convert("RGB")).save(alt)
        px_p, pid_p, ns_p = preprocess.load_image(
            alt,
            max_soft_tokens=args.max_soft_tokens,
            image_size=args.image_size or None,
        )
        pi, mi, pl = build_prompt(
            tokenizer, "What is in this picture?", ns_p, args.system
        )
        t_prior = time.time()
        m.prefill(pi, image_embeds=tower.encode(px_p, pid_p), image_mask=mi, ple_ids=pl)
        print(
            f"[flash] prior turn {len(pi)} tokens prefilled in "
            f"{time.time() - t_prior:.2f}s (untimed, warms the cache)"
        )
        p0 = m._image_safe_prefix(m._reusable_prefix(ids), mask, soft)

    t0 = time.time()
    if on_gpu:
        logits = dec.prefill(ids, image_embeds=soft, image_mask=mask, ple_ids=ple_ids)
    else:
        logits = m.prefill(
            ids,
            image_embeds=soft,
            image_mask=mask,
            ple_ids=ple_ids,
            reuse=args.reuse,
        )
    t_prefill = time.time() - t0
    ttft = t_vision + t_prefill
    if args.reuse:
        print(
            f"[flash] reused {p0} of {len(ids)} rows, recomputed {len(ids) - p0} "
            f"({((len(ids) - p0) + 127) // 128 * 128} rows after bucketing, "
            f"against {(len(ids) + 127) // 128 * 128} for a full prefill)"
        )
    first = int(torch.argmax(logits))
    print(
        f"[flash] vision {t_vision:.2f}s + prefill {t_prefill:.2f}s = "
        f"{ttft:.2f}s; first token {first} "
        f"{tokenizer.decode([first])!r}"
    )

    if args.prefill_only:
        return 0

    # `--max-tokens` is the TOTAL output token count, the prefill's own first
    # token included -- the production pipeline counts it that way ("16 and 64
    # output tokens" = 1 from the NPU plus 15 and 63 from the GPU). The two
    # decode paths disagree natively: `Gemma4GpuDecode.generate` returns
    # `n_tokens` with `first` among them, while mlir-air's returns `first` PLUS
    # `max_tokens`. Normalized here so the flag means one thing.
    want = args.max_tokens
    args.max_tokens = want if args.decode == "gpu" else want - 1

    if args.ignore_eos:
        if args.decode != "gpu":
            raise SystemExit(
                "--ignore-eos is GPU-decode only: mlir-air's fused decode stops "
                "on EOS unconditionally and its generate() takes no argument "
                "for it (see harness.generate_via_kv_arrays). Use --decode gpu."
            )
        # `generate_on_gpu` reads the stop set off the config module, so
        # emptying it there is the only channel. Done on the module rather than
        # on a copy because that is what it reads; the process ends with the
        # run, so nothing outlives it.
        config.EOS_IDS = ()

    t0 = time.time()
    if args.decode == "gpu":
        # `generate_on_gpu` builds `Gemma4GpuDecode` from the prefiller and
        # reads the KV in place -- an alias of the very pages the prefill wrote
        # where the slab is shared, one copy where it is not. It needs
        # `cfg.EOS_IDS`, which this config re-exports from the text example.
        out = harness.generate_on_gpu(config, args, m, first, ids)
    else:
        out = harness.generate_via_kv_arrays(
            air, spec, config, args, ids, m, first, ttft
        )
    t_dec = time.time() - t0
    # The NPU path's own throughput line comes from mlir-air; the GPU path has
    # none, so it is printed here for both. Measured around `generate` either
    # way, so the two are the same quantity.
    if out:
        print(
            f"[flash] decode({args.decode}): {len(out)} tokens in "
            f"{t_dec:.2f}s {len(out) / t_dec:.2f} tok/s"
        )
    print(f"[flash] {tokenizer.decode(out)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except harness.ExampleUnavailable as e:
        print(f"[flash] unavailable: {e}")
        raise SystemExit(harness.SKIP_EXIT_CODE) from e
