# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""An image turn on Gemma4: vision tower -> prefill -> decode, each placed on
the NPU or the iGPU.

    python gemma4_e2b_flash_inference.py --image /tmp/shapes.png \\
        --text "Describe the shapes and their colors and positions." \\
        --vision igpu --prefill-engine air-fused --decode npu

The three stages and where each can run:

    --vision          air      mlir-air's NPU tower (gemma4_vision_air)
                      triton   the same tower pipeline, its GEMMs and attention
                               Triton kernels on the NPU (gemma4_vision_device)
                      igpu     the tower as Triton kernels on the iGPU
                      host     the tower on the host, its GEMMs placed by
                               --backend/--ops like the Triton prefill's
                      auto     igpu where there is one, else air
    --prefill-engine  air-fused     mlir-air's fused NPU prefill
                      triton-fused  the same device, its ops from Triton (E2B)
                      triton        this repository's prefill, per --backend
                      igpu          the prefill as Triton kernels on the iGPU
    --decode          npu      mlir-air's fused PLE decode
                      gpu      the decode as Triton kernels on the iGPU

The soft tokens the tower produces replace the image placeholders' embeddings
in the prompt. Whatever the prefill, it writes its K/V straight into ONE slab
in the layout both decoders read (`kv_layout`), in pages the NPU (an XRT BO)
and the iGPU (a HIP mapping of the same pages) both address: the decode reads
the cache where the prefill left it, with no copy in between.

A model is this module plus its `config` (`examples/gemma4_*_flash/`).
"""

import argparse
import os
import statistics
import sys
import time

import torch

import gemma4_image
import harness
import registry

DEFAULT_TEXT = "Describe the shapes and their colors and positions."

VISION = ("auto", "air", "triton", "igpu", "host")
ENGINES = ("air-fused", "triton-fused", "triton", "igpu")


def build_parser(doc):
    ap = argparse.ArgumentParser(
        description=(doc or __doc__).splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 2)[2].split("\n\nThe soft tokens")[0],
    )
    ap.add_argument("--image", help="default: the two-shape test image")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--system", default=None, help="a system turn before the image")
    ap.add_argument("--vision", choices=VISION, default="auto")
    ap.add_argument("--prefill-engine", choices=ENGINES, default="air-fused")
    ap.add_argument("--decode", choices=("npu", "gpu"), default="npu")
    ap.add_argument(
        "--backend",
        choices=("cpu", "npu", "hetero"),
        default=None,
        help="--prefill-engine triton and --vision host: where their ops run",
    )
    ap.add_argument("--ops", default=None, help="NPU ops: 'all' or a comma list")
    ap.add_argument(
        "--max-soft-tokens",
        type=int,
        default=None,
        help="the image resolution budget: "
        + ", ".join(map(str, gemma4_image.SOFT_TOKEN_BUDGETS))
        + f" (default {gemma4_image.MAX_SOFT_TOKENS}; FastFlowLM ships 70)",
    )
    ap.add_argument("--image-size", type=int, default=0, help="square to NxN first")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=48,
        help="output tokens, the prefill's first token included",
    )
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--model", default=None, help="model dir or HF repo id")
    ap.add_argument(
        "--prefill-only", action="store_true", help="no decode build needed"
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="time the vision + prefill this many times after a warm-up turn",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="fail unless the description names both shapes, both colors and a "
        "position (the shapes image)",
    )
    return ap


def _resolve_vision(kind):
    if kind != "auto":
        return kind
    return "igpu" if torch.cuda.is_available() else "air"


def _check(args, cfg):
    """Refuse the combinations that cannot run, before minutes of loading."""
    gpu = (
        args.vision == "igpu"
        or args.prefill_engine == "igpu"
        or args.decode == "gpu"
        or args.backend == "hetero"
    )
    if gpu and not torch.cuda.is_available():
        raise harness.ExampleUnavailable(
            "an iGPU stage was asked for and torch has no GPU: install a ROCm "
            "build of torch, or place every stage on the NPU"
        )
    if os.environ.get("AMD_TRITON_NPU_RUNTIME") == "hsa":
        raise SystemExit("the Gemma4 decode runs under XRT only; unset RUNTIME=hsa")
    if args.prefill_engine == "triton-fused" and cfg.N_KV_HEADS != 1:
        raise SystemExit("--prefill-engine triton-fused is built for gemma4-e2b only")
    triton_only = args.prefill_engine != "triton" and args.vision != "host"
    if triton_only and (args.backend or args.ops):
        raise SystemExit(
            "--backend/--ops place the Triton prefill's and the host tower's "
            "ops; neither runs here"
        )


def _prefill_cls(args, cfg):
    from gemma4_prefill import Gemma4IgpuPrefill, Gemma4Prefill

    if args.prefill_engine in ("air-fused", "triton-fused"):
        config = harness.fused_prefill_config(cfg, args.max_seq)
        return harness.fused_prefill_cls(
            cfg,
            Gemma4Prefill,
            config,
            triton=args.prefill_engine == "triton-fused",
        )
    if args.prefill_engine == "igpu":
        return Gemma4IgpuPrefill
    return Gemma4Prefill


def _tower(args, weights):
    import gemma4_vision

    if args.vision == "air":
        import gemma4_vision_air

        return gemma4_vision_air.AirVisionTower(weights=weights)
    if args.vision == "triton":
        import gemma4_vision_device

        return gemma4_vision_device.tower(weights=weights)
    if args.vision == "igpu":
        return gemma4_vision.Gemma4GpuVisionTower(weights=weights)
    from gemma4_prefill import Gemma4Prefill

    # The host tower borrows its operators' placement from a Gemma4Prefill
    # whose language weights are never loaded.
    host = Gemma4Prefill(
        backend=args.backend or "cpu", ops=args.ops, n_layers=1, max_seq=64
    )
    return gemma4_vision.Gemma4VisionTower(host, weights=weights)


def _sync():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def main(cfg, doc=None, argv=None):
    """One image turn for the model `cfg` describes."""
    import gemma4_vision_q4nx as vw

    args = build_parser(doc).parse_args(argv)
    # Opened before torch starts ROCm, which would otherwise take the NPU.
    harness.hold_npu()
    args.vision = _resolve_vision(args.vision)
    _check(args, cfg)
    spec = registry.spec(cfg.MODEL_NAME)
    model = args.model or cfg.MODEL_DEFAULT

    air = None
    if not args.prefill_only and args.decode == "npu":
        air = harness.air_inference_module(spec)
        harness.decode_build.check_stamp(spec)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model or harness.tokenizer_dir(air, spec)
    )

    image = args.image or gemma4_image.shapes_image("/tmp/shapes.png")
    pixels, pos_ids, n_soft = gemma4_image.load_image(
        image, max_soft_tokens=args.max_soft_tokens, image_size=args.image_size or None
    )
    ids, mask, ple_ids = gemma4_image.build_prompt(
        tokenizer, args.text, n_soft, args.system
    )
    print(
        f"[flash] {cfg.MODEL_NAME}: {image}, {pixels.shape[0]} patches -> "
        f"{n_soft} soft tokens; prompt {len(ids)} tokens",
        flush=True,
    )
    if len(ids) > args.max_seq:
        raise SystemExit(
            f"the prompt is {len(ids)} tokens and --max-seq is {args.max_seq}; "
            f"raise both it and the decode build (make compile-decode "
            f"DECODE_L={args.max_seq})"
        )

    t0 = time.time()
    prefill_cls = _prefill_cls(args, cfg)
    if args.prefill_engine == "triton":
        harness.check_host_memory(spec)
    m = harness.make_prefiller(
        prefill_cls,
        cfg,
        args.backend or "npu",
        args.ops,
        args.max_seq,
        args.model,
        npu_resident=args.decode != "gpu",
    )
    tower = _tower(args, vw.load_vision(model))
    print(
        f"[flash] loaded in {time.time() - t0:.1f}s: vision={args.vision} "
        f"prefill={args.prefill_engine} decode={args.decode}",
        flush=True,
    )

    def turn():
        t0 = time.perf_counter()
        soft = tower.encode(pixels, pos_ids)
        _sync()
        t1 = time.perf_counter()
        logits = m.prefill(ids, image_embeds=soft, image_mask=mask, ple_ids=ple_ids)
        _sync()
        return logits, t1 - t0, time.perf_counter() - t1

    logits, t_vis, t_pre = turn()
    if args.repeat > 1:
        times = [turn()[1:] for _ in range(args.repeat)]
        t_vis = statistics.median(t[0] for t in times)
        t_pre = statistics.median(t[1] for t in times)
    first = int(torch.argmax(logits))
    print(
        f"[flash] vision {t_vis * 1e3:.0f} ms + prefill {t_pre * 1e3:.0f} ms = "
        f"TTFT {(t_vis + t_pre) * 1e3:.0f} ms"
        + (f" (median of {args.repeat})" if args.repeat > 1 else "")
        + f"; first token {first} {tokenizer.decode([first])!r}",
        flush=True,
    )
    if args.prefill_only:
        return 0

    # The tower's NPU contexts are the decode's to use now.
    if hasattr(tower, "release"):
        tower.release()

    # `--max-tokens` counts the prefill's first token. Both decodes return it
    # first; the iGPU's returns `max_tokens` in all, mlir-air's `max_tokens`
    # more, so the latter is asked for one fewer.
    want = args.max_tokens
    t0 = time.perf_counter()
    if args.decode == "gpu":
        out = harness.generate_on_gpu(cfg, args, m, first, ids)
    else:
        args.max_tokens, args.greedy = want - 1, True
        out = harness.generate_via_kv_arrays(
            air, spec, cfg, args, ids, m, first, t_vis + t_pre
        )
    t_dec = time.perf_counter() - t0
    eos = set(cfg.EOS_IDS)
    text_ids = [t for t in out if t not in eos]
    if args.decode == "gpu":
        n = max(len(out) - 1, 1)
        rate = f"{n / t_dec:.2f} tok/s"
    else:
        # mlir-air builds its decoder inside `generate`, weights and all, and
        # prints the decode's own rate above; this wall time includes that.
        rate = "including the decoder's setup"
    print(
        f"[flash] decode({args.decode}): {len(out)} tokens in {t_dec:.2f}s, {rate}",
        flush=True,
    )
    text = tokenizer.decode(text_ids)
    print(f"[flash] {text}", flush=True)
    if args.gate:
        missing = gemma4_image.shapes_missing(text)
        if missing:
            print(f"[flash] FAIL: the description does not mention {missing}")
            return 1
        print("[flash] PASS")
    return 0


def run(cfg, doc=None, argv=None):
    """`main`, with a missing environment reported as a skip (exit 77)."""
    try:
        return main(cfg, doc, argv)
    except harness.ExampleUnavailable as e:
        print(f"[flash] unavailable: {e}")
        return harness.SKIP_EXIT_CODE


if __name__ == "__main__":
    sys.exit("run an example's driver, e.g. ../gemma4_e2b_flash/")
