#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
GPT-2 end-to-end inference using Triton kernels on AMD iGPU and NPU.

Config-driven across all four GPT-2 sizes via --model
{gpt2, gpt2-medium, gpt2-large, gpt2-xl}.

Loads pretrained weights from HuggingFace, runs a forward pass (prefill),
and compares output logits against the HuggingFace reference implementation.

Supports autoregressive generation with KV cache via --max-tokens.

Usage:
    # GPU path (iGPU via ROCm/Triton) — single forward pass
    python gpt2_inference.py --backend gpu

    # Larger variant (gpt2-xl, 1.5B)
    python gpt2_inference.py --model gpt2-xl --backend hetero-fast --max-tokens 20

    # GPU path — generate 20 tokens with KV cache
    python gpt2_inference.py --backend gpu --max-tokens 20

    # Interactive mode — type prompts, get completions
    python gpt2_inference.py --backend gpu --interactive
    python gpt2_inference.py --backend gpu --interactive --max-tokens 50

    # NPU path (requires XRT + env setup)
    python gpt2_inference.py --backend npu

    # Hetero path (attention on iGPU, MLP/LN/add on NPU)
    python gpt2_inference.py --backend hetero

    # Reference only (HuggingFace PyTorch)
    python gpt2_inference.py --backend reference
"""

import argparse
import logging
import sys
import os
import time

import torch

# Add parent dir to path for benchmark imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


def load_hf_model(hf_name="gpt2"):
    """Load HuggingFace GPT-2 model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading HuggingFace model: {hf_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_name)
    hf_model.eval()
    logger.info("HuggingFace model loaded.")
    return hf_model, tokenizer


def run_reference(hf_model, tokenizer, prompt):
    """Run HuggingFace reference forward pass."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = hf_model(input_ids)
        ref_logits = outputs.logits  # (1, seq_len, 50257)

    return input_ids, ref_logits


def run_triton_model(state_dict, input_ids, backend, profile=False, config=None):
    """Run our Triton-based GPT-2 model."""
    from model import GPT2Model

    # Select backend driver
    # In hetero mode, leave default GPU driver active — per-kernel npu_driver_scope()
    # handles NPU switching inside each kernel wrapper.
    if backend == "npu":
        import benchmark

        benchmark.select_npu_backend()
    elif backend in ("gpu", "hetero", "hetero-fast"):
        import benchmark

        benchmark.select_gpu_backend()

    model = GPT2Model(state_dict, backend=backend, config=config)
    model.timer.enabled = profile

    logger.info(f"Running Triton GPT-2 forward pass (backend={backend})...")
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, _ = model.forward(input_ids)
    t1 = time.perf_counter()
    logger.info(f"Forward pass completed in {t1 - t0:.3f}s")

    if profile:
        print_profile(
            model.timer, label=f"Op Timing Profile (prefill, backend={backend})"
        )

    return logits


def compare_logits(ref_logits, triton_logits, tokenizer, input_ids):
    """Compare reference and Triton model logits."""
    ref = ref_logits.to(torch.float32)
    tri = triton_logits.to(torch.float32)

    # Metrics
    max_diff = torch.max(torch.abs(ref - tri)).item()
    mean_diff = torch.mean(torch.abs(ref - tri)).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref.reshape(-1).unsqueeze(0),
        tri.reshape(-1).unsqueeze(0),
    ).item()

    print("\n" + "=" * 60)
    print("LOGITS COMPARISON")
    print("=" * 60)
    print(f"  Max absolute difference:  {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Cosine similarity:        {cos_sim:.6f}")

    # Top-k token agreement at last position
    last_ref = ref[0, -1]  # (50257,)
    last_tri = tri[0, -1]  # (50257,)

    k = 10
    ref_topk = torch.topk(last_ref, k).indices.tolist()
    tri_topk = torch.topk(last_tri, k).indices.tolist()

    ref_tokens = [tokenizer.decode([t]) for t in ref_topk]
    tri_tokens = [tokenizer.decode([t]) for t in tri_topk]

    print(f"\n  Top-{k} next-token predictions (last position):")
    print(f"  {'Rank':<6} {'Reference':<20} {'Triton':<20} {'Match'}")
    print(f"  {'----':<6} {'---------':<20} {'------':<20} {'-----'}")
    for i in range(k):
        match = "Y" if ref_topk[i] == tri_topk[i] else " "
        print(f"  {i+1:<6} {ref_tokens[i]:<20} {tri_tokens[i]:<20} {match}")

    top1_match = ref_topk[0] == tri_topk[0]
    top5_overlap = len(set(ref_topk[:5]) & set(tri_topk[:5]))
    print(f"\n  Top-1 match: {'YES' if top1_match else 'NO'}")
    print(f"  Top-5 overlap: {top5_overlap}/5")
    print("=" * 60)

    return max_diff, mean_diff, top1_match


def print_generation(logits, tokenizer, input_ids, prompt):
    """Print the predicted next token for a qualitative check."""
    last_logits = logits[0, -1]  # (vocab_size,)
    next_token_id = torch.argmax(last_logits).item()
    next_token = tokenizer.decode([next_token_id])
    print(f'\n  Prompt: "{prompt}"')
    print(f'  Predicted next token: "{next_token}"')
    print(f'  Full: "{prompt}{next_token}"')


def run_generation(hf_model, tokenizer, input_ids, args):
    """Run autoregressive generation with KV cache and report metrics."""
    from model import GPT2Model

    # Select backend driver
    # In hetero mode, leave default GPU driver active — per-kernel npu_driver_scope()
    # handles NPU switching inside each kernel wrapper.
    if args.backend == "npu":
        import benchmark

        benchmark.select_npu_backend()
    elif args.backend in ("gpu", "hetero", "hetero-fast"):
        import benchmark

        benchmark.select_gpu_backend()

    state_dict = hf_model.state_dict()
    model = GPT2Model(
        state_dict, backend=args.backend, config=getattr(args, "config", None)
    )
    model.timer.enabled = getattr(args, "profile", False)

    # Triton generation
    logger.info(f"Generating {args.max_tokens} tokens (backend={args.backend})...")
    generated_ids, timing = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        progress_callback=make_progress_bar(),
    )
    sys.stderr.write("\n")
    sys.stderr.flush()

    generated_text = tokenizer.decode(generated_ids)
    print(f"\n--- Triton ({args.backend.upper()}) Generation ---")
    print(f"  {args.prompt}{generated_text}")

    # Metrics
    ttft = timing["prefill_ms"]
    decode_times = timing["decode_times_ms"]
    num_decode = len(decode_times)
    tpot = sum(decode_times) / num_decode if num_decode > 0 else 0
    total_time_s = (ttft + sum(decode_times)) / 1000
    total_tokens = 1 + num_decode  # 1 from prefill + decode steps
    tps = total_tokens / total_time_s if total_time_s > 0 else 0

    print(f"\n  Performance Metrics:")
    print(f"    Prompt tokens:  {input_ids.shape[1]}")
    print(f"    Output tokens:  {total_tokens}")
    print(f"    TTFT:           {ttft:.1f} ms")
    print(f"    TPOT:           {tpot:.1f} ms")
    print(f"    TPS:            {tps:.1f} tokens/sec")
    print(f"    Total time:     {total_time_s:.3f} s")

    if num_decode > 0:
        print(f"    Decode min:     {min(decode_times):.1f} ms")
        print(f"    Decode max:     {max(decode_times):.1f} ms")

    # Steady-state metrics (exclude first decode step which may include JIT)
    if num_decode > 1:
        steady = decode_times[1:]
        steady_tpot = sum(steady) / len(steady)
        steady_total_s = (ttft + sum(steady)) / 1000
        steady_tokens = 1 + len(steady)
        steady_tps = steady_tokens / steady_total_s if steady_total_s > 0 else 0
        print(f"    Steady TPOT:    {steady_tpot:.1f} ms  (excl. 1st decode)")
        print(f"    Steady TPS:     {steady_tps:.1f} tokens/sec")

    # Op-level profiling
    if getattr(args, "profile", False):
        steps = 1 + num_decode  # prefill + decode
        print_profile(
            model.timer,
            label=f"Op Timing Profile ({steps} steps, backend={args.backend})",
        )

    # HuggingFace reference generation for comparison
    print(f"\n--- HuggingFace Reference Generation ---")
    with torch.no_grad():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        )
    hf_text = tokenizer.decode(hf_output[0][input_ids.shape[1] :])
    print(f"  {args.prompt}{hf_text}")


def run_interactive(hf_model, tokenizer, args):
    """Interactive REPL: type prompts, get completions with timing metrics."""
    from model import GPT2Model

    max_tokens = args.max_tokens if args.max_tokens > 0 else 40
    backend = args.backend

    # Select backend driver once
    if backend == "npu":
        import benchmark

        benchmark.select_npu_backend()
    elif backend in ("gpu", "hetero", "hetero-fast"):
        import benchmark

        benchmark.select_gpu_backend()

    state_dict = hf_model.state_dict()
    model = GPT2Model(state_dict, backend=backend, config=getattr(args, "config", None))

    model_name = getattr(args, "model", "gpt2")
    print(f"\n{model_name} ({backend.upper()}) | max_tokens={max_tokens}")
    print(f"Type a prompt and press Enter. Ctrl-C or 'quit' to exit.\n")

    # Warm up with a short generation so JIT compilation doesn't hit the first real prompt
    print("Warming up (first compilation)...", end="", flush=True)
    warmup_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        model.generate(warmup_ids, max_new_tokens=1)
    print(" done.\n")

    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit"):
            print("Bye.")
            break

        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

        with torch.no_grad():
            generated_ids, timing = model.generate(input_ids, max_new_tokens=max_tokens)

        generated_text = tokenizer.decode(generated_ids)
        print(f"{prompt}{generated_text}")

        # Compact metrics line
        ttft = timing["prefill_ms"]
        decode_times = timing["decode_times_ms"]
        num_decode = len(decode_times)
        total_tokens = 1 + num_decode
        total_time_s = (ttft + sum(decode_times)) / 1000
        tps = total_tokens / total_time_s if total_time_s > 0 else 0
        tpot = sum(decode_times) / num_decode if num_decode > 0 else 0

        parts = [
            f"{tps:.1f} tok/s",
            f"TTFT {ttft:.0f}ms",
            f"TPOT {tpot:.0f}ms",
            f"{total_tokens} tokens",
        ]
        # Show steady-state if enough decode steps
        if num_decode > 1:
            steady_tpot = sum(decode_times[1:]) / len(decode_times[1:])
            parts.append(f"steady {steady_tpot:.0f}ms")

        print(f"  [{' | '.join(parts)}]\n")


def make_progress_bar(width=30):
    """Return a callback that renders an in-place progress bar to stderr."""

    def callback(done, total):
        frac = done / total if total > 0 else 1.0
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(f"\r  [{bar}] {done}/{total} tokens")
        sys.stderr.flush()

    return callback


def print_profile(timer, label="Op Timing Profile"):
    """Print a formatted table from an OpTimer's summary."""
    rows = timer.summary()
    if not rows:
        return
    total = timer.total_ms()
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")
    print(
        f"  {'Op':<14} {'Total (ms)':>10}   {'Count':>5}   {'Avg (ms)':>8}   {'%':>6}"
    )
    print(f"  {'-'*14} {'-'*10}   {'-'*5}   {'-'*8}   {'-'*6}")
    for op, op_total, count, avg in rows:
        pct = (op_total / total * 100) if total > 0 else 0
        print(f"  {op:<14} {op_total:>10.1f}   {count:>5}   {avg:>8.2f}   {pct:>5.1f}%")
    print(f"  {'-'*14} {'-'*10}   {'-'*5}   {'-'*8}   {'-'*6}")
    print(f"  {'Total':<14} {total:>10.1f}")
    print(f"{'=' * 65}")


def main():
    parser = argparse.ArgumentParser(description="GPT-2 Triton Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="GPT-2 variant: gpt2 (124M), gpt2-medium (355M), gpt2-large (774M), gpt2-xl (1.5B)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gpu",
        choices=["gpu", "npu", "hetero", "hetero-fast", "reference"],
        help="Backend: gpu, npu, hetero (consistent NPU/GPU split), hetero-fast (GPU-only decode), reference (HF only)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox",
        help="Input prompt for inference",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Number of tokens to generate (0 = single forward pass only)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: type prompts, get completions in a loop",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-op timing profiling for the forward pass",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Resolve variant config from --model
    from model import GPT2_CONFIGS

    args.config = GPT2_CONFIGS[args.model]

    # Load HuggingFace model
    hf_model, tokenizer = load_hf_model(args.config["hf_name"])

    if args.interactive:
        if args.backend == "reference":
            print("Interactive mode is not supported with --backend reference.")
            sys.exit(1)
        run_interactive(hf_model, tokenizer, args)
        return

    # Tokenize prompt
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    print(f'\nPrompt: "{args.prompt}" ({seq_len} tokens)')

    if args.max_tokens > 0 and args.backend != "reference":
        # --- Generation mode ---
        run_generation(hf_model, tokenizer, input_ids, args)
    else:
        # --- Single forward pass mode (original behavior) ---
        _, ref_logits = run_reference(hf_model, tokenizer, args.prompt)

        print("\n--- HuggingFace Reference ---")
        print_generation(ref_logits, tokenizer, input_ids, args.prompt)

        if args.backend == "reference":
            return

        state_dict = hf_model.state_dict()
        triton_logits = run_triton_model(
            state_dict,
            input_ids,
            args.backend,
            profile=args.profile,
            config=args.config,
        )

        print(f"\n--- Triton ({args.backend.upper()}) ---")
        print_generation(triton_logits, tokenizer, input_ids, args.prompt)

        # Compare
        compare_logits(ref_logits, triton_logits, tokenizer, input_ids)


if __name__ == "__main__":
    main()
    # Backstop for the XRT/ROCm process-global teardown fault: results are fully
    # computed and printed by now, so skip interpreter finalization (C++ static
    # destructors) entirely. Flush first since os._exit does not. NPU resources
    # are released in order via _FusedMLP/NPUChain.__del__ during normal GC; this
    # only guards the residual global-teardown crash GC ordering cannot reach.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
