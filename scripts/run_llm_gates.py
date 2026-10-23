#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Run the Q4NX LLM gates on the NPU.

Each model runs every prefill/decode placement that
examples/llm_q4nx/registry.py says it supports. `--scope nightly` runs that for
every model; `--scope pr` runs a subset on two models that together still
reach every placement once.

A model's cases run back to back after its own decode build: the decode
artifacts live in one directory per decode engine, stamped with the model they
were built for, so the next model's build replaces them.

Exit 77 from an example is a skip (the environment cannot run it), as in
run_tests.py. Exit status is 1 if anything failed.

    python scripts/run_llm_gates.py --scope pr
    python scripts/run_llm_gates.py --scope nightly -m qwen3_4b_q4nx
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
SHARED = EXAMPLES / "llm_q4nx"

# Directory name -> registry name, in the order the models run. Most of what a
# model adds fails into fluent wrong text rather than an error, which is why
# the gates compare token ids against recorded ones.
MODELS = {
    "llama32_1b_q4nx": "llama-3.2-1b",
    "llama32_3b_q4nx": "llama-3.2-3b",
    # Declines (exit 77) on a host too small for its bf16 weights.
    "llama31_8b_q4nx": "llama-3.1-8b",
    # Per-head qk-norm and a decoupled q dim.
    "qwen3_4b_q4nx": "qwen3-4b",
    # Untied LM head; its own decode core stack and weight split.
    "qwen3_8b_q4nx": "qwen3-8b",
    # Biased projections; the only decode env with W_DUAL_CHAN=0.
    "qwen25_3b_q4nx": "qwen2.5-3b",
    # Quantizes an upstream checkpoint on load. Declines (exit 77) on a host
    # too small for its bf16 weights.
    "qwen25_7b_q4nx": "qwen2.5-7b",
    # Partial rotation (96 of 128 lanes) on LongRoPE frequencies.
    "phi4_mini_q4nx": "phi4-mini",
    # Norm sandwich, dual-theta RoPE, GELU GLU, sliding window in the decode.
    "gemma3_4b_q4nx": "gemma3-4b",
    # Per-layer embeddings and class map, KV sharing, the PLE decode engine.
    "gemma4_e2b_q4nx": "gemma4-e2b",
    # E2B's architecture at larger dims: two KV heads, one per attention CU.
    "gemma4_e4b_q4nx": "gemma4-e4b",
}

# On PRs: Llama-3.2-1B takes the KV handoffs and the full-ELF decode, Gemma4-E2B
# the other model family and the hetero prefill. Case names are those in `plan`.
PR_CASES = {
    "llama32_1b_q4nx": {
        "default",
        "triton",
        "triton-gpu",
        "igpu-gpu",
        "igpu-npu",
        "zero-copy",
        "dense-decode-vs-prefill",
        "elf",
    },
    "gemma4_e2b_q4nx": {
        "default",
        "triton",
        "triton-fused",
        "triton-fused-gemm",
        "triton-fused-lm-head",
        "hetero",
        "triton-gpu",
        "igpu-gpu",
        "decode-vs-prefill",
    },
}

# Greedy tokens to generate: at least as many as any recorded continuation.
MAX_TOKENS = "10"
TIMEOUT = 1200

HIDE_IGPU = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": "",
}
SHOW_IGPU = {"HSA_OVERRIDE_GFX_VERSION": "11.5.1"}


@dataclass
class Case:
    name: str
    # What to run, relative to `cwd`.
    cmd: list
    cwd: Path
    env: dict = field(default_factory=dict)
    # Needs the model's NPU decode built.
    npu_decode: bool = False
    # Needs it built as mlir-air's full ELF (DECODE_ELF=1).
    elf: bool = False
    # Check that the KV cache reached the NPU decode in place.
    zero_copy: bool = False


def has_fused_prefill(model):
    import triton.language as tl

    try:
        tl.extra.npu.PrefillConfig(model)
    except tl.extra.npu.PrefillConfigError:
        return False
    return True


def plan(ex, spec, fused):
    """Every case `spec` supports, for example directory `ex`."""
    d = EXAMPLES / ex
    entry = [sys.executable, f"{ex}_inference.py"]
    gen = ["--max-tokens", MAX_TOKENS, "--greedy"]
    cases = [
        # The default prefill engine (mlir-air's fused one where the model has
        # it) and mlir-air's NPU decode, gated on the recorded continuation.
        Case("default", entry + gen, d, HIDE_IGPU, npu_decode=True),
    ]
    if fused:
        cases.append(
            Case(
                "triton",
                entry + ["--prefill-engine", "triton"] + gen,
                d,
                HIDE_IGPU,
                npu_decode=True,
            )
        )
    if spec.supports_igpu:
        cases += [
            # RoPE and attention of the Triton prefill on the iGPU.
            Case(
                "hetero",
                entry + ["--prefill-engine", "triton", "--backend", "hetero"] + gen,
                d,
                SHOW_IGPU,
                npu_decode=True,
            ),
            Case(
                "triton-gpu",
                entry + ["--prefill-engine", "triton", "--decode", "gpu"] + gen,
                d,
                SHOW_IGPU,
            ),
            Case(
                "igpu-gpu",
                entry + ["--prefill-engine", "igpu", "--decode", "gpu"] + gen,
                d,
                SHOW_IGPU,
            ),
            Case(
                "igpu-npu",
                entry + ["--prefill-engine", "igpu"] + gen,
                d,
                SHOW_IGPU,
                npu_decode=True,
                zero_copy=spec.supports_xrt_zero_copy,
            ),
        ]
    if spec.supports_xrt_zero_copy:
        # The default run again, with the iGPU visible so the prefill writes
        # the decode's KV cache into pages XRT and HIP share.
        cases.append(
            Case(
                "zero-copy", entry + gen, d, SHOW_IGPU, npu_decode=True, zero_copy=True
            )
        )
    if ex == "llama32_1b_q4nx":
        # Each iGPU decode step against a prefill re-run over the growing
        # prefix, after both the host and the iGPU prefill.
        cases.append(
            Case(
                "dense-decode-vs-prefill",
                [sys.executable, "test_dense_decode_vs_prefill.py", "--example", ex],
                SHARED,
                SHOW_IGPU,
            )
        )
    if ex == "gemma4_e2b_q4nx":
        # mlir-air's fused prefill device with its GEMMs and LM head from
        # Triton, then its NPU decode; and the GEMM and the LM head alone
        # against a reference, packed and dispatched by mlir-air's runtime.
        cases.append(
            Case(
                "triton-fused",
                entry + ["--prefill-engine", "triton-fused"] + gen,
                d,
                HIDE_IGPU,
                npu_decode=True,
            )
        )
        cases += [
            Case(name, [sys.executable, test], SHARED, HIDE_IGPU)
            for name, test in (
                ("triton-fused-gemm", "test_gemma4_gemm.py"),
                ("triton-fused-lm-head", "test_gemma4_lm_gemv.py"),
            )
        ]
        # Each decode against a prefill re-run over the growing prefix, and
        # mlir-air's NPU decode from the prefill's slab against its own
        # `seed_kv`. Needs the iGPU for its other rows.
        cases.append(
            Case(
                "decode-vs-prefill",
                [
                    sys.executable,
                    "test_decode_vs_prefill.py",
                    "--npu-decode",
                    "--prefill-engine",
                    "air-fused",
                ],
                SHARED,
                SHOW_IGPU,
                npu_decode=True,
            )
        )
    if spec.engine == "fused_decode":
        # mlir-air's full-ELF decode under XRT, which only the shared decode
        # engine builds.
        cases.append(
            Case(
                "elf",
                entry + gen,
                d,
                {**HIDE_IGPU, "DECODE_ELF": "1"},
                npu_decode=True,
                elf=True,
            )
        )
    return cases


class Report:
    def __init__(self):
        self.passed = self.failed = self.skipped = 0
        self.failures = []

    def ok(self, what):
        print(f"   ✅ PASS: {what}", flush=True)
        self.passed += 1

    def skip(self, what, why):
        print(f"   ⏭️  SKIP: {what} -- {why}", flush=True)
        self.skipped += 1

    def fail(self, what, why):
        print(f"   ❌ FAIL: {what} ({why})", flush=True)
        self.failed += 1
        self.failures.append(what)


def run(cmd, cwd, env, capture=False):
    full_env = {**os.environ, **env}
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            env=full_env,
            timeout=TIMEOUT,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
        )
    except subprocess.TimeoutExpired:
        return None, ""
    if capture:
        print(p.stdout, end="", flush=True)
    return p.returncode, p.stdout if capture else ""


def build_decode(ex, elf, report):
    """`make compile-decode` for one model, as the xclbin templates or as the
    full ELF (which also builds the templates). Returns whether it succeeded."""
    what = f"{ex}: decode build ({'elf' if elf else 'xclbin'})"
    print(f"📁 {what}", flush=True)
    env = {**HIDE_IGPU, "DECODE_ELF": "1"} if elf else HIDE_IGPU
    rc, _ = run(["make", "compile-decode"], EXAMPLES / ex, env)
    if rc == 0:
        report.ok(what)
        return True
    report.fail(what, "timeout" if rc is None else f"exit code {rc}")
    return False


def run_case(ex, case, report):
    what = f"{ex}: {case.name} ({' '.join(case.cmd[1:])})"
    print(f"📁 {what}", flush=True)
    start = time.time()
    rc, out = run(case.cmd, case.cwd, case.env, capture=case.zero_copy)
    took = f"{time.time() - start:.0f}s"
    if rc is None:
        report.fail(what, f"timeout after {TIMEOUT}s")
    elif rc == 77:
        report.skip(what, "declined (exit 77)")
    elif rc != 0:
        report.fail(what, f"exit code {rc}, {took}")
    elif not case.zero_copy:
        report.ok(f"{what} [{took}]")
    elif "KV handed over in place" in out and "copied once" not in out:
        report.ok(f"{what}, KV handed over in place [{took}]")
    elif "XRT zero-copy unavailable" in out:
        report.skip(what, "no shared pages here")
    else:
        # Gated green, but the zero-copy path silently stopped running.
        report.fail(what, "neither handed the KV over in place nor said why not")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scope", choices=("pr", "nightly"), default="pr")
    ap.add_argument(
        "-m",
        "--model",
        action="append",
        choices=list(MODELS),
        help="only these example directories (repeatable)",
    )
    ap.add_argument(
        "--list", action="store_true", help="print the cases instead of running them"
    )
    args = ap.parse_args()

    sys.path.insert(0, str(SHARED))
    import registry

    report = Report()
    models = args.model or (list(MODELS) if args.scope == "nightly" else list(PR_CASES))
    for ex in models:
        name = MODELS[ex]
        spec = registry.spec(name)
        cases = plan(ex, spec, has_fused_prefill(name))
        if args.scope == "pr":
            cases = [c for c in cases if c.name in PR_CASES.get(ex, ())]
        print(f"\n========== {ex} ({len(cases)} cases) ==========", flush=True)
        if args.list:
            for c in cases:
                decode = (
                    ("elf-decode" if c.elf else "npu-decode") if c.npu_decode else ""
                )
                print(f"  {c.name:24} {decode:11} {' '.join(c.cmd[1:])}")
            continue

        built = {}
        for case in cases:
            if case.npu_decode and case.elf not in built:
                built[case.elf] = build_decode(ex, case.elf, report)
            if case.npu_decode and not built[case.elf]:
                report.skip(f"{ex}: {case.name}", "its decode build failed")
                continue
            run_case(ex, case, report)

    if args.list:
        return 0
    print("\n" + "-" * 50)
    print(f"LLM gate results ({args.scope}):")
    print(f"  ✅ Passed:  {report.passed}")
    print(f"  ❌ Failed:  {report.failed}")
    print(f"  ⏭️  Skipped: {report.skipped}")
    print(f"  📊 Total:   {report.passed + report.failed + report.skipped}")
    for what in report.failures:
        print(f"     ❌ {what}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
