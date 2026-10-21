#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Run the Q4NX LLM gates on the NPU.

Each model runs every prefill/decode placement and runtime that
examples/llm_q4nx/registry.py says it supports. `--scope nightly` runs that for
every model; `--scope pr` runs a subset on two models that together still
reach every placement and runtime once.

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

# On PRs: Llama-3.2-1B takes the KV handoffs, Gemma4-E2B the other model family
# and the hetero prefill. The full-ELF decode and HSA are left to nightly while
# they fail on the runner: the ELF decode build does not write the
# decode_scratchpad.args mlir-air now reads, and HSA hits issue #151. Case names
# are those in `plan`.
PR_CASES = {
    "llama32_1b_q4nx": {
        "default",
        "triton",
        "triton-gpu",
        "igpu-gpu",
        "igpu-npu",
        "zero-copy",
        "dense-decode-vs-prefill",
    },
    "gemma4_e2b_q4nx": {
        "default",
        "triton",
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
    # Decode build this case needs: "xclbin", "elf", "hsa" or None.
    decode: str = None
    # Check that the KV cache reached the NPU decode in place.
    zero_copy: bool = False


def has_fused_prefill(model):
    import triton.language as tl

    try:
        tl.extra.npu.PrefillConfig(model)
    except tl.extra.npu.PrefillConfigError:
        return False
    return True


def rocr_env():
    """HSA settings for a process that also imports ROCm torch, or None if no
    AIE-capable ROCR is installed. See README, "Sharing a process with
    PyTorch"."""
    root = os.environ.get("AMD_NPU_ROCR_PATH")
    if not root:
        return None
    lib = Path(root) / "lib" / "libhsa-runtime64.so.1"
    if not lib.is_file():
        return None
    preload = os.environ.get("LD_PRELOAD")
    return {
        "AMD_TRITON_NPU_RUNTIME": "hsa",
        "LD_PRELOAD": f"{lib}:{preload}" if preload else str(lib),
    }


def plan(ex, spec, fused):
    """Every case `spec` supports, for example directory `ex`."""
    d = EXAMPLES / ex
    entry = [sys.executable, f"{ex}_inference.py"]
    gen = ["--max-tokens", MAX_TOKENS, "--greedy"]
    cases = [
        # The default prefill engine (mlir-air's fused one where the model has
        # it) and mlir-air's NPU decode, gated on the recorded continuation.
        Case("default", entry + gen, d, HIDE_IGPU, decode="xclbin"),
    ]
    if fused:
        cases.append(
            Case(
                "triton",
                entry + ["--prefill-engine", "triton"] + gen,
                d,
                HIDE_IGPU,
                decode="xclbin",
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
                decode="xclbin",
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
                decode="xclbin",
                zero_copy=spec.supports_xrt_zero_copy,
            ),
        ]
    if spec.supports_xrt_zero_copy:
        # The default run again, with the iGPU visible so the prefill writes
        # the decode's KV cache into pages XRT and HIP share.
        cases.append(
            Case(
                "zero-copy", entry + gen, d, SHOW_IGPU, decode="xclbin", zero_copy=True
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
                decode="xclbin",
            )
        )
    if spec.engine == "fused_decode":
        # mlir-air's full-ELF decode under XRT, which only the shared decode
        # engine builds.
        cases.append(
            Case("elf", entry + gen, d, {**HIDE_IGPU, "DECODE_ELF": "1"}, decode="elf")
        )
    if spec.supports_hsa:
        # The HSA runtime cannot share a process with torch's HIP, so it only
        # pairs with an NPU prefill and the NPU decode.
        cases.append(Case("hsa", entry + gen, d, HIDE_IGPU, decode="hsa"))
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


def build_decode(ex, fmt, first, rocr, report):
    """Build the decode in one format. The model's first build compiles its AIE
    kernels too; later ones only relower the templates. Returns whether the
    build succeeded."""
    what = f"{ex}: decode build ({fmt})"
    env = dict(HIDE_IGPU)
    target = "compile-decode" if first else "recompile-decode"
    if fmt == "elf":
        env["DECODE_ELF"] = "1"
    elif fmt == "hsa":
        env.update(rocr)
        env["RUNTIME"] = "hsa"
    print(f"📁 {what}", flush=True)
    rc, _ = run(["make", target], EXAMPLES / ex, env)
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

    rocr = rocr_env()
    report = Report()
    models = args.model or (list(MODELS) if args.scope == "nightly" else list(PR_CASES))
    if args.list:
        print(f"AIE-capable ROCR: {'yes' if rocr else 'no (HSA cases skip)'}")
    for ex in models:
        name = MODELS[ex]
        spec = registry.spec(name)
        cases = plan(ex, spec, has_fused_prefill(name))
        if args.scope == "pr":
            cases = [c for c in cases if c.name in PR_CASES.get(ex, ())]
        print(f"\n========== {ex} ({len(cases)} cases) ==========", flush=True)
        if args.list:
            for c in cases:
                print(f"  {c.name:24} decode={c.decode or '-':7} {' '.join(c.cmd[1:])}")
            continue

        built = {}
        for case in cases:
            fmt = case.decode
            if fmt == "hsa" and rocr is None:
                report.skip(
                    f"{ex}: {case.name}", "no AIE-capable ROCR (AMD_NPU_ROCR_PATH)"
                )
                continue
            if fmt and fmt not in built:
                built[fmt] = build_decode(ex, fmt, not built, rocr, report)
            if fmt and not built[fmt]:
                report.skip(f"{ex}: {case.name}", f"its {fmt} decode build failed")
                continue
            if fmt == "hsa":
                case.env = {**case.env, **rocr}
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
