# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""End to end: a Triton prefill feeding mlir-air's Q4NX fused decode.

    python llama32_3b_q4nx_inference.py --backend npu --max-tokens 20
    python llama32_3b_q4nx_inference.py --prefill-only     # no decode build needed

The prefill is this repo's -- Triton kernels on XDNA (prefill.py, and
../llm_q4nx/kernels.py). The decode is mlir-air's fused Q4NX decode, run
unmodified: the shared harness writes the KV handoff npz that its `generate()`
already loads, and neutralizes only the step that would have produced it.

Llama-3.2-3B is architecturally the 1B -- SwiGLU, one norm pair per block, no
qk-norm -- so it shares the 1B's forward as well as the harness. Everything
3B-specific is in `config.py`: 28 layers of 3072, 128-wide heads, and its own
weight bundle.

Prerequisite for generation (not for --prefill-only):

    make compile-decode
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

sys.path.insert(0, config._SHARED)

import harness  # noqa: E402
import registry  # noqa: E402

SKIP_EXIT_CODE = harness.SKIP_EXIT_CODE


def main(argv=None):
    from llama_prefill import LlamaPrefill

    return harness.main(
        registry.spec(config.MODEL_NAME),
        config,
        LlamaPrefill,
        doc=__doc__,
        argv=argv,
    )


if __name__ == "__main__":
    from llama_prefill import LlamaPrefill

    raise SystemExit(
        harness.run(registry.spec(config.MODEL_NAME), config, LlamaPrefill, doc=__doc__)
    )
