# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""End to end: a Triton prefill feeding mlir-air's Q4NX fused decode.

    python llama32_1b_q4nx_inference.py --backend npu --max-tokens 20
    python llama32_1b_q4nx_inference.py --prefill-only     # no decode build needed

The prefill is this repo's -- Triton kernels on XDNA (prefill.py, and
../llm_q4nx/kernels.py). The decode is mlir-air's fused Q4NX decode, run
unmodified: the shared harness writes the KV handoff npz that its `generate()`
already loads, and neutralizes only the step that would have produced it.

Everything here is Llama-3.2-1B-specific and nothing else is: the driver,
the decode build and the HSA dispatch all live in ../llm_q4nx/ and are shared
with the other model families.

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
    from model import LlamaPrefill

    return harness.main(
        registry.spec("llama-3.2-1b"),
        config,
        LlamaPrefill,
        doc=__doc__,
        argv=argv,
    )


if __name__ == "__main__":
    from model import LlamaPrefill

    raise SystemExit(
        harness.run(registry.spec("llama-3.2-1b"), config, LlamaPrefill, doc=__doc__)
    )
