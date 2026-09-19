# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""End to end: a Triton prefill feeding mlir-air's Q4NX fused decode.

    python qwen3_8b_q4nx_inference.py --backend npu --max-tokens 20
    python qwen3_8b_q4nx_inference.py --prefill-only     # no decode build needed

The prefill is this repo's -- Triton kernels on XDNA (prefill.py, and
../llm_q4nx/kernels.py). The decode is mlir-air's fused Q4NX decode, run
unmodified: 36 layers plus the untied LM head in one dispatch.

Qwen3-8B is Qwen3-4B's block at 4096 and reuses its forward
(`../llm_q4nx/qwen3_prefill.py`) with no change at all -- the per-head qk-norm
between the QKV projection and RoPE is the same, and so is the single-theta
RoPE. What it adds is size, in two places that are not the forward: the LM head
is untied (its own Q4NX tensor, as on Llama-3.1-8B), and at K=4096 the decode
needs both a smaller AIE core stack and its DDR weights split over four buffers
-- 36 layers is 4.04 GiB and a shim BD's byte offset only reaches 4. Both are
recorded in `../llm_q4nx/registry.py`.

Its driver's `generate()` takes no handoff parameter, as Qwen3-4B's does not,
so the harness substitutes the one prefill function it calls;
`ModelSpec.driver_api` names which of the three each model uses.

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
    from qwen3_prefill import Qwen3Prefill

    return harness.main(
        registry.spec(config.MODEL_NAME),
        config,
        Qwen3Prefill,
        doc=__doc__,
        argv=argv,
    )


if __name__ == "__main__":
    from qwen3_prefill import Qwen3Prefill

    raise SystemExit(
        harness.run(registry.spec(config.MODEL_NAME), config, Qwen3Prefill, doc=__doc__)
    )
