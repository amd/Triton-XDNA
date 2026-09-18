# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""End to end: a Triton prefill feeding mlir-air's Q4NX fused decode.

    python qwen25_7b_q4nx_inference.py --backend npu --max-tokens 20
    python qwen25_7b_q4nx_inference.py --prefill-only     # no decode build needed

The prefill is this repo's -- Triton kernels on XDNA (prefill.py, and
../llm_q4nx/kernels.py). The decode is mlir-air's fused Q4NX decode, run
unmodified: 28 layers plus the untied LM head in one dispatch.

Qwen2.5-7B is Llama-shaped in every respect the operator routing cares about,
and differs in one: q, k and v each carry a bias, added to the raw projection
output before RoPE. That is a forward of its own
(`../llm_q4nx/qwen25_prefill.py`) because nothing else here has a bias on any
projection. It is *not* Qwen3 -- there is no qk-norm.

It is also the first example here whose weights are not a Q4NX bundle. There is
no FastFlowLM Qwen2.5-7B NPU2 bundle, so mlir-air rounds an ungated upstream HF
checkpoint onto the Q4NX grid at load time, with the quantizer the decode's own
cascade cache uses. Expect the weight load to dominate a prefill-only run.

Its driver's `generate()` takes no handoff parameter, as the Qwen3 ones do not,
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
    from qwen25_prefill import Qwen25Prefill

    return harness.main(
        registry.spec(config.MODEL_NAME),
        config,
        Qwen25Prefill,
        doc=__doc__,
        argv=argv,
    )


if __name__ == "__main__":
    from qwen25_prefill import Qwen25Prefill

    raise SystemExit(
        harness.run(
            registry.spec(config.MODEL_NAME), config, Qwen25Prefill, doc=__doc__
        )
    )
