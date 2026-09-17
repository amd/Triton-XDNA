# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""End to end: a Triton prefill feeding mlir-air's Q4NX fused decode.

    python qwen3_4b_q4nx_inference.py --backend npu --max-tokens 20
    python qwen3_4b_q4nx_inference.py --prefill-only     # no decode build needed

The prefill is this repo's -- Triton kernels on XDNA (prefill.py, and
../llm_q4nx/kernels.py). The decode is mlir-air's fused Q4NX decode, run
unmodified: 36 layers plus the tied LM head in one dispatch.

Qwen3-4B is the first non-Llama family here, and the first that needed its own
forward (`../llm_q4nx/qwen3_prefill.py`): each head's 128 lanes are
RMS-normalized between the QKV projection and RoPE. It is also the first with
DQ != D -- 32 heads of 128 against a 2560 model dim -- so o_proj contracts
4096 -> 2560 rather than being square.

Its driver's `generate()` takes no handoff parameter, unlike the 1B's npz and
the 3B's prefiller object, so the harness substitutes the one prefill function
it calls; `ModelSpec.driver_api` names which of the three each model uses.

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
