# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""End to end: a Triton prefill feeding mlir-air's Q4NX fused decode.

    python phi4_mini_q4nx_inference.py --backend npu --max-tokens 20
    python phi4_mini_q4nx_inference.py --prefill-only     # no decode build needed

The prefill is this repo's -- Triton kernels on XDNA (prefill.py, and
../llm_q4nx/kernels.py). The decode is mlir-air's fused Q4NX decode, run
unmodified: 32 layers plus the tied LM head in one dispatch.

Phi-4-mini is Llama-shaped but for the rotation. `partial_rotary_factor=0.75`
means RoPE covers only the leading 96 of each head's 128 lanes and the trailing
32 are copied through, and its frequencies come from a LongRoPE factor table
carried in the bundle rather than from a closed form. That is one overridden
operator (`../llm_q4nx/phi4_prefill.py`), not a new block -- the first model
here whose delta is in `_rope` rather than in `_layer`.

Its driver takes the prefill object, as the 3B's and 8B's do, so ours goes in
directly with nothing to neutralize -- with one argument that matters:
`min_prefill=1`. Their default is 96 tokens, below which the prompt is replayed
token-by-token through the decode and the prefill this example exists to test is
never touched. The harness passes it; `ModelSpec.driver_api` names which of the
three shapes each model uses.

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
    from phi4_prefill import Phi4Prefill

    return harness.main(
        registry.spec(config.MODEL_NAME),
        config,
        Phi4Prefill,
        doc=__doc__,
        argv=argv,
    )


if __name__ == "__main__":
    from phi4_prefill import Phi4Prefill

    raise SystemExit(
        harness.run(registry.spec(config.MODEL_NAME), config, Phi4Prefill, doc=__doc__)
    )
