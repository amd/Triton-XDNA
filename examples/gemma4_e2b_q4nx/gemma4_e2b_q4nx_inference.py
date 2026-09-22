# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""End to end: a Triton prefill feeding mlir-air's Q4NX fused decode.

    python gemma4_e2b_q4nx_inference.py --backend npu --max-tokens 4
    python gemma4_e2b_q4nx_inference.py --prefill-only     # no decode build needed

The prefill is this repo's -- Triton kernels on XDNA (prefill.py, and
../llm_q4nx/kernels.py). The decode is mlir-air's fused Q4NX decode, run
unmodified: 34 layers plus the tied LM head in one dispatch.

Gemma3-4B is the furthest from the Llama block of anything here
(`../llm_q4nx/gemma4_prefill.py`): each sublayer is wrapped in a norm sandwich
rather than preceded by one norm, RoPE uses two thetas chosen per layer, five
layers in six limit attention to a 1024-token sliding window, and the GLU is
GELU-tanh instead of SiLU -- the one new Triton kernel this model needed. It
also carries Qwen3's per-head qk-norm and decoupled q dim.

Four Gemma conventions are already resolved in the weight bundle -- the (1+w)
norm fold, the embedding scale, the separately-stored unscaled LM head, and the
folded qk-norm weights. `config.load_q4nx` lists them; re-applying any produces
fluent wrong text rather than an error.

Its driver takes the same `kv_arrays` handoff Qwen3's does: no handoff
parameter, so the harness substitutes the one prefill function it calls.

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
    from gemma4_prefill import Gemma4Prefill

    return harness.main(
        registry.spec(config.MODEL_NAME),
        config,
        Gemma4Prefill,
        doc=__doc__,
        argv=argv,
    )


if __name__ == "__main__":
    from gemma4_prefill import Gemma4Prefill

    raise SystemExit(
        harness.run(
            registry.spec(config.MODEL_NAME), config, Gemma4Prefill, doc=__doc__
        )
    )
