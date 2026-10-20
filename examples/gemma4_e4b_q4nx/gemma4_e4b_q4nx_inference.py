# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""End to end: mlir-air's fused NPU prefill feeding its Q4NX fused decode.

    python gemma4_e4b_q4nx_inference.py --max-tokens 4
    python gemma4_e4b_q4nx_inference.py --prefill-only     # no decode build needed

Gemma4-E4B is Gemma4-E2B's architecture at a larger size -- see
`../gemma4_e2b_q4nx/gemma4_e2b_q4nx_inference.py` for what sets the family apart
(per-layer embeddings, sliding and full layers of different head widths, layers
that attend a lower layer's KV). What changes is in `config.py`: the width, the
layer count and pattern, and two KV heads instead of one. mlir-air runs E2B's
driver and builders at these dims, and so does this example.

The default prefill is mlir-air's fused NPU prefill (`--prefill-engine
air-fused`); the decode is its `fused_decode_ple`, 42 layers plus the LM head
in one dispatch.

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
