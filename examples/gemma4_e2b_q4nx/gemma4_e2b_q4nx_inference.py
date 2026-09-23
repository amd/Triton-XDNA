# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""End to end: a Triton prefill feeding mlir-air's Q4NX fused decode.

    python gemma4_e2b_q4nx_inference.py --backend npu --max-tokens 4
    python gemma4_e2b_q4nx_inference.py --prefill-only     # no decode build needed

The prefill is this repo's -- Triton kernels on XDNA (prefill.py, and
../llm_q4nx/kernels.py). The decode is mlir-air's fused Q4NX decode, run
unmodified: 35 layers plus its own untied LM head in one dispatch.

Gemma4-E2B takes over from Gemma3-4B as the furthest from the Llama block of
anything here (`../llm_q4nx/gemma4_prefill.py`). 35 layers, an untied LM head,
and three things no other model here has:

* **per-layer embeddings** -- a 256-wide vector per layer per token, computed
  once from the input embeddings and injected after the MLP through a gated
  projection and a fifth norm;
* **layers that differ in shape from each other** -- four sliding layers then
  one full, repeating, with the head dim (256 vs 512), the RoPE base (1e4 vs
  1e6) and the window (512 vs none) all following, and the FFN doubling to
  12288 from layer 15 up;
* **twenty layers with no KV cache of their own**, which attend a lower
  layer's.

It keeps Gemma3's norm sandwich, GELU-tanh GLU and per-head qk-norm. Two
smaller things are quiet when wrong: the attention scale is 1.0 rather than
`head_dim**-0.5`, and each block's output carries a per-layer scalar.

Unlike Gemma3, the (1+w) norm fold is **not** applied to this model -- the
bundle ships raw HF weights and the device multiplies by `w` with no add.
Copying the sibling here is the obvious mistake; `config.RMS_NORM_ADDS_ONE`
records it.

**Its decode is the only one here built by a different engine**: mlir-air's
`fused_decode_ple`, a fork carrying the per-layer-embedding branch.
`registry.ModelSpec.engine` says so and `tl.extra.npu.fused_decode` takes it as
an argument.

Its driver takes the same `kv_arrays` handoff Qwen3's and Gemma3's do: no
handoff parameter, so the harness substitutes the one prefill function it
calls. On this model that handoff is a pair of LISTS rather than stacked
arrays, because the per-layer caches are not the same width.

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
