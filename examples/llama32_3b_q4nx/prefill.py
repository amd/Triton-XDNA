# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the Llama-3.2-3B Q4NX prefill and write the decode's KV handoff.

    python prefill.py --backend cpu
    python prefill.py --backend npu --ops all
    python prefill.py --backend npu --kv-out /tmp/prefill_kv.npz

The gate is the first generated token: 12366 (" Paris") for the canonical
prompt. The npz is what mlir-air's fused decode consumes; `model.save_kv_npz`
states that layout.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

# After `config`, and only after: it puts the shared harness on sys.path, and
# the forward llama_prefill holds resolves its dims from the `config` this
# directory just bound.
sys.path.insert(0, config._SHARED)

import prefill_cli  # noqa: E402
from llama_prefill import LlamaPrefill  # noqa: E402


def main(argv=None):
    return prefill_cli.main(LlamaPrefill, config, doc=__doc__, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
