# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the Qwen3-4B Q4NX prefill and write the decode's KV handoff.

    python prefill.py --backend cpu
    python prefill.py --backend npu --ops all
    python prefill.py --backend npu --kv-out /tmp/prefill_kv.npz

The gate is the first generated token: 12095 (" Paris") for the canonical
prompt. `--kv-out` writes the handoff as an npz; unlike the 1B, Qwen3-4B's
mlir-air driver does not read one (it is handed the arrays -- see
`ModelSpec.driver_api`), so this is for inspecting a prefill, not for feeding
the decode. `model.save_kv_npz` states the layout.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

# After `config`, and only after: it puts the shared harness on sys.path, and
# the forward qwen3_prefill holds resolves its dims from the `config` this
# directory just bound.
sys.path.insert(0, config._SHARED)

import prefill_cli  # noqa: E402
from qwen3_prefill import Qwen3Prefill  # noqa: E402


def main(argv=None):
    return prefill_cli.main(Qwen3Prefill, config, doc=__doc__, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
