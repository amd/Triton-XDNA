# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the Phi-4-mini Q4NX prefill and write the decode's KV handoff.

    python prefill.py --backend cpu
    python prefill.py --backend npu --ops all
    python prefill.py --backend npu --kv-out /tmp/prefill_kv.npz

The gate is the first generated token: 12650 (" Paris") for the canonical
prompt -- its own tokenizer, so neither the prompt ids nor the expected token
match any other model's. `--kv-out` writes the handoff as an npz; Phi-4-mini's
mlir-air driver takes the prefill object rather than a file (see
`ModelSpec.driver_api`), so this is for inspecting a prefill, not for feeding
the decode. `model.save_kv_npz` states the layout.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

# After `config`, and only after: it puts the shared harness on sys.path, and
# the forward phi4_prefill holds resolves its dims from the `config` this
# directory just bound.
sys.path.insert(0, config._SHARED)

import prefill_cli  # noqa: E402
from phi4_prefill import Phi4Prefill  # noqa: E402


def main(argv=None):
    return prefill_cli.main(Phi4Prefill, config, doc=__doc__, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
