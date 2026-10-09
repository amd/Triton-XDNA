# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the Gemma4-E2B Q4NX prefill and gate its first token.

    python prefill.py --backend cpu
    python prefill.py --backend npu --ops all
    python prefill.py --backend npu --compare-cpu

The gate is the first generated token: 9079 (" Paris") for the canonical
prompt, whose leading `2` is a <bos> this model's tokenizer does not emit --
see `config.PROMPT`.

**No `--kv-out` here.** Every sibling can write the handoff as one npz because
its layers all carry the same-width cache; this model's do not (256 lanes on a
sliding layer, 512 on a full one), so there is no array to save. `--compare-cpu`
covers what that option was for.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

# After `config`, and only after: it puts the shared harness on sys.path, and
# the forward gemma4_prefill holds resolves its dims from the `config` this
# directory just bound.
sys.path.insert(0, config._SHARED)

import prefill_cli  # noqa: E402
from gemma4_prefill import Gemma4Prefill  # noqa: E402


def main(argv=None):
    return prefill_cli.main(Gemma4Prefill, config, doc=__doc__, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
