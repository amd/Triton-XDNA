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

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

# After `config`, and only after: it puts the shared harness on sys.path, and
# the forward gemma4_prefill holds resolves its dims from the `config` this
# directory just bound.
sys.path.insert(0, config._SHARED)

from gemma4_prefill import Gemma4Prefill  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("cpu", "npu", "hetero"), default="npu")
    ap.add_argument(
        "--ops",
        default=None,
        help="NPU ops to enable: 'all', or a comma list of "
        "matmul,rms_norm,geglu (default: all)",
    )
    ap.add_argument("--prompt", default=None, help="token ids, comma separated")
    ap.add_argument("--n-layers", type=int, default=config.N_LAYERS)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--compare-cpu",
        action="store_true",
        help="also run the CPU reference and report per-layer KV divergence",
    )
    args = ap.parse_args(argv)

    ids = (
        [int(t) for t in args.prompt.split(",")] if args.prompt else list(config.PROMPT)
    )

    m = Gemma4Prefill(
        backend=args.backend,
        ops=args.ops,
        n_layers=args.n_layers,
        max_seq=args.max_seq,
        model=args.model,
    )
    t0 = time.time()
    m.load_weights()
    t_load = time.time() - t0
    print(
        f"[prefill] weights {config.MODEL_DEFAULT} ({m.fingerprint}) in {t_load:.1f}s"
    )

    t0 = time.time()
    logits = m.prefill(ids)
    t_run = time.time() - t0

    top = torch.topk(logits, 5)
    first = int(top.indices[0])
    print(
        f"[prefill] backend={args.backend} ops={sorted(m.enabled)} "
        f"P={len(ids)} in {t_run:.2f}s"
    )
    print(
        f"[prefill] top5 {top.indices.tolist()} {[round(float(v),2) for v in top.values]}"
    )

    ok = args.n_layers == config.N_LAYERS and ids == list(config.PROMPT)
    if ok:
        verdict = "PASS" if first == config.EXPECT_FIRST else "FAIL"
        print(
            f"[prefill] first token {first} (expect {config.EXPECT_FIRST}) -- {verdict}"
        )
    else:
        print(f"[prefill] first token {first} (gate not applicable)")
        verdict = "PASS"

    if args.compare_cpu and args.backend != "cpu":
        ref = Gemma4Prefill(backend="cpu", n_layers=args.n_layers, max_seq=args.max_seq)
        # Not an attribute-by-attribute copy: Gemma has two RoPE tables, and
        # listing them here is what the next subclass would forget. The class
        # declares what it owns (`WEIGHT_ATTRS`) and this asks for all of it.
        ref.share_weights_from(m)
        ref.prefill(ids)
        # Only the layers that own a cache. The 20 from FIRST_KV_SHARED up
        # hold `None` rather than an array of zeros -- see
        # `Gemma4Prefill.__init__` for why.
        for L in range(min(args.n_layers, config.FIRST_KV_SHARED)):
            dk = np.abs(m.kv_k[L] - ref.kv_k[L]).max()
            dv = np.abs(m.kv_v[L] - ref.kv_v[L]).max()
            print(f"[compare] layer {L:2d}  dK {dk:.4f}  dV {dv:.4f}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
