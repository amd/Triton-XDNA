# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared implementation of each Q4NX example's `prefill.py`.

    python prefill.py --backend cpu
    python prefill.py --backend npu --ops all
    python prefill.py --backend npu --ops matmul --compare-cpu
    python prefill.py --backend npu --kv-out /tmp/prefill_kv.npz

Runs the Triton prefill alone and gates the canonical prompt's first token.
`--compare-cpu` reports the per-layer KV difference from the CPU reference on
the same weights; `--kv-out` writes the handoff as an npz. The weights are not
made NPU-resident, so the CPU reference can share them.
"""

import argparse
import time

import numpy as np
import torch


def main(prefill_cls, config, doc=None, argv=None):
    """Run `prefill_cls` for the example whose `config` module is bound.

    `--backend` choices are the rows of the class's `PLACEMENT`, and `--ops`
    names its `NPU_OPS`. `--kv-out` is offered only where `SAVES_KV_NPZ`.
    """
    ap = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", choices=tuple(prefill_cls.PLACEMENT), default="npu")
    ap.add_argument(
        "--ops",
        default=None,
        help="NPU ops to enable: 'all', or a comma list of "
        f"{','.join(prefill_cls.NPU_OPS)} (default: the model's own, which is "
        "not necessarily all -- see its `DEFAULT_OPS`)",
    )
    ap.add_argument("--prompt", default=None, help="token ids, comma separated")
    ap.add_argument("--n-layers", type=int, default=config.N_LAYERS)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--model", default=None)
    if prefill_cls.SAVES_KV_NPZ:
        ap.add_argument(
            "--kv-out", default=None, help="write the decode handoff npz here"
        )
    ap.add_argument(
        "--compare-cpu",
        action="store_true",
        help="also run the CPU reference and report per-layer KV divergence",
    )
    args = ap.parse_args(argv)

    ids = (
        [int(t) for t in args.prompt.split(",")] if args.prompt else list(config.PROMPT)
    )

    m = prefill_cls(
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
        ref = prefill_cls(backend="cpu", n_layers=args.n_layers, max_seq=args.max_seq)
        # Shares everything the class lists in `WEIGHT_ATTRS`.
        ref.share_weights_from(m)
        ref.prefill(ids)
        # Only layers that own a cache: from `FIRST_KV_SHARED` up (Gemma4) a
        # layer reads a lower layer's. `kv_view` returns the prompt's rows as
        # [P, dh] whatever the cache layout; a bf16 cache bottoms out at one
        # ulp rather than zero.
        for L in range(min(args.n_layers, getattr(config, "FIRST_KV_SHARED", 1 << 30))):
            gk, gv = m.kv_view(L)
            rk, rv = ref.kv_view(L)
            dk = np.abs(np.asarray(gk) - np.asarray(rk)).max()
            dv = np.abs(np.asarray(gv) - np.asarray(rv)).max()
            print(f"[compare] layer {L:2d}  dK {dk:.4f}  dV {dv:.4f}")

    if getattr(args, "kv_out", None):
        K, V = m.save_kv_npz(args.kv_out, first, ids)
        print(f"[prefill] wrote {args.kv_out}  k{K.shape} v{V.shape} first={first}")

    return 0 if verdict == "PASS" else 1
