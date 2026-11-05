#!/usr/bin/env python3
"""Gate `Gemma4Prefill.prefill(..., reuse=True)` against a full prefill.

    python test_prefix_reuse.py
    python test_prefix_reuse.py --backend npu

Prefix reuse recomputes only the rows a prompt does not share with the
previous one. Every operator in a block is row-wise except attention, which
reads the whole prompt's keys from the dense f32 cache, so the recomputed
rows should match a full prefill bit for bit.

Failures this catches (none of them raise):

* the RoPE table sliced from 0 instead of `p0`;
* the attention mask anchored on the diagonal instead of offset by `p0`;
* KV written at row 0 instead of row `p0`.

The check is exact equality of the logits and the KV cache. Prefix rows are
not recomputed by reuse, so they are compared straight from the cache.

Exits 77 (graded as a skip) when the weights cannot be loaded.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SKIP_EXIT_CODE = 77

#: A shared leading run (standing in for a cached system prompt), then two
#: diverging tails.
PREFIX = [2, 818, 5279, 529, 7001, 563, 9079, 236761, 818, 5279]
#: The tails must differ at their first token, or the common prefix is longer
#: than `PREFIX` and the expected-prefix check fails.
TAIL_A = [7001, 563]
TAIL_B = [12811, 563, 1018, 84750]


def _max_abs(a, b):
    return float(torch.as_tensor(np.abs(np.asarray(a) - np.asarray(b))).max())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("cpu", "npu", "hetero"), default="cpu")
    ap.add_argument("--ops", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-seq", type=int, default=256)
    args = ap.parse_args(argv)

    # `config` lives in the model's example directory, not the shared harness.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "gemma4_e2b_q4nx"))
    import config  # noqa: E402

    from gemma4_prefill import Gemma4Prefill  # noqa: E402

    def build():
        m = Gemma4Prefill(
            backend=args.backend,
            ops=args.ops,
            n_layers=config.N_LAYERS,
            max_seq=args.max_seq,
            model=args.model,
        )
        m.load_weights(args.model)
        m.npu_attention = False
        return m

    try:
        m = build()
    except Exception as e:  # noqa: BLE001 -- no weights here, not a failure
        print(f"[skip] cannot load the q4nx weights: {type(e).__name__}: {e}")
        return SKIP_EXIT_CODE

    # Reuse runs only where a full pass runs the same host attention as a
    # reusing one; with the NPU attention on, it must be refused.
    if m._device("attention") == "npu":
        m.npu_attention = True
        refused = not m._host_attention()
        m.npu_attention = False
        if not refused:
            print("[reuse] FAIL -- reuse allowed with the NPU attention on")
            return 1
    if not m._host_attention():
        print("[reuse] FAIL -- reuse refused with the host attention")
        return 1

    prompt_a = PREFIX + TAIL_A
    prompt_b = PREFIX + TAIL_B
    print(
        f"[reuse] backend={args.backend} A={len(prompt_a)} tokens, "
        f"B={len(prompt_b)} tokens, shared prefix {len(PREFIX)}"
    )

    # Reference: B prefilled from an empty context.
    m.clear_context()
    full = m.prefill(prompt_b)
    full_kv = [tuple(np.array(t, copy=True) for t in m.kv_view(L)) for L in range(6)]

    # Under test: A, then B with reuse.
    m.clear_context()
    m.prefill(prompt_a)
    p0 = m._reusable_prefix(prompt_b)
    reused = m.prefill(prompt_b, reuse=True)
    reused_kv = [tuple(np.array(t, copy=True) for t in m.kv_view(L)) for L in range(6)]

    print(
        f"[reuse] reused {p0} of {len(prompt_b)} rows, recomputed {len(prompt_b) - p0}"
    )
    bad = []
    if p0 != len(PREFIX):
        # The prompts share exactly PREFIX; anything else is a prefix-search
        # bug.
        bad.append(f"prefix {p0}, expected {len(PREFIX)}")

    d = _max_abs(full, reused)
    ok = d == 0.0
    print(f"  logits      max|delta| {d:.3e}  {'ok' if ok else 'FAIL'}")
    if not ok:
        bad.append("logits differ")

    print(f"  argmax      full {int(full.argmax())} reused {int(reused.argmax())}")
    if int(full.argmax()) != int(reused.argmax()):
        bad.append("argmax differs")

    for L, ((fk, fv), (rk, rv)) in enumerate(zip(full_kv, reused_kv)):
        dk, dv = _max_abs(fk, rk), _max_abs(fv, rv)
        ok = dk == 0.0 and dv == 0.0
        print(f"  layer {L:2d} KV  dK {dk:.3e}  dV {dv:.3e}  {'ok' if ok else 'FAIL'}")
        if not ok:
            bad.append(f"layer {L} KV differs")

    if bad:
        print(f"[reuse] FAIL: {'; '.join(bad)}")
        return 1
    print("[reuse] PASS -- reuse is exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
