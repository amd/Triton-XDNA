#!/usr/bin/env python3
"""Gate `Gemma4Prefill.prefill(..., reuse=True)` against a full prefill.

    python test_prefix_reuse.py            # cpu, ~2 min including the load
    python test_prefix_reuse.py --backend npu

Prefix reuse recomputes only the rows a prompt does not share with the previous
one. The claim it makes is **exactness**, not approximation: every operator in a
block is row-wise except the attention, and the attention is handed the whole
prompt's keys out of the dense f32 cache, so the rows it does compute should
come out bit-for-bit as a full prefill would have computed them.

That claim is what this checks, and it is worth checking rather than reasoning
about, because all three ways it can break are silent:

* a RoPE table sliced from 0 instead of from `p0` -- every suffix row rotated
  as if it were at the start of the prompt;
* an attention mask still anchored on the diagonal -- suffix rows attending the
  wrong span;
* KV written at row 0 instead of row `p0` -- the cache overwritten with the
  suffix and the prefix lost.

None raises. All three produce fluent, wrong text.

So the test is equality of the logits and of the KV cache, not a tolerance on
either. The two paths run the same arithmetic in the same order on the rows
they share; the only licensed difference is that reuse never recomputed the
prefix rows, and those are compared straight out of the cache.

Exits 77 -- graded as a skip -- when the weights cannot be loaded.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SKIP_EXIT_CODE = 77

#: A prompt with an obvious seam: a shared leading run, then a divergence.
#: `PREFIX` stands in for the system prompt a production hybrid caches.
PREFIX = [2, 818, 5279, 529, 7001, 563, 9079, 236761, 818, 5279]
#: The tails must DIVERGE AT THEIR FIRST TOKEN. When this test was written both
#: began with 529, so the real common prefix was 11 rather than 10 and the
#: expected-prefix check failed for a reason that had nothing to do with the
#: code under test. The implementation was right and the constants were wrong.
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

    # After argv: importing `config` needs an example directory on sys.path,
    # and this file lives in the shared harness rather than in one of them.
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

    prompt_a = PREFIX + TAIL_A
    prompt_b = PREFIX + TAIL_B
    print(
        f"[reuse] backend={args.backend} A={len(prompt_a)} tokens, "
        f"B={len(prompt_b)} tokens, shared prefix {len(PREFIX)}"
    )

    # --- the reference: B computed from nothing ---
    m.clear_context()
    full = m.prefill(prompt_b)
    full_kv = [tuple(np.array(t, copy=True) for t in m.kv_view(L)) for L in range(6)]

    # --- the path under test: A, then B reusing what it can ---
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
        # Not a tolerance question: the two prompts share exactly PREFIX, so
        # anything else means the prefix search is wrong and the rest of the
        # comparison would be measuring the wrong thing.
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
