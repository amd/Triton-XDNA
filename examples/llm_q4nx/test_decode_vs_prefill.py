#!/usr/bin/env python3
"""Does a decode step predict what a prefill over the same prefix predicts?

That equivalence is the definition of a correct KV handoff, and nothing here
tested it. The decode gate compares against `EXPECT_IDS`, a two-token
continuation recorded for one prompt -- which catches a decode that is broken
everywhere and misses one that is broken anywhere else.

The oracle is the prefill: re-run it over the growing prefix, one token at a
time, and compare. Slow by construction -- N prefills for N tokens -- and
unimpeachable, because the prefill is the path CI already gates.

Run it by hand; `llm_q4nx` is excluded from the `run_tests.py` sweep as a
library. Exits 77 -- graded as a skip -- without an iGPU, since the decode
under test is the GPU one.

    python test_decode_vs_prefill.py [--tokens 4] [--backend npu]

What it found when it was written
---------------------------------
The GPU decode matches the oracle on every prompt tried. mlir-air's fused NPU
decode matches on the recorded prompt and diverges from the oracle at the first
decode step on two others -- which the existing gate cannot see, because it
only ever runs the recorded prompt. Whether that is the template's build
configuration, the KV seeded into `air.generate`, or the superkernel itself is
not isolated here; this test establishes only that the two decoders disagree
and which one tracks the prefill.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
#: Gemma4-specific, so it reaches into that example for `config`. This
#: directory is a library and has none of its own -- every model keeps its
#: dimensions beside its driver, which is what lets one harness serve ten.
_EXAMPLE = os.path.join(os.path.dirname(_HERE), "gemma4_e2b_q4nx")
sys.path.insert(0, _HERE)
sys.path.insert(0, _EXAMPLE)

#: Short prompts, because the oracle costs a full prefill per token. The first
#: is the one `EXPECT_IDS` records -- kept so a regression on the gated path
#: shows up here too rather than only in CI.
PROMPTS = [
    [2, 818, 5279, 529, 7001, 563],
    [2, 818, 5279, 529, 15636, 563],
    [2, 818, 2707, 529],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--backend", default="npu", choices=("npu", "hetero", "cpu"))
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: no ROCm device; the decode under test is the GPU one")
        return 77

    import config as cfg
    from gemma4_prefill import Gemma4GpuDecode, Gemma4Prefill
    from harness import load_prefill_weights

    m = Gemma4Prefill(backend=a.backend, n_layers=cfg.N_LAYERS, max_seq=2048)
    # `npu_resident=False`: the padded resident form is one-way and a decode
    # that then multiplies with it raises. See `Gemma4GpuDecode`.
    load_prefill_weights(m, a.backend, npu_resident=False)

    ok = True
    for ids in PROMPTS:
        seq, oracle = list(ids), []
        for _ in range(a.tokens):
            t = int(torch.as_tensor(m.prefill(seq)).argmax())
            oracle.append(t)
            seq.append(t)

        first = int(torch.as_tensor(m.prefill(ids)).argmax())
        dec = Gemma4GpuDecode(m, max_L=len(ids) + a.tokens + 2)
        # `eos=()` so the comparison runs to `--tokens` regardless of where the
        # model would have stopped; the oracle does not stop either.
        got = dec.generate(first, a.tokens, eos=())[: len(oracle)]

        match = got == oracle
        ok &= match
        print(f"prompt {ids}")
        print(f"  prefill oracle : {oracle}")
        print(f"  gpu decode     : {got}   {'ok' if match else 'MISMATCH'}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
