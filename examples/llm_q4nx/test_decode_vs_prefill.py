#!/usr/bin/env python3
"""Does a decode step predict what a prefill over the same prefix predicts?

That equivalence is what a correct KV handoff means, and nothing tested it. The
decode gate compares against `EXPECT_IDS`, a short continuation recorded for
one prompt, so it catches a decode broken everywhere and misses one broken
anywhere else.

The oracle is the prefill itself, re-run over the growing prefix a token at a
time. Slow, since that is N prefills for N tokens, and trustworthy, since the
prefill is what CI already gates.

It also runs the iGPU prefill (`Gemma4GpuDecode.prefill`) over each prompt
and checks that the decode continues from it as from the host prefill: the
same greedy tokens, and K/V in the same slab rows. Its int8 activations make
its logits close to the host prefill's rather than equal, so those are the
checks, not the logits.

Run it by hand; `llm_q4nx` is excluded from the `run_tests.py` sweep as a
library. Exits 77 without an iGPU, the decode under test being the GPU one.

    python test_decode_vs_prefill.py [--tokens 4] [--long]

It reports a disagreement between the two decoders. The GPU decode matches the
oracle on every prompt tried; mlir-air's fused NPU decode matches on the
recorded prompt and diverges at the first decode step on others, which the
existing gate cannot see because it only runs the recorded prompt. Whether the
cause is the template's build configuration, the KV seeded into
`air.generate`, or the superkernel is not isolated here.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
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


#: Long enough to put whole key blocks outside the sliding window, which none
#: of the prompts above reach. That is where `attn_prefill` produced NaN until
#: the running maximum was guarded. Opt-in: the oracle re-prefills per token,
#: so this costs minutes.
LONG_PROMPT_TOKENS = 600


def _long_prompt(n):
    base = [818, 5279, 529, 7001, 563]
    return [2] + [base[i % 5] for i in range(n - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--backend", default="npu", choices=("npu", "hetero", "cpu"))
    ap.add_argument(
        "--long",
        action="store_true",
        help=f"also run a {LONG_PROMPT_TOKENS}-token prompt, past the sliding "
        "window. Minutes, not seconds -- the oracle re-prefills per token.",
    )
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

    prompts = list(PROMPTS)
    if a.long:
        prompts.append(_long_prompt(LONG_PROMPT_TOKENS))

    ok = True
    for ids in prompts:
        seq, oracle = list(ids), []
        for _ in range(a.tokens):
            t = int(torch.as_tensor(m.prefill(seq)).argmax())
            oracle.append(t)
            seq.append(t)

        logits = torch.as_tensor(m.prefill(ids))
        assert not torch.isnan(logits).any(), f"prefill produced NaN at P={len(ids)}"
        dec = Gemma4GpuDecode(m, max_L=len(ids) + a.tokens + 2)
        host_kv = dec._slab_t.clone()  # the host prefill's K/V, before decoding
        # No `first`: the decoder takes the prefiller's own last prediction and
        # context length, so it decodes the prefill that actually ran. The
        # oracle above left the KV on an extended prefix; passing a token from
        # an earlier call would answer for neither.
        #
        # `eos=()` so the comparison runs to `--tokens` regardless of where the
        # model would have stopped; the oracle does not stop either.
        got = dec.generate(n_tokens=a.tokens, eos=())[: len(oracle)]

        match = got == oracle
        ok &= match
        print(f"prompt (P={len(ids)}) {ids if len(ids) <= 8 else ids[:8] + ['...']}")
        print(f"  prefill oracle : {oracle}")
        print(f"  gpu decode     : {got}   {'ok' if match else 'MISMATCH'}")

        dec.prefill(ids)
        kv = dec._slab_t
        diff = (kv.float() - host_kv.float()).norm() / host_kv.float().norm()
        lanes = ((kv != 0) == (host_kv != 0)).float().mean().item() > 0.9999
        mirrored = dec._slab_shared or torch.equal(
            torch.from_numpy(m._kv_host.view(np.int16)), kv.view(torch.int16).cpu()
        )
        got = dec.generate(n_tokens=a.tokens, eos=())[: len(oracle)]
        good = got == oracle and lanes and diff.item() < 5e-2 and mirrored
        ok &= good
        print(
            f"  gpu prefill    : {got}   KV rel {diff.item():.1e}, same lanes "
            f"{lanes}, host slab in sync {mirrored}   {'ok' if good else 'MISMATCH'}"
        )

    # A prompt longer than the slab or the RoPE tables (`max_seq`) must be
    # refused before it writes anything.
    try:
        dec.prefill([2] * (m.max_seq + 1))
        print("prompt past max_seq: accepted   FAIL")
        ok = False
    except ValueError:
        print("prompt past max_seq: refused   ok")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
