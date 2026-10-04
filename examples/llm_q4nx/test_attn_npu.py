#!/usr/bin/env python3
"""Standalone check of `attn_npu.NPUAttention` against a torch reference.

Runs both of Gemma4's attention shapes -- sliding (head dim 256, window 512)
and full (head dim 512, causal) -- at prompt lengths below one query block, at
exactly one, and spanning several with a ragged last block, and asserts:

1. every block's mask admits exactly the keys causal attention does: key j is
   visible to query i iff j <= i, j < N and, under a window W, i - j < W. This
   is checked exactly, on the host, because it cannot be checked numerically:
   one key too many out of 512 moves the output far less than the GEMMs'
   block-float rounding does;
2. the result matches causal multi-query attention computed in the device's
   precision (bf16 Q, K, V and probabilities, f32 accumulation) to 3e-2. Every
   case sits at ~1.3e-2, the bfp16 MAC's own error; what this catches is the
   gross failure, e.g. a block of zero-padded keys admitted at score 0;
3. a length seen again after others have evicted its chains comes back
   identical: plans are rebuilt on demand, and a page left dirty between
   calls would show up here and nowhere else.

Not an example: `scripts/run_tests.py` runs every *.py directly under an
example directory, and `llm_q4nx` is excluded from the sweep as a library.
Invoke it by hand.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_npu  # noqa: E402

N_Q = 8


def reference(q, k, v, dh, window):
    """Causal MQA with bf16 operands and probabilities, f32 accumulation."""
    N = q.shape[0]
    bf = lambda t: t.to(torch.bfloat16).to(torch.float32)  # noqa: E731
    qh = bf(q).reshape(N, N_Q, dh).transpose(0, 1)
    s = qh @ bf(k).T
    i = torch.arange(N)[:, None]
    j = torch.arange(N)[None, :]
    keep = j <= i
    if window:
        keep &= i - j < window
    p = bf(torch.softmax(s.masked_fill(~keep, float("-inf")), -1))
    return (p @ bf(v)).transpose(0, 1).reshape(N, N_Q * dh)


def masks_exact(att, N):
    """Each block's additive mask against visibility computed directly."""
    _, _, spans = att._plan(-(-N // attn_npu.QB))
    for b, (mask, (lo, nc)) in enumerate(zip(att._block_masks(N, spans), spans)):
        i = b * attn_npu.QB + torch.arange(attn_npu.QB)[:, None]
        j = lo + torch.arange(nc)[None, :]
        visible = (j <= i) & (j >= 0) & (j < N)
        if att.window:
            visible &= i - j < att.window
        if not torch.equal(mask == 0, visible) or not bool(
            (mask[~visible] == float("-inf")).all()
        ):
            return False
        # ...and the span holds every key the row may see, not just the ones
        # it happens to cover: i + 1 of them, or W under a window.
        real = i[:, 0] < N
        want = i[:, 0] + 1 if not att.window else (i[:, 0] + 1).clamp(max=att.window)
        if not torch.equal(visible.sum(1)[real], want[real]):
            return False
    return True


def check(att, dh, window, N, seen):
    torch.manual_seed(N)
    q = torch.randn(N, N_Q * dh)
    k = torch.randn(N, dh) / dh**0.5
    v = torch.randn(N, dh)
    got = att(q, k, v, scale=1.0)
    ref = reference(q, k, v, dh, window)
    rel = ((got - ref).norm() / ref.norm()).item()
    exact = masks_exact(att, N)
    ok = bool(torch.isfinite(got).all()) and rel < 3e-2 and exact
    line = f"  dh={dh} window={window} N={N:5d}: masks {'exact' if exact else 'WRONG'}, rel {rel:.2e}"
    if N in seen:
        drift = (got - seen[N]).abs().max().item()
        line += f"   again after eviction: drift {drift}"
        ok &= drift == 0.0
    seen[N] = got
    print(line, "" if ok else "  <-- FAIL", flush=True)
    return ok


def main():
    ok = True
    # 600 comes back last: by then two later lengths have evicted its plan.
    lengths = (6, 512, 600, 1100, 2040, 600)
    for dh, window in ((256, 512), (512, None)):
        att = attn_npu.NPUAttention(N_Q, dh, window)
        seen = {}
        for N in lengths:
            ok &= check(att, dh, window, N, seen)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
