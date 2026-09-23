#!/usr/bin/env python3
"""Standalone check of `fused_mlp.FusedMLP` against a torch reference.

Runs the chain at Gemma4's two FFN widths and at a prompt length that needs
more than one row tile, and asserts three things the chain could get wrong
without changing the shape of the answer:

1. the result matches `down(gelu_tanh(gate(h)) * up(h))` in bf16-in/f32-out
   arithmetic, which is what the device computes;
2. a second dispatch for the same `bo_key` matches the first -- the operands
   the chain declares as `static_indices` are staged once, and an intermediate
   left dirty between calls would show up here and nowhere else;
3. two layers on the one chain do not bleed into each other, which is what
   `bo_key` is separating.

Not an example: `scripts/run_tests.py` runs every *.py directly under an
example directory, and `llm_q4nx` is excluded from the sweep as a library.
Invoke it by hand.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_mlp import FusedMLP  # noqa: E402

D = 1536  # Gemma4's model dim


def reference(h, gate_up, down, inter):
    """What the device computes, in the device's precision.

    bf16 operands with f32 accumulation, and the merge rounded to bf16 before
    `down` -- rounding here rather than keeping f32 throughout is not cosmetic:
    it is the same source of error the chain has, so a mismatch means the chain
    is wrong rather than merely less precise.
    """
    a = h.to(torch.bfloat16).to(torch.float32)
    gu = gate_up.to(torch.bfloat16).to(torch.float32)
    g = a @ gu[:, :inter]
    u = a @ gu[:, inter:]
    merged = (torch.nn.functional.gelu(g, approximate="tanh") * u).to(torch.bfloat16)
    return merged.to(torch.float32) @ down.to(torch.bfloat16).to(torch.float32)


def check(inter, n_rows, layers=(0, 1)):
    torch.manual_seed(0)
    print(f"\n=== inter={inter}  N={n_rows}  layers={list(layers)} ===", flush=True)

    mlp = FusedMLP(D, inter)
    w = {}
    for L in layers:
        gate_up = (torch.randn(D, 2 * inter) * 0.05).to(torch.bfloat16)
        down = (torch.randn(inter, D) * 0.05).to(torch.bfloat16)
        w[L] = (gate_up, down)
        mlp.add_layer(L, gate_up, down)

    h = (torch.randn(n_rows, D) * 0.5).to(torch.float32)
    ok = True

    firsts = {}
    for L in layers:
        got = mlp.run(L, h)
        ref = reference(h, *w[L], inter)
        scale = ref.abs().max().item()
        err = (got - ref).abs().max().item()
        rel = err / scale
        print(f"  L{L} vs torch      : max|err| {err:.4f}  rel {rel:.5f}")
        ok &= rel < 0.05
        firsts[L] = got

    # (2) repeat determinism, and (3) no cross-layer bleed -- run every layer a
    # second time, after the others have used the chain.
    for L in layers:
        again = mlp.run(L, h)
        drift = (again - firsts[L]).abs().max().item()
        print(f"  L{L} repeat drift   : {drift}   (must be 0.0)")
        ok &= drift == 0.0

    mlp.close()
    print("  ->", "PASS" if ok else "FAIL")
    return ok


def main():
    ok = True
    # One row tile, then more than one, at both of Gemma4's FFN widths.
    ok &= check(6144, 6)
    ok &= check(6144, 163)
    ok &= check(12288, 6)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
