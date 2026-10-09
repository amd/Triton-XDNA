#!/usr/bin/env python3
"""Check the host attention body against the whole-prompt form it replaced.

`LlamaPrefill._attention` scores a chunk of queries at a time against only the
keys that chunk's mask can let through. That is meant to be the same arithmetic
on fewer elements -- masked scores are -inf, and -inf contributes exactly zero
to both the softmax denominator and `p @ v` -- so the reference here is the
whole-prompt `[n_q, N, N]` body, spelled out below as it used to read.

The bound is where this goes wrong, and quietly. Taking the window back from
the chunk's LAST row rather than its first drops keys the earliest rows of each
chunk can see: order 1e0 off the answer at a long prompt, and a division by
zero once the window is narrower than the chunk, because those rows lose their
own diagonal. Both are among the cases below; both were live before this test.

The two bodies are NOT bit-identical and are not asked to be -- summing a row
in one piece and in several is the same arithmetic in a different order. So the
reference is float64 and what is checked is that the chunked body sits no
further from it than the whole-prompt body does. That separates rounding from a
dropped key by orders of magnitude, rather than by a threshold chosen to fit.

Needs no hardware and no weights: random tensors through both bodies.
"""

from __future__ import annotations

import contextlib
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# `llama_prefill` binds a model's dims at import, from whichever example's
# `config` is on the path. Any of them will do -- `_attention` reads none of
# them, it is parameterized on the head counts its caller passes.
sys.path.insert(0, os.path.join(_HERE, "..", "gemma4_e2b_q4nx"))

#: (N, n_q, n_kv, dh, window). Gemma4's two layer shapes at a real prompt
#: length, then the narrow windows where a chunk can outrun the window.
CASES = (
    (6, 8, 1, 256, 512),
    (37, 8, 1, 256, None),
    (163, 8, 1, 512, None),
    (700, 8, 1, 256, 512),  # first prompt past the window
    (2040, 8, 1, 256, 512),  # Gemma4 sliding, full length
    (2040, 8, 1, 512, None),  # Gemma4 full attention, full length
    (129, 8, 1, 256, 64),  # window under one chunk
    (300, 4, 2, 64, 8),  # GQA, and a window far under one chunk
    (300, 8, 1, 256, 1),  # only the diagonal survives
)

#: How much further from the float64 answer the chunked body may sit than the
#: whole-prompt body does. Both round; the question is only whether one of them
#: is also dropping keys, which costs orders of magnitude, not a factor of ten.
TOLERANCE_FACTOR = 10.0


def whole_prompt(q, k, v, n_q, n_kv, dh, window, scale):
    """The body this replaced: score everything, then mask.

    In whatever dtype it is handed, so the same code serves as the f32 form
    that shipped and as the f64 reference.
    """
    N = q.shape[0]
    rep = n_q // n_kv
    qh = q.reshape(N, n_q, dh).transpose(0, 1)
    kh = k.reshape(N, n_kv, dh).transpose(0, 1).repeat_interleave(rep, 0)
    vh = v.reshape(N, n_kv, dh).transpose(0, 1).repeat_interleave(rep, 0)
    scores = (qh @ kh.transpose(1, 2)) * scale
    inf = torch.full((N, N), float("-inf"), dtype=q.dtype)
    mask = inf.triu(1)
    if window is not None:
        mask = mask + inf.tril(-window)
    p = torch.softmax(scores + mask, dim=-1)
    return (p @ vh).transpose(0, 1).reshape(N, n_q * dh)


class _Bare:
    """Enough of a prefill for `_attention`: a timer, a backend, no device."""

    backend = "cpu"

    class _Timer:
        def track(self, _name):
            return contextlib.nullcontext()

    timer = _Timer()

    def _device(self, op, backend=None):
        return "cpu"


def compare(got, q, k, v, n_q, n_kv, dh, window):
    """(pass, chunked error, whole-prompt error), each against float64."""
    if bool(torch.isnan(got).any()):
        return False, float("nan"), 0.0
    ref = whole_prompt(q.double(), k.double(), v.double(), n_q, n_kv, dh, window, 1.0)
    scale = ref.abs().max().item() + 1e-30
    mine = (got.double() - ref).abs().max().item() / scale
    whole = whole_prompt(q, k, v, n_q, n_kv, dh, window, 1.0)
    theirs = (whole.double() - ref).abs().max().item() / scale
    return mine <= max(theirs * TOLERANCE_FACTOR, 1e-12), mine, theirs


def main():
    from llama_prefill import LlamaPrefill

    bare = _Bare()
    bare.ATTN_CHUNK = LlamaPrefill.ATTN_CHUNK
    attention = LlamaPrefill._attention
    torch.manual_seed(0)
    ok = True

    print(f"error against float64, chunk={bare.ATTN_CHUNK}")
    print(f"  {'case':<36} {'chunked':>9} {'whole':>9}")
    for N, n_q, n_kv, dh, window in CASES:
        q = torch.randn(N, n_q * dh)
        k = torch.randn(N, n_kv * dh)
        v = torch.randn(N, n_kv * dh)
        got = attention(bare, q, k, v, n_q, n_kv, dh, window, 1.0)
        good, mine, theirs = compare(got, q, k, v, n_q, n_kv, dh, window)
        ok &= good
        case = f"N={N} n_q={n_q} dh={dh} window={window}"
        print(f"  {case:<36} {mine:>9.2e} {theirs:>9.2e}  {'ok' if good else 'FAIL'}")

    # The chunk size is a tiling choice, so the answer must not depend on it.
    # A bound that is right at 128 and wrong at 32 is one that happens to fit
    # this model's window rather than one that follows from the mask.
    print("\n  chunk-size invariance, N=600 dh=256 window=512")
    q = torch.randn(600, 8 * 256)
    k = torch.randn(600, 256)
    v = torch.randn(600, 256)
    for chunk in (1, 7, 32, 128, 512, 4096):
        bare.ATTN_CHUNK = chunk
        got = attention(bare, q, k, v, 8, 1, 256, 512, 1.0)
        good, mine, theirs = compare(got, q, k, v, 8, 1, 256, 512)
        ok &= good
        label = f"chunk={chunk}"
        print(f"  {label:<36} {mine:>9.2e} {theirs:>9.2e}  {'ok' if good else 'FAIL'}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
