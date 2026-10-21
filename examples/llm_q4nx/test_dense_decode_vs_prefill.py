#!/usr/bin/env python3
"""Does the iGPU decode predict what a prefill over the same prefix predicts?

`test_decode_vs_prefill.py` for the Llama-family models (`LlamaGpuDecode`).
The oracle is the host prefill, re-run over the growing prefix a token at a
time. Against it, for each prompt:

* host prefill, iGPU decode: the prefill bound to the decoder's shared cache
  (`kv_sink`), the path an NPU prefill takes, then decoded in place;
* iGPU prefill, iGPU decode: `LlamaGpuDecode.prefill`, whose cache rows must
  also agree with the host prefill's. Its int8 activations make them close
  rather than equal, so tokens and rows are the checks, not logits.

Run it by hand from this directory, naming the model's example:

    python test_dense_decode_vs_prefill.py --example llama32_1b_q4nx \
        [--tokens 4] [--backend npu] [--long 600]

Exits 77 without an iGPU, or when the weights or mlir-air's sources cannot
be had.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="llama32_1b_q4nx")
    ap.add_argument("--tokens", type=int, default=4)
    # The host forward is the reference the NPU kernels are themselves held
    # to; an NPU prefill as the oracle would grade the iGPU against its
    # numerics instead.
    ap.add_argument("--backend", default="cpu", choices=("npu", "hetero", "cpu"))
    ap.add_argument(
        "--long",
        type=int,
        default=0,
        help="also run a prompt this long, e.g. past a sliding window. The "
        "oracle re-prefills per token, so this is slow.",
    )
    a = ap.parse_args()

    sys.path.insert(0, _HERE)
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), a.example))

    import harness

    # XRT before ROCm; see `harness.hold_npu`. Only the NPU prefill needs it.
    if a.backend != "cpu":
        harness.hold_npu()

    if not torch.cuda.is_available():
        print("SKIP: no ROCm device; the decode under test is the GPU one")
        return 77

    import gpu_kernels  # noqa: F401 -- Triton before any torch matmul

    import airsrc
    import config as cfg
    import registry
    from dense_kv_layout import K_REGION, V_REGION, gather
    from llama_prefill import LlamaGpuDecode

    try:
        airsrc.air_llms_root()
    except RuntimeError as e:
        print(f"SKIP: {e}")
        return harness.SKIP_EXIT_CODE

    prefill_cls = _prefill_cls(a.example)
    m = prefill_cls(backend=a.backend, n_layers=cfg.N_LAYERS, max_seq=2048)
    try:
        harness.load_prefill_weights(m, a.backend, npu_resident=False)
        dec = LlamaGpuDecode(m, max_L=2048)
    except Exception as e:  # noqa: BLE001 -- as harness.run_prefill
        print(f"SKIP: cannot load the q4nx weights: {e}")
        return harness.SKIP_EXIT_CODE
    assert registry.spec(cfg.MODEL_NAME).supports_igpu

    base = list(cfg.PROMPT)
    prompts = [base, base[:-1], base + base[1:]]
    if a.long:
        prompts.append(
            [base[0]] + [base[1 + i % (len(base) - 1)] for i in range(a.long)]
        )

    def rows(P):
        return torch.cat(
            [
                gather(dec._slab_t, L, r, dec.geom, P).float().cpu()
                for L in range(dec.n_layers)
                for r in (K_REGION, V_REGION)
            ]
        )

    ok = True
    for ids in prompts:
        P = len(ids)
        m.bind_kv_sink(None)
        seq, oracle, scores = list(ids), [], []
        for _ in range(a.tokens):
            m.clear_context()
            scores.append(torch.as_tensor(m.prefill(seq)).float())
            oracle.append(int(scores[-1].argmax()))
            seq.append(oracle[-1])
        print(f"prompt (P={P}) {ids if P <= 8 else ids[:8] + ['...']}")
        print(f"  prefill oracle : {oracle}")

        host_kv = None
        try:
            sink = dec.kv_sink()
        except RuntimeError as e:
            print(f"  host -> gpu    : SKIP, {e}")
        else:
            dec._slab_t.zero_()
            m.bind_kv_sink(sink)
            m.clear_context()
            first = int(torch.as_tensor(m.prefill(ids)).argmax())
            host_kv = rows(P)
            got = dec.generate(first, n_tokens=a.tokens, eos=())
            good, note = _agrees(got, oracle, scores)
            ok &= good
            print(f"  host -> gpu    : {got}{note}   {'ok' if good else 'MISMATCH'}")
            m.bind_kv_sink(None)

        dec.prefill(ids)
        got = dec.generate(n_tokens=a.tokens, eos=())
        good, note = _agrees(got, oracle, scores)
        if host_kv is not None:
            kv = rows(P)
            diff = ((kv - host_kv).norm() / host_kv.norm()).item()
            good &= diff < 5e-2
            note += f"   KV rel {diff:.1e}"
        ok &= good
        print(f"  gpu prefill    : {got}{note}   {'ok' if good else 'MISMATCH'}")

    # A prompt longer than the cache must be refused before it writes anything.
    try:
        dec.prefill([base[0]] * (dec.max_L + 1))
        print("prompt past the cache: accepted   FAIL")
        ok = False
    except ValueError:
        print("prompt past the cache: refused   ok")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _agrees(got, oracle, scores):
    """Whether `got` is a greedy continuation the oracle's own logits allow.

    Where the oracle's logits tie exactly between its token and `got`'s, both
    are greedy, and the sequences part there; they are compared up to it.
    """
    for i, (g, o) in enumerate(zip(got, oracle)):
        if g != o:
            if scores[i][g] == scores[i][o]:
                return True, f"   (oracle ties {o} and {g} at step {i})"
            return False, ""
    return len(got) == len(oracle), ""


def _prefill_cls(example):
    """The prefill class the example's entry point hands the harness."""
    import importlib.util

    path = os.path.join(os.path.dirname(_HERE), example, "prefill.py")
    spec = importlib.util.spec_from_file_location("example_prefill", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (cls,) = {
        v for k, v in vars(mod).items() if isinstance(v, type) and k.endswith("Prefill")
    }
    return cls


if __name__ == "__main__":
    sys.exit(main())
