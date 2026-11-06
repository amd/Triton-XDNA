#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""A cache-only triton-fused prefill, finished by the NPU decoder, against the
whole prefill.

    python test_cache_only_prefill.py [--example gemma4_e4b_q4nx] [--tokens 16]

`Gemma4TritonFusedPrefill(cache_only=True)` runs only the layers that write the
KV cache and returns no logits. Prefilling all but the prompt's last token that
way, then running that token as the decoder's first step, must generate what the
whole prefill followed by the decoder generates. Also reports the weight bytes
the cache-only prefill leaves out.

Needs the NPU and the model's decode build (`make compile-decode` in the
example); exits 77 without them.
"""

import argparse
import functools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--example", default="gemma4_e4b_q4nx")
    ap.add_argument("--tokens", type=int, default=16)
    a = ap.parse_args()
    ex = os.path.join(os.path.dirname(HERE), a.example)
    sys.path[:0] = [HERE, ex]
    import config as cfg
    import gemma4_flash as F
    import harness
    import registry

    try:
        harness.hold_npu()
        spec = registry.spec(cfg.MODEL_NAME)
        air = harness.air_inference_module(spec)
        harness.decode_build.check_stamp(spec)
    except (harness.ExampleUnavailable, SystemExit, RuntimeError) as e:
        print(f"SKIP: {e}")
        return 77
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(harness.tokenizer_dir(air, spec))
    msg = [
        {
            "role": "user",
            "content": "What is the capital of France? Answer in one sentence.",
        }
    ]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, add_special_tokens=False)
    args = argparse.Namespace(prefill_engine="triton-fused", max_seq=512)
    cls = F._prefill_cls(args, cfg)

    def run(cache_only):
        make = functools.partial(cls, cache_only=True) if cache_only else cls
        m = harness.make_prefiller(make, cfg, "npu", None, 512, None)
        K, V = harness._install_shared_kv(air, m)
        dec = air.FusedDecoder(model=cfg.MODEL_DEFAULT, max_L=512)
        if cache_only:
            assert m.prefill(ids[:-1]) is None, "a cache-only prefill returned logits"
            dec.seed_kv(K, V, len(ids) - 1)
            out = [int(dec.dispatch(ids[-1], len(ids) - 1).argmax())]
        else:
            out = [int(m.prefill(ids).argmax())]
            dec.seed_kv(K, V, len(ids))
        for p in range(len(ids), len(ids) + a.tokens - 1):
            out.append(int(dec.dispatch(out[-1], p).argmax()))
        import numpy as np

        def nbytes(b):
            return int(np.prod(b.shape)) * np.dtype(b.np_dtype).itemsize

        f = m._fused
        w = sum(nbytes(s["w"]) for s in f.sites.values())
        w += sum(nbytes(b) for b in getattr(f, "lm_w", []))
        m.suspend()  # the second run needs the NPU contexts
        return out, w

    whole, w_whole = run(False)
    part, w_part = run(True)
    print(f"whole prefill + decoder:    {whole}")
    print(f"cache-only prefill + decoder: {part}")
    print(
        f"prefill weights on the NPU: {w_whole / 2**30:.2f} -> {w_part / 2**30:.2f} GiB"
    )
    if whole != part:
        at = next(i for i, (x, y) in enumerate(zip(whole, part)) if x != y)
        print(f"RESULT: FAIL (first difference at token {at})")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
