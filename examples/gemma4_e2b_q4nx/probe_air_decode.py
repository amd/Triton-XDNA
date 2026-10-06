# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""mlir-air's Gemma4 NPU decode on its own, with nothing of ours in the path.

Loads mlir-air's driver unmodified, against the decode template
`make compile-decode` built, and runs it two ways in this order:

  bare    the decoder alone, from an empty cache at position 0. No prefill
          and no `seed_kv`: only mlir-air's decoder and the driver.
  seeded  mlir-air's own `generate` with its CPU reference prefill, so its
          own `seed_kv` fills the cache before the first dispatch.

Our Triton prefill and decoder adapter appear in neither.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

sys.path.insert(0, config._SHARED)

import harness  # noqa: E402
import registry  # noqa: E402


def main():
    air = harness.air_inference_module(registry.spec("gemma4-e2b"))
    ids = list(config.PROMPT)

    print("[probe] bare: FusedDecoder from an empty cache", flush=True)
    dec = air.FusedDecoder(model=config.MODEL_DEFAULT, max_L=8)
    try:
        for p, tok in enumerate(ids[:3]):
            t0 = time.perf_counter()
            pred = int(dec.dispatch(tok, p).argmax())
            ms = (time.perf_counter() - t0) * 1e3
            print(f"[probe] bare: pos{p} ok ({ms:.0f} ms) -> {pred}", flush=True)
    finally:
        dec.close()

    print("[probe] seeded: mlir-air generate, numpy prefill", flush=True)
    gen, stop = air.generate(ids, len(config.EXPECT_IDS), numpy_prefill=True)
    got = list(gen) + ([stop] if stop is not None else [])
    ok = got == config.EXPECT_IDS
    print(
        f"[probe] seeded: {got} want {config.EXPECT_IDS} "
        f"{'ok' if ok else 'MISMATCH'}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
