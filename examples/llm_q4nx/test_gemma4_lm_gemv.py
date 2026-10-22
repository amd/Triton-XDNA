#!/usr/bin/env python3
"""Standalone check of the Triton LM head (`gemma4_lm_gemv`) against numpy.

Builds the LM head the way `gemma4_triton_fused.build` does and runs it the
way mlir-air's runtime runs an `lm` build: weights packed by `lm_gemv.pack`, x
bf16, logits read back by `lm_gemv.unpack_y`. A change on either side of that
interface shows up here rather than as wrong tokens.

Not an example: `scripts/run_tests.py` runs every *.py directly under an
example directory, and `llm_q4nx` is excluded from the sweep as a library.
CI runs it from the test action.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import airsrc  # noqa: E402
import gemma4_triton_fused as T  # noqa: E402

airsrc.add_air_paths("gemma4_e2b_q4nx")
from air_examples.llms.shared.fused_prefill import lm_gemv as G  # noqa: E402


def main():
    import pyxrt as xrt
    from air_examples.llms.shared.fused_prefill.engine import FROM, TO, Ctx, view

    rng = np.random.default_rng(0)
    q = rng.integers(0, 16, (G.V, G.K)).astype(np.uint8)
    sc = (rng.standard_normal((G.V, G.K // 32)) * 0.02).astype(np.float32)
    mn = (rng.standard_normal((G.V, G.K // 32)) * 0.05).astype(np.float32)
    x = rng.standard_normal(G.K).astype(bfloat16)
    s32 = np.repeat(sc.astype(bfloat16).astype(np.float32), 32, 1)
    m32 = np.repeat(mn.astype(bfloat16).astype(np.float32), 32, 1)
    ref = (q.astype(np.float32) * s32 + m32) @ x.astype(np.float32)

    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        cwd = os.getcwd()
        try:
            T._build_lm(out, G)
        finally:
            os.chdir(cwd)
        ctx = Ctx(xrt.device(0), out / "lm.xclbin")
        w = G.pack(q, sc, mn)
        bw = ctx.bo(w.nbytes, 0)
        bw.write(w.tobytes(), 0)
        bw.sync(TO)
        bx, by = ctx.bo(G.K * 2, 1), ctx.bo(G.V * 2, 2)
        ins = np.fromfile(out / "lm.insts.bin", np.uint32)
        ib, _ = ctx.insts(ins)
        run = ctx.run(ib, len(ins), [bw, bx, by])
        np.copyto(view(bx, bfloat16, (G.K,)), x)
        bx.sync(TO)
        run.r.start()
        st = run.r.wait()
        assert "COMPLETED" in str(st), st
        by.sync(FROM)
        got = G.unpack_y(view(by, bfloat16, (G.V,)).astype(np.float32))
        ctx.close()

    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"V={G.V}: relerr {rel:.5f}, argmax {got.argmax()} (ref {ref.argmax()})")
    assert rel < 1e-2 and got.argmax() == ref.argmax()
    print("PASS")


if __name__ == "__main__":
    main()
