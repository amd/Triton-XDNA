#!/usr/bin/env python3
"""Standalone check of the Triton LM head (`gemma4_lm_gemv`) against numpy.

Builds the LM head the way `gemma4_triton_fused.build` does and calls it as
the prefill does, `gemv_q4[grid](...)` on a `FusedProgram`: weights packed by `gemma4_lm_gemv.pack`,
x bf16, logits read back by `gemma4_lm_gemv.unpack_y`. A change on either
side of that interface shows up here rather than as wrong tokens.

Not an example: `scripts/run_tests.py` runs every *.py directly under an
example directory, and `llm_q4nx` is excluded from the sweep as a library.
CI runs it from the test action.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_e2b_q4nx")]
import gemma4_lm_gemv as LG  # noqa: E402
import gemma4_triton_fused as T  # noqa: E402


def main():
    from triton.backends.amd_triton_npu.fused_program import FusedProgram

    rng = np.random.default_rng(0)
    q = rng.integers(0, 16, (LG.V, LG.K)).astype(np.uint8)
    sc = (rng.standard_normal((LG.V, LG.K // 32)) * 0.02).astype(np.float32)
    mn = (rng.standard_normal((LG.V, LG.K // 32)) * 0.05).astype(np.float32)
    x = rng.standard_normal(LG.K).astype(bfloat16)
    s32 = np.repeat(sc.astype(bfloat16).astype(np.float32), 32, 1)
    m32 = np.repeat(mn.astype(bfloat16).astype(np.float32), 32, 1)
    ref = (q.astype(np.float32) * s32 + m32) @ x.astype(np.float32)

    with tempfile.TemporaryDirectory() as d:
        T.build_lm(d)
        prog = FusedProgram(d, "lm")
        w = LG.pack(q, sc, mn)
        wb, xb, yb = (
            prog.empty(w.shape, np.uint8),
            prog.empty(LG.K, bfloat16),
            prog.empty(LG.V, bfloat16),
        )
        wb.host(write=True)[...] = w
        xb.host(write=True)[...] = x
        with prog:
            LG.gemv_q4[(1,)](*LG.args(wb, xb, yb))
        got = LG.unpack_y(yb.host().astype(np.float32))

    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"V={LG.V}: relerr {rel:.5f}, argmax {got.argmax()} (ref {ref.argmax()})")
    assert rel < 1e-2 and got.argmax() == ref.argmax()
    print("PASS")


if __name__ == "__main__":
    main()
