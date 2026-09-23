#!/usr/bin/env python3
"""Check `gpu_kernels` against torch, at the shapes Gemma4-E2B decode uses.

torch is the oracle here, which is the role `README.md` gives it. Every case is
a real shape from the model -- both head dims, both FFN widths, the sliding
window and a context short enough to exercise the `BLOCK_S` mask -- because the
kernels are written for N=1 and a shape they were not written for is exactly
what would slip through a generic test.

Not picked up by `scripts/run_tests.py`: `llm_q4nx` is excluded from the sweep
as a library, like `test_fused_mlp.py` and `test_decode_artifact.py`. Run it by
hand. Exits 77 -- graded as a skip -- with no iGPU.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = 1536
DH_SLIDING, DH_GLOBAL = 256, 512
N_Q_HEADS = 8
RMS_EPS = 1e-6


def rel(got, ref):
    got, ref = got.float(), ref.float()
    scale = ref.abs().max().item()
    return (got - ref).abs().max().item() / (scale if scale else 1.0)


def check(name, got, ref, tol=2e-2):
    r = rel(got, ref)
    ok = r < tol
    print(f"  {name:<34} rel {r:.2e}  {'ok' if ok else 'FAIL'}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("SKIP: no ROCm device visible to torch")
        return 77

    import gpu_kernels as G

    dev = "cuda"
    torch.manual_seed(0)
    ok = True
    print(f"device: {torch.cuda.get_device_name(0)}")

    # --- gemv, at every projection width the model actually uses ---
    print("\ngemv  [1,K] @ [K,N]")
    for K, N in (
        (D, N_Q_HEADS * DH_GLOBAL + 2 * DH_GLOBAL),  # qkv, full layer
        (D, N_Q_HEADS * DH_SLIDING + 2 * DH_SLIDING),  # qkv, sliding layer
        (D, 2 * 6144),  # gate_up, narrow FFN
        (D, 2 * 12288),  # gate_up, wide FFN
        (6144, D),  # down, narrow
        (D, 256),  # inp_gate / model_proj (PLE)
        (256, D),  # per_layer_projection
        (D, 262144),  # lm_head -- by far the widest
    ):
        x = torch.randn(1, K, device=dev)
        w = (torch.randn(K, N, device=dev) * 0.05).to(torch.bfloat16)
        ref = (x.to(torch.bfloat16) @ w).to(torch.float32)
        ok &= check(f"K={K} N={N}", G.gemv(x, w), ref)

    # The LM head is reached as `lm_head.T`, so B's contiguous axis is K, not
    # N. Assuming otherwise reads the wrong elements and still returns
    # plausible logits -- the decode ran and produced fluent wrong tokens, so
    # only a shape test catches it.
    K, N = D, 262144
    x = torch.randn(1, K, device=dev)
    head = (torch.randn(N, K, device=dev) * 0.05).to(torch.bfloat16)
    wt = head.T  # [K, N], stride (1, K)
    assert not wt.is_contiguous()
    ref = (x.to(torch.bfloat16) @ wt).to(torch.float32)
    ok &= check(f"K={K} N={N} transposed", G.gemv(x, wt), ref)

    # --- rmsnorm, weighted and weightless ---
    print("\nrmsnorm")
    for rows, N, has_w in ((1, D, True), (1, 256, True), (1, DH_GLOBAL, False)):
        x = torch.randn(rows, N, device=dev)
        w = torch.randn(N, device=dev) if has_w else None
        inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        ref = x * inv if w is None else x * inv * w
        ok &= check(f"rows={rows} N={N} w={has_w}", G.rmsnorm(x, w, RMS_EPS), ref)

    # --- the fused sublayer tail, including the per-layer out_scale ---
    print("\nrmsnorm_residual  (residual + norm(x)*w) * scale")
    for N, scale in ((D, 1.0), (D, 1.37)):
        x = torch.randn(1, N, device=dev)
        w = torch.randn(N, device=dev)
        r = torch.randn(1, N, device=dev)
        inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        ref = (r + x * inv * w) * scale
        ok &= check(
            f"N={N} scale={scale}",
            G.rmsnorm_residual(x, w, RMS_EPS, r, scale),
            ref,
        )

    # --- rope ---
    print("\nrope")
    for dh, nh in ((DH_SLIDING, N_Q_HEADS), (DH_GLOBAL, 1)):
        x = torch.randn(1, nh * dh, device=dev)
        lut = torch.randn(dh, device=dev)
        half = dh // 2
        cos, sin = lut[:half], lut[half:]
        v = x.reshape(nh, dh)
        x1, x2 = v[:, :half], v[:, half:]
        ref = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).reshape(1, -1)
        ok &= check(f"dh={dh} heads={nh}", G.rope(x, lut, nh, dh), ref)

    # --- geglu: exactness against gelu_tanh matters, see the C2 note ---
    print("\ngeglu")
    for n in (6144, 12288, 256):
        g = torch.randn(1, n, device=dev)
        u = torch.randn(1, n, device=dev)
        ref = torch.nn.functional.gelu(g, approximate="tanh") * u
        ok &= check(f"n={n}", G.geglu(g, u), ref)

    # --- logit softcap, spelled through sigmoid (see the kernel) ---
    print("\nlogit_softcap")
    for n, cap in ((262144, 30.0), (1024, 30.0)):
        x = torch.randn(n, device=dev) * 20
        ref = cap * torch.tanh(x / cap)
        ok &= check(f"n={n} cap={cap}", G.logit_softcap(x, cap), ref)

    # --- attention, including a context that does not fill BLOCK_S ---
    print("\nattn_decode  (MQA, one query row)")
    for S, dh in ((7, DH_SLIDING), (1, DH_GLOBAL), (163, DH_GLOBAL), (512, DH_SLIDING)):
        q = torch.randn(1, N_Q_HEADS * dh, device=dev)
        kc = torch.randn(S, dh, device=dev)
        vc = torch.randn(S, dh, device=dev)
        qh = q.reshape(N_Q_HEADS, dh)
        ref = (torch.softmax((qh @ kc.T) * 1.0, -1) @ vc).reshape(1, N_Q_HEADS * dh)
        ok &= check(f"S={S} dh={dh}", G.attn_decode(q, kc, vc, N_Q_HEADS, dh, 1.0), ref)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
