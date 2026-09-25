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


def _phi4_partial_rope(dev):
    """`Phi4Prefill._rope` must agree with itself on both backends.

    Phi-4 advertises `hetero`, and its `_rope` override used to ignore the
    backend and stay on the host -- so the one operator that differs from
    Llama's was also the one that never reached the iGPU. This drives the real
    method, host path against GPU path, on a stub: the model's bundle is
    several GB and nothing here needs a weight.
    """
    print("\nPhi4Prefill._rope  (partial rotary, host vs iGPU)")
    # `phi4_prefill` resolves its dims from whichever `config` the example
    # directory bound, so that directory has to lead sys.path. This file
    # deliberately binds no config of its own, so there is nothing to shadow.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "phi4_mini_q4nx"))
    try:
        import phi4_prefill
        from llama_prefill import OpTimer
    except ImportError as e:  # its config reaches into mlir-air's sources
        print(f"  SKIP: {e}")
        return True

    pf = phi4_prefill.Phi4Prefill.__new__(phi4_prefill.Phi4Prefill)
    pf.timer = OpTimer()
    R, nh, dh, N = pf.ROPE_DIM, 3, 128, 21
    x = torch.randn(N, nh * dh)
    lut = torch.randn(N, R)

    pf._gpu_device = lambda backend=None: None
    host = pf._rope(x, lut, nh)
    pf._gpu_device = lambda backend=None: dev
    gpu = pf._rope(x, lut, nh)

    # The tail is the half the rotation must not touch, so it is checked on its
    # own rather than being averaged into the whole row.
    tail_moved = (gpu.reshape(N, nh, dh)[..., R:] - x.reshape(N, nh, dh)[..., R:]).abs()
    if tail_moved.max() > 0:
        print(f"  FAIL: the unrotated tail moved by {tail_moved.max():.3e}")
        return False
    print(f"  PASS: tail of {dh - R} lanes per head passed through untouched")
    return check(f"N={N} heads={nh} dh={dh} rot={R}", gpu, host)


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

    # The same attention against the SHARED cache layout: rows REGION_W wide,
    # the head scattered as [real_lo | zeros | real_hi | zeros]. Same answer as
    # the dense case above or the layout unification is silently wrong -- and
    # silently is the word, because a mis-read cache still produces fluent
    # text. Compared against the dense call rather than against torch so that
    # only the layout is under test.
    print("\nattn_decode  (shared cache layout: strided rows, padded head)")
    import kv_layout as KVL

    for S, dh in (
        (7, DH_SLIDING),
        (163, DH_GLOBAL),
        (512, DH_SLIDING),
        (2048, DH_SLIDING),
    ):
        q = torch.randn(1, N_Q_HEADS * dh, device=dev)
        kd = torch.randn(S, dh, device=dev).to(torch.bfloat16)
        vd = torch.randn(S, dh, device=dev).to(torch.bfloat16)
        slab = torch.zeros(S, KVL.REGION_W, dtype=torch.bfloat16, device=dev)
        slabv = torch.zeros_like(slab)
        KVL.scatter_rows(slab, kd, dh)
        KVL.scatter_rows(slabv, vd, dh)
        shift = KVL.DH_A // 2 - dh // 2
        ref = G.attn_decode(q, kd, vd, N_Q_HEADS, dh, 1.0)
        got = G.attn_decode(q, slab, slabv, N_Q_HEADS, dh, 1.0, lane_shift=shift)
        # Bit-identical, not merely close: the same values reach the same
        # accumulator in the same order, and only the address arithmetic
        # differs. A tolerance here would hide a lane map that is off by one.
        ok &= check(f"S={S} dh={dh} shift={shift}", got, ref, tol=1e-9)

    # --- prefill shapes: N > 1, which the decode cases above never reach ---
    print("\nmatmul  [M,K]@[K,N]  (prefill)")
    for M, K, N in ((6, D, 5120), (163, D, 2 * 12288), (163, 6144, D), (37, D, 256)):
        x = torch.randn(M, K, device=dev)
        w = (torch.randn(K, N, device=dev) * 0.05).to(torch.bfloat16)
        ref = (x.to(torch.bfloat16) @ w).to(torch.float32)
        ok &= check(f"M={M} K={K} N={N}", G.matmul(x, w), ref)

    # `rot < dh` is Phi-4-mini's partial rotary (128-wide heads, 96 rotated).
    # The tail must come through untouched and the pairing must stay inside the
    # rotated slice: pairing across the whole head would reach lane 0 against a
    # tail lane and give fluent wrong text rather than an error. 128/96 is the
    # real shape; 64/34 is a rot whose half is not a power of two.
    print("\nrope_batch  (prefill; whole head and partial rotary)")
    for N, nh, dh, rot in (
        (6, N_Q_HEADS, DH_SLIDING, None),
        (163, 1, DH_GLOBAL, None),
        (37, 24, 128, 96),
        (8, 3, 64, 34),
    ):
        R = dh if rot is None else rot
        x = torch.randn(N, nh * dh, device=dev)
        lut = torch.randn(N, R, device=dev)
        half = R // 2
        cos, sin = lut[:, :half].unsqueeze(1), lut[:, half:].unsqueeze(1)
        v = x.reshape(N, nh, dh)
        x1, x2, tail = v[..., :half], v[..., half:R], v[..., R:]
        ref = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, tail], -1).reshape(
            N, nh * dh
        )
        ok &= check(
            f"N={N} heads={nh} dh={dh} rot={R}",
            G.rope_batch(x, lut, nh, dh, rot=rot),
            ref,
        )

    # A LUT that does not match the rotation is the silent failure this guards:
    # it would reinterpret the head boundary rather than raise.
    try:
        G.rope_batch(
            torch.randn(4, 2 * 128, device=dev),
            torch.randn(4, 128, device=dev),
            2,
            128,
            rot=96,
        )
        print("FAIL: rope_batch accepted a LUT wider than its rotation")
        ok = False
    except ValueError:
        print("PASS: rope_batch refuses a LUT that does not match `rot`")

    ok &= _phi4_partial_rope(dev)

    # Causal GQA, both window modes, and lengths past the window -- a key block
    # entirely outside it leaves every score masked, which is where the softmax
    # produced NaN until the running maximum was guarded. Non-powers of two on
    # purpose: the tiles are fixed, so the masked tail is the interesting part.
    print("\nattn_prefill  (prefill; causal GQA, windowed and not)")
    for N, dh, win in (
        (6, DH_SLIDING, 512),
        (37, DH_SLIDING, None),
        (163, DH_GLOBAL, None),
        (163, DH_SLIDING, 512),
        (700, DH_SLIDING, 512),  # past the window: the NaN case
        (1024, DH_SLIDING, 512),
    ):
        q = torch.randn(N, N_Q_HEADS * dh, device=dev)
        k = torch.randn(N, dh, device=dev)
        v = torch.randn(N, dh, device=dev)
        qh = q.reshape(N, N_Q_HEADS, dh).transpose(0, 1)
        kh = k.reshape(N, 1, dh).transpose(0, 1).repeat_interleave(N_Q_HEADS, 0)
        vh = v.reshape(N, 1, dh).transpose(0, 1).repeat_interleave(N_Q_HEADS, 0)
        mask = torch.full((N, N), float("-inf"), device=dev).triu(1)
        if win is not None:
            mask = mask + torch.full((N, N), float("-inf"), device=dev).tril(-win)
        ref = torch.softmax((qh @ kh.transpose(1, 2)) * 1.0 + mask, -1) @ vh
        ref = ref.transpose(0, 1).reshape(N, N_Q_HEADS * dh)
        got = G.attn_prefill(q, k, v, N_Q_HEADS, 1, dh, win, 1.0)
        if torch.isnan(got).any():
            print(f"  N={N} dh={dh} win={win}  NaN  FAIL")
            ok = False
            continue
        ok &= check(f"N={N} dh={dh} win={win}", got, ref)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
