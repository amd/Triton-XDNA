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


def _random_q4nx(G, N, K, dev):
    """A Codec B matrix of random nibbles with realistic scales and mins.

    Random bytes throughout would put NaN and inf in the bf16 scale fields.
    Scales positive and mins negative, as every one in the real bundle is --
    which Codec K relies on.
    """
    nb = (N // G.Q4NX_ROWS) * (K // G.Q4NX_COLS)
    data = torch.randint(0, 256, (nb, G.Q4NX_CHUNK_BYTES), dtype=torch.uint8)
    sc = torch.rand(nb, 256) * 0.05 + 0.001
    mn = -sc * (torch.rand(nb, 256) * 12 + 1)
    sm = torch.cat([sc, mn], 1).to(torch.bfloat16)
    data[:, :1024] = sm.view(torch.uint8)
    return G.Q4NXWeight(data.reshape(-1), N, K).to(dev)


def _gemv_q4nx(G, dev):
    """The W4A16 GEMV, against the torch dequantization and against mlir-air's.

    Two references because they catch different things. `dequant_q4nx` is the
    same reading of the layout written a second way, so it catches a kernel
    bug. mlir-air's `Q4nxModel.dequant` is the reading every other path in this
    stack uses, so it catches `dequant_q4nx` and the kernel sharing one wrong
    idea of where a nibble lives -- which the first check cannot see.
    """
    print("\ngemv_q4nx  [1,K] @ Codec B [N,K]^T")
    ok = True
    for K, N in (
        (D, N_Q_HEADS * DH_GLOBAL + 2 * DH_GLOBAL),  # qkv, full layer
        (D, N_Q_HEADS * DH_SLIDING),  # q alone, KV-shared sliding layer
        (N_Q_HEADS * DH_GLOBAL, D),  # o, full layer
        (D, 2 * 12288),  # gate_up, wide FFN
        (12288, D),  # down, wide
        # Not this model's: chunk counts the split-K heuristic does not divide
        # (25 -> no split, 39 -> 3). Both once dropped their trailing chunks.
        (6400, D),
        (9984, D),
        (D, 262144),  # lm_head
    ):
        w = _random_q4nx(G, N, K, dev)
        x = torch.randn(1, K, device=dev)
        # The reference is checked on at most the first 4096 rows: a full f32
        # dequantization of the head is 1.6 GB before its intermediates, more
        # than this iGPU has free. Chunks are block-major, so a row prefix is
        # a chunk prefix.
        n = min(N, 4096)
        head = G.Q4NXWeight(w.data[: n // 32 * (K // 256) * 5120], n, K)
        ref = x.to(torch.bfloat16).float() @ G.dequant_q4nx(head)
        # f32 on both sides, so the tolerance is summation order, not rounding.
        got = G.gemv_q4nx(x, w)[:, :n]
        ok &= check(f"K={K} N={N}", got, ref, tol=1e-4)

    # Codec K: the kernel against the torch decoding of the same bytes, and the
    # re-encoding against the Codec B it came from -- the latter in units of
    # the 4-bit step, since the loss is meant to be small next to it.
    #
    # Encoded on the host, as `Gemma4GpuDecode._packed` does: the encoder's
    # torch ops reset the iGPU under `HSA_OVERRIDE`. Only the result moves.
    for K, N in ((D, 2048), (12288, D), (D, 262144)):
        wk = _random_q4nx(G, N, K, "cpu").to_codec_k().to(dev)
        x = torch.randn(1, K, device=dev)
        n = min(N, 4096)
        head = G.Q4NXWeight(wk.data[: n // 32 * (K // 256) * wk.chunk_bytes], n, K, "k")
        ref = x.to(torch.bfloat16).float() @ G.dequant_q4nx(head)
        got = G.gemv_q4nx(x, wk)[:, :n]
        ok &= check(f"codec K  K={K} N={N}", got, ref, tol=1e-4)
    wb = _random_q4nx(G, 2048, D, "cpu")
    wk = wb.to_codec_k()
    sc_b, _ = wb._scales_mins()
    step = sc_b.mean().item()
    err = (G.dequant_q4nx(wk) - G.dequant_q4nx(wb)).pow(2).mean().sqrt().item()
    rel = err / (step / 12**0.5)
    print(f"  codec K re-encoding error          {rel:.3f} x the 4-bit noise")
    ok &= rel < 0.5

    # The GeGLU fused into `down`'s activation load, against the launch it
    # replaces -- at both FFN widths, the wide one being the split-K path.
    for inter in (6144, 12288):
        gu = torch.randn(1, 2 * inter, device=dev)
        down = _random_q4nx(G, D, inter, dev)
        ref = G.gemv_q4nx(G.geglu(gu[:, :inter], gu[:, inter:]), down)
        got = G.gemv_q4nx(gu, down, glu_in=True)
        ok &= check(f"geglu -> down K={inter} fused", got, ref, tol=1e-4)

    # Through `airsrc`, not the example's `config`: binding a `config` here
    # would shadow the one `_phi4_partial_rope` needs.
    try:
        import numpy as np

        import airsrc

        airsrc.add_air_paths("gemma4_e2b_q4nx")
        import gemma4_e2b_q4nx_weights as gw
    except (ImportError, RuntimeError) as e:
        print(f"  SKIP mlir-air layout cross-check: {e}")
        return ok
    N, K = 64, 512
    w = _random_q4nx(G, N, K, "cpu")
    qm = gw.Q4nxModel.__new__(gw.Q4nxModel)
    qm._hdr = {"t": {"shape": [(N // 32) * (K // 256)]}}
    qm._raw = lambda name, dtype: w.data.numpy().view(dtype)
    theirs = torch.from_numpy(qm.dequant("t", N, K).astype(np.float32))
    ok &= check("dequant_q4nx vs mlir-air dequant", G.dequant_q4nx(w).T, theirs, 1e-6)
    return ok


def _qkv_post(G, dev):
    """`qkv_post` against the separate norm / rope / scatter it fuses."""
    import kv_layout

    print("\nqkv_post  head norms + rope + KV append")
    ok = True
    maxl, pos, fan = 16, 5, [0, 2, 3]
    for dh, owns in ((256, True), (512, True), (256, False)):
        n = N_Q_HEADS * dh + (2 * dh if owns else 0)
        qkv = torch.randn(1, n, device=dev)
        qn, kn = torch.randn(dh, device=dev), torch.randn(dh, device=dev)
        row = torch.randn(dh, device=dev)
        slab = torch.zeros(
            kv_layout.slab_shape(4, maxl), dtype=torch.bfloat16, device=dev
        )
        kv = None
        if owns:
            fan_t = torch.tensor(fan, dtype=torch.int32, device=dev)
            kv = (slab, fan_t, pos, kv_layout.region_stride(maxl))
        q = G.qkv_post(qkv, qn, kn if owns else None, row, N_Q_HEADS, dh, RMS_EPS, kv)

        def normed_roped(x, w, heads):
            y = G.rmsnorm(x.reshape(heads, dh), w, RMS_EPS).reshape(1, -1)
            return G.rope(y, row, heads, dh)

        qd = N_Q_HEADS * dh
        ok &= check(f"q  dh={dh} owns={owns}", q, normed_roped(qkv[:, :qd], qn, 8))
        if not owns:
            continue
        k = normed_roped(qkv[:, qd : qd + dh], kn, 1)
        v = G.rmsnorm(qkv[:, qd + dh :], None, RMS_EPS)
        want = torch.zeros_like(slab)
        for L in fan:
            for region, t in ((kv_layout.K_REGION, k), (kv_layout.V_REGION, v)):
                view = kv_layout.region_view(want, L, region, maxl)
                kv_layout.scatter_rows(view[pos : pos + 1], t.to(torch.bfloat16), dh)
        ok &= check(f"kv slab dh={dh}", slab.float(), want.float(), tol=1e-6)

    # The prefill's form: T tokens in one launch, rows pos .. pos + T - 1, and
    # the dense K/V its attention reads. Must equal T single-token calls.
    T, pos0 = 5, 3
    fan_t = torch.tensor(fan, dtype=torch.int32, device=dev)
    for dh in (256, 512):
        n = N_Q_HEADS * dh + 2 * dh
        qkv = torch.randn(T, n, device=dev)
        qn, kn = torch.randn(dh, device=dev), torch.randn(dh, device=dev)
        rows = torch.randn(T, dh, device=dev)
        slab = torch.zeros(
            kv_layout.slab_shape(4, maxl), dtype=torch.bfloat16, device=dev
        )
        kv = (slab, fan_t, pos0, kv_layout.region_stride(maxl))
        q, kd, vd = G.qkv_post(
            qkv, qn, kn, rows, N_Q_HEADS, dh, RMS_EPS, kv, dense_kv=True
        )
        one = torch.zeros_like(slab)
        q1 = []
        for t in range(T):
            kv1 = (one, fan_t, pos0 + t, kv_layout.region_stride(maxl))
            q1.append(
                G.qkv_post(qkv[t : t + 1], qn, kn, rows[t], N_Q_HEADS, dh, RMS_EPS, kv1)
            )
        ok &= check(f"T={T} q  dh={dh}", q, torch.cat(q1), tol=1e-6)
        ok &= check(f"T={T} kv slab dh={dh}", slab.float(), one.float(), tol=1e-6)
        qd = N_Q_HEADS * dh
        kn_rows = G.rmsnorm(qkv[:, qd : qd + dh], kn, RMS_EPS)
        k_ref = torch.cat(
            [G.rope(kn_rows[t : t + 1], rows[t], 1, dh) for t in range(T)]
        )
        ok &= check(f"T={T} dense K dh={dh}", kd, k_ref, tol=1e-5)
        v_ref = G.rmsnorm(qkv[:, qd + dh :], None, RMS_EPS)
        ok &= check(f"T={T} dense V dh={dh}", vd, v_ref, tol=1e-5)
    try:
        G.qkv_post(
            qkv,
            qn,
            kn,
            rows,
            N_Q_HEADS,
            dh,
            RMS_EPS,
            (slab, fan_t, maxl - 2, kv_layout.region_stride(maxl)),
        )
        print("  rows past the slab: no error  FAIL")
        ok = False
    except ValueError:
        print("  rows past the slab: ValueError  ok")
    return ok


def _gemm_w4a8(G, dev):
    """The prefill's W4A8 GEMM, its q8 activations, and the fused gate|up.

    Two references for the GEMM: the dequantized q8 activation against the
    f32 dequantized weights, where only summation order differs, and the f32
    activation, which measures what the int8 rounding costs.
    """
    print("\nquant_q8 / gemm_w4a8  q8 [M,K] @ Codec B/K [N,K]^T")
    ok = True
    x = torch.randn(37, D, device=dev)
    xq, dx, xs = G.quant_q8(x)
    xr = (xq.float().view(37, -1, 32) * dx[:, :, None]).view(37, D)
    ok &= check("quant_q8 round trip", xr, x, tol=1e-2)
    ok &= check(
        "quant_q8 scaled sum", xs, dx * xq.float().view(37, -1, 32).sum(-1), 1e-6
    )
    for K, N in (
        (D, N_Q_HEADS * DH_SLIDING + 2 * DH_SLIDING),  # qkv, sliding layer
        (N_Q_HEADS * DH_GLOBAL, D),  # o, full layer
        (6144, D),  # down, narrow
    ):
        wb = _random_q4nx(G, N, K, "cpu")
        # Codec K is encoded on the host, as `Gemma4GpuDecode._packed` does.
        for codec, w in (("B", wb.to(dev)), ("K", wb.to_codec_k().to(dev))):
            wf = G.dequant_q4nx(w)
            for M in (1, 37, 128):
                x = torch.randn(M, K, device=dev)
                xq, dx, xs = G.quant_q8(x)
                xr = (xq.float().view(M, -1, 32) * dx[:, :, None]).view(M, K)
                got = G.gemm_w4a8(x, w)
                name = f"{codec} K={K} N={N} M={M}"
                ok &= check(f"{name} vs q8 x", got, xr @ wf, tol=1e-4)
                ok &= check(f"{name} vs f32 x", got, x @ wf)

    print("\ngemm_w4a8_glu_q8  quant_q8(gelu(gate) * up), fused into gate|up")
    for inter in (2048, 6144):
        wb = _random_q4nx(G, 2 * inter, D, "cpu")
        for codec, w in (("B", wb.to(dev)), ("K", wb.to_codec_k().to(dev))):
            for M in (1, 37, 128):
                x = torch.randn(M, D, device=dev)
                gu = G.gemm_w4a8(x, w)
                act = G.geglu(gu[:, :inter], gu[:, inter:]).view(M, inter)
                want = G.quant_q8(act)
                got = G.gemm_w4a8_glu_q8(x, w)
                name = f"{codec} inter={inter} M={M}"
                diff = (got[0].int() - want[0].int()).abs().max().item()
                print(
                    f"  {name:<34} codes max |diff| {diff}  {'ok' if diff <= 1 else 'FAIL'}"
                )
                ok &= diff <= 1
                ok &= check(f"{name} scales", got[1], want[1], tol=1e-5)
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

    # The PLE pair: `inp_gate` split over K, its partials summed and the GeGLU
    # applied as `per_layer_projection` loads them.
    x = torch.randn(1, D, device=dev)
    wg = (torch.randn(D, 256, device=dev) * 0.05).to(torch.bfloat16)
    wp = (torch.randn(256, D, device=dev) * 0.05).to(torch.bfloat16)
    u = torch.randn(1, 256, device=dev)
    ref = G.gemv(G.geglu(G.gemv(x, wg), u), wp)
    got = G.gemv(G.gemv(x, wg, split_k=4), wp, glu_up=u)
    ok &= check("split-K -> GeGLU-on-load (PLE pair)", got, ref, tol=1e-4)

    ok &= _gemv_q4nx(G, dev)
    ok &= _qkv_post(G, dev)
    ok &= _gemm_w4a8(G, dev)

    # --- rmsnorm, weighted and weightless ---
    print("\nrmsnorm")
    for rows, N, has_w in ((1, D, True), (1, 256, True), (1, DH_GLOBAL, False)):
        x = torch.randn(rows, N, device=dev)
        w = torch.randn(N, device=dev) if has_w else None
        inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        ref = x * inv if w is None else x * inv * w
        ok &= check(f"rows={rows} N={N} w={has_w}", G.rmsnorm(x, w, RMS_EPS), ref)

    # --- the fused sublayer tail, including the per-layer out_scale ---
    # One row is a decode step; the prefill passes the whole prompt.
    print("\nrmsnorm_residual  (residual + norm(x)*w) * scale")
    for rows, N, scale in ((1, D, 1.0), (1, D, 1.37), (37, D, 1.37)):
        x = torch.randn(rows, N, device=dev)
        w = torch.randn(N, device=dev)
        r = torch.randn(rows, N, device=dev)
        inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        ref = (r + x * inv * w) * scale
        name = f"rows={rows} N={N} scale={scale}"
        ok &= check(name, G.rmsnorm_residual(x, w, RMS_EPS, r, scale), ref)
        w2 = torch.randn(N, device=dev)
        y, h = G.rmsnorm_residual(x, w, RMS_EPS, r, scale, next_norm=w2)
        ok &= check(f"{name} +next_norm (y)", y, ref)
        ok &= check(f"{name} +next_norm (h)", h, G.rmsnorm(ref, w2, RMS_EPS))

    # --- the per-layer inputs, all 35 layers in one launch ---
    print("\nple_combine  (rmsnorm(p*s_in)*w + t) * s_out, per layer")
    L, n, s_in, s_out = 35, 256, D**-0.5, 2**-0.5
    p = torch.randn(1, L * n, device=dev)
    w = torch.randn(n, device=dev)
    t = torch.randn(L, n, device=dev)
    ps = p.reshape(L, n) * s_in
    inv = torch.rsqrt(ps.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    ref = (ps * inv * w + t) * s_out
    ok &= check(f"L={L} n={n}", G.ple_combine(p, w, t, s_in, s_out, RMS_EPS), ref)

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
    print("\nattn_prefill, attn_prefill_fa  (prefill; causal GQA, windowed and not)")
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
        # The fp16 kernel is checked on the same cases; its operands are
        # rounded to fp16, so it gets the default tolerance, not a tighter one.
        for name, got in (
            ("f32", G.attn_prefill(q, k, v, N_Q_HEADS, 1, dh, win, 1.0)),
            ("fa ", G.attn_prefill_fa(q, k, v, N_Q_HEADS, dh, win, 1.0)),
        ):
            if torch.isnan(got).any():
                print(f"  {name} N={N} dh={dh} win={win}  NaN  FAIL")
                ok = False
                continue
            ok &= check(f"{name} N={N} dh={dh} win={win}", got, ref)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
