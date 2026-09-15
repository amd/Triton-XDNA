# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Llama-3.2-1B prefill: the model, with each op routed to a backend.

Structured like examples/gpt2 and examples/qwen2_5: the model carries a
`backend`, and each operator is a `_op(..., backend=None)` method that falls
back to it. The torch body of each method is the reference its NPU kernel is
checked against, so a wrong answer can be bisected op by op -- the only
practical way to debug a 16-layer model whose end-to-end signal is one token
id. `--ops` narrows which operators go to the NPU; `--compare-cpu` builds a
second model with backend="cpu" and diffs the KV cache layer by layer.

This forward is much shorter than gpt2's for two reasons, both structural:
there is no decode here (it is mlir-air's fused superkernel, reached through
the npz `save_kv_npz` describes), and there is no iGPU path, so no per-op
device placement.
"""

import os
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

from config import (
    D,
    DQ,
    DH,
    DK,
    DV,
    INTER,
    N_KV_HEADS,
    N_LAYERS,
    N_Q_HEADS,
    Q_PER_KV,
    RMS_EPS,
    VOCAB,
    load_q4nx,
    rope_lut,
)


def _t(a, dtype=torch.float32):
    """numpy (possibly ml_dtypes bfloat16) -> torch tensor."""
    if a.dtype.kind == "V" or str(a.dtype) == "bfloat16":
        a = a.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(a)).to(dtype)


class OpTimer:
    """Per-op wall-clock timer. Zero overhead when disabled.

    Same shape as examples/gpt2's, without its CUDA synchronise: there is no
    iGPU path here, so the wall time around a call already bounds the work.
    """

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.records = []  # (op_name, duration_ms)

    def reset(self):
        self.records.clear()

    @contextmanager
    def track(self, op_name):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        yield
        self.records.append((op_name, (time.perf_counter() - t0) * 1000))

    def summary(self):
        """Aggregate by op name, total time descending."""
        agg = defaultdict(float)
        for op, ms in self.records:
            agg[op] += ms
        return sorted(agg.items(), key=lambda kv: -kv[1])

    def report(self, total_s):
        total_ms = total_s * 1000
        for op, ms in self.summary():
            print(f"[profile] {op:10s} {ms:8.1f} ms  {100 * ms / total_ms:5.1f}%")
        print(f"[profile] {'TOTAL':10s} {total_ms:8.1f} ms")


class LlamaPrefill:
    """Q4NX Llama-3.2-1B prefill producing the decode's KV handoff.

    backend: "cpu" runs every operator in torch, which is the reference the NPU
    path is checked against; "npu" runs the operators named by `ops` as Triton
    kernels and leaves the rest in torch.
    """

    #: Operators with an NPU kernel. RoPE and attention are absent because
    #: neither has a transform script yet -- the same position
    #: examples/gpt2 --backend npu takes on attention, not a Triton limit.
    NPU_OPS = ("matmul", "rms_norm", "swiglu")

    def __init__(
        self, backend="cpu", ops="all", n_layers=N_LAYERS, max_seq=2048, model=None
    ):
        self.backend = backend
        self.enabled = self._resolve_ops(ops)
        self.timer = OpTimer(enabled=False)
        self.n_layers = n_layers
        self.max_seq = max_seq
        self.model = model
        self.current_context_length = 0
        # Per-layer KV cache: roped K and raw V, [max_seq, 512], head-major.
        self.kv_k = [np.zeros((max_seq, DK), np.float32) for _ in range(n_layers)]
        self.kv_v = [np.zeros((max_seq, DV), np.float32) for _ in range(n_layers)]
        self._w = None

    @classmethod
    def _resolve_ops(cls, spec):
        """Which operators may go to the NPU: "all", or a comma-separated list.

        LLAMA_NPU_OPS is consulted when the caller passes nothing, so a
        bisection can be driven from the environment without touching argv.
        """
        if spec is None:
            spec = os.environ.get("LLAMA_NPU_OPS", "all")
        if spec in ("all", "*"):
            return set(cls.NPU_OPS)
        enabled = {o.strip() for o in spec.split(",") if o.strip()}
        unknown = enabled - set(cls.NPU_OPS)
        if unknown:
            raise ValueError(f"unknown ops {sorted(unknown)}; known: {cls.NPU_OPS}")
        return enabled

    def _on_npu(self, op, backend):
        """True when `op` should run as a Triton kernel for this call."""
        return (backend or self.backend) == "npu" and op in self.enabled

    # ---- operators ----
    # Each is torch by default and Triton on the NPU when enabled. The torch
    # body is the reference --compare-cpu diffs against, so it models what the
    # device actually computes rather than the ideal.

    def _rms_norm(self, x, weight, eps, backend=None):
        """x: [N, D] -> normalized by RMS over D, scaled by `weight` [D]."""
        with self.timer.track("rms_norm"):
            if self._on_npu("rms_norm", backend):
                import kernels

                return kernels.triton_rms_norm(x, weight, eps)
            v = x.to(torch.float32)
            inv = torch.rsqrt((v * v).mean(-1, keepdim=True) + eps)
            return v * inv * weight

    def _matmul(self, x, w, backend=None):
        """x: [N, K] float32, w: [K, M] bfloat16 -> [N, M] float32.

        bf16 inputs with f32 accumulation, as the NPU GEMM does. Rounding x to
        bf16 here is not cosmetic: it is what the device sees, so keeping the
        reference in full f32 would hide a real error source.
        """
        with self.timer.track("matmul"):
            if self._on_npu("matmul", backend):
                import kernels

                return kernels.triton_matmul(x, w)
            return (x.to(torch.bfloat16).to(torch.float32) @ w.to(torch.float32)).to(
                torch.float32
            )

    def _swiglu(self, gate, up, backend=None):
        """SiLU(gate) * up, elementwise."""
        with self.timer.track("swiglu"):
            if self._on_npu("swiglu", backend):
                import kernels

                return kernels.triton_swiglu(gate, up)
            return torch.nn.functional.silu(gate) * up

    def _rope(self, x, lut, n_heads, backend=None):
        """Half-split RoPE (HuggingFace Llama convention). CPU only, see NPU_OPS.

        x:   [N, n_heads*64]
        lut: [N, 64] = [cos_0..cos_31, sin_0..sin_31]

            out[i]      = x[i]*cos[i] - x[i+32]*sin[i]
            out[i+32]   = x[i]*sin[i] + x[i+32]*cos[i]

        Pairs (i, i+32), NOT adjacent (2i, 2i+1).
        """
        with self.timer.track("rope"):
            N = x.shape[0]
            half = lut.shape[-1] // 2
            cos = lut[:, :half].unsqueeze(1)  # [N, 1, 32]
            sin = lut[:, half:].unsqueeze(1)
            v = x.reshape(N, n_heads, 2 * half)
            x1, x2 = v[..., :half], v[..., half:]
            return torch.cat(
                [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
            ).reshape(N, -1)

    def _attention(self, q, k, v, n_q, n_kv, dh, backend=None):
        """Causal GQA. q: [N, n_q*dh], k/v: [N, n_kv*dh] -> [N, n_q*dh].

        CPU only, see NPU_OPS.
        """
        with self.timer.track("attention"):
            N = q.shape[0]
            rep = n_q // n_kv
            qh = q.reshape(N, n_q, dh).transpose(0, 1)  # [n_q, N, dh]
            kh = k.reshape(N, n_kv, dh).transpose(0, 1)  # [n_kv, N, dh]
            vh = v.reshape(N, n_kv, dh).transpose(0, 1)
            kh = kh.repeat_interleave(rep, dim=0)  # GQA broadcast
            vh = vh.repeat_interleave(rep, dim=0)
            scores = (qh @ kh.transpose(1, 2)) * (dh**-0.5)  # [n_q, N, N]
            mask = torch.full((N, N), float("-inf")).triu(1)
            scores = scores + mask
            p = torch.softmax(scores, dim=-1)
            return (p @ vh).transpose(0, 1).reshape(N, n_q * dh)

    def _lm_head(self, x, w, backend=None):
        """x: [N, D] -> logits [N, VOCAB]. w is [VOCAB, D] (tied embed).

        Kept in bf16: `w.to(torch.float32)` would materialize a fresh 1 GB copy
        of the tied embedding on every call, costing more than the matmul.
        CPU only -- one GEMV, off the hot path.
        """
        with self.timer.track("lm_head"):
            return torch.matmul(x.to(torch.bfloat16), w.t()).to(torch.float32)

    # ---- weights ----
    def load_weights(self, model=None):
        raw = load_q4nx(model or self.model)
        self.fingerprint = raw["fingerprint"]
        self.embed = raw["embed"]  # float32 [VOCAB, D], kept as numpy (1 GB)
        self.final_norm = _t(raw["final_norm"])
        self.lm_head = _t(raw["lm_head"], torch.bfloat16)  # tied to embed
        self._w = []
        for L in raw["layers"]:
            w = {
                "attn_norm": _t(L["attn_norm"]),
                "ffn_norm": _t(L["ffn_norm"]),
                **{
                    k: _t(L[k], torch.bfloat16)
                    for k in ("q", "k", "v", "o", "gate", "up", "down")
                },
            }
            # An NPU dispatch costs ~24 ms of fixed overhead regardless of
            # size, so projections that share an input are concatenated along
            # their output dimension and issued as one GEMM: Q|K|V (one
            # normalized hidden in) and gate|up. Same arithmetic, 48 fewer
            # launches per prefill.
            w["qkv"] = torch.cat([w["q"], w["k"], w["v"]], dim=1).contiguous()
            w["gate_up"] = torch.cat([w["gate"], w["up"]], dim=1).contiguous()
            self._w.append(w)
        # bf16 cos/sin, as the device applies them.
        self._lut = _t(rope_lut(self.max_seq)).to(torch.bfloat16).to(torch.float32)

    # ---- forward ----
    def _layer(self, x, L, N, keep=None):
        """One transformer block. x: [N, D] float32. Returns [N, D].

        `keep` is how many leading rows are real; only those reach the KV
        cache. The rest are sequence padding (see SEQ_BUCKET).
        """
        keep = N if keep is None else keep
        w = self._w[L]
        h = self._rms_norm(x, w["attn_norm"], RMS_EPS)

        qkv = self._matmul(h, w["qkv"])  # [N, 2048+512+512]
        q, k, v = qkv.split([DQ, DK, DV], dim=1)

        q = self._rope(q, self._lut[:N], N_Q_HEADS)
        k = self._rope(k, self._lut[:N], N_KV_HEADS)

        # The decode's handoff: roped K, raw V, head-major within a position.
        self.kv_k[L][:keep] = k[:keep].to(torch.float32).numpy()
        self.kv_v[L][:keep] = v[:keep].to(torch.float32).numpy()

        a = self._attention(q, k, v, N_Q_HEADS, N_KV_HEADS, DH)  # [N, 2048]
        x = x + self._matmul(a, w["o"])

        h = self._rms_norm(x, w["ffn_norm"], RMS_EPS)
        g, u = self._matmul(h, w["gate_up"]).split([INTER, INTER], dim=1)
        return x + self._matmul(self._swiglu(g, u), w["down"])

    #: Prompt lengths are rounded up to a multiple of this before the forward
    #: pass. Every NPU kernel is compiled for constexpr shapes and cached on
    #: them, so a prompt of a new length otherwise recompiles RMSNorm and
    #: SwiGLU -- seconds of latency on the first turn at each new length, which
    #: in a chat REPL is most turns. 128 matches the matmul's M block, so the
    #: GEMMs see no extra work at all.
    SEQ_BUCKET = 128

    def prefill(self, ids):
        """Run the prompt. Returns logits [VOCAB] for the final position."""
        assert self._w is not None, "call load_weights() first"
        N = len(ids)
        assert N <= self.max_seq, (N, self.max_seq)
        # Pad to a bucket. Safe under causal masking: the padded rows sit after
        # every real one, so no real position attends to them, and their own
        # outputs are discarded.
        Nb = min(
            self.max_seq,
            ((N + self.SEQ_BUCKET - 1) // self.SEQ_BUCKET) * self.SEQ_BUCKET,
        )
        x = torch.zeros((Nb, D), dtype=torch.float32)
        x[:N] = _t(self.embed[np.asarray(ids)])
        for L in range(self.n_layers):
            x = self._layer(x, L, Nb, keep=N)
        self.current_context_length = N
        # Final norm on the prediction row only, then the LM head.
        xn = self._rms_norm(x[N - 1 : N], self.final_norm, RMS_EPS)
        return self._lm_head(xn, self.lm_head)[0]

    # ---- decode handoff (mirrors mlir-air's causal_lm interface) ----
    def kv_view(self, layer_idx):
        c = self.current_context_length
        return self.kv_k[layer_idx][:c], self.kv_v[layer_idx][:c]

    def get_current_context_length(self):
        return self.current_context_length

    def clear_context(self):
        self.current_context_length = 0
        for L in range(self.n_layers):
            self.kv_k[L][:] = 0
            self.kv_v[L][:] = 0

    def save_kv_npz(self, path, first, prompt):
        """Write the handoff mlir-air's `generate()` loads.

        The contract, derived from both ends -- what its `kv_view()` returns
        and what its `seed_kv()` assumes -- because getting any of it wrong
        degrades quality without failing:

            k, v : [16, P, 512] float32   (bf16-exact values)

        512 is 8 KV heads x 64, laid out as column `h*64 + d`, position-major,
        heads contiguous within a position. No permutation and no interleaving
        at this boundary: the region-major scatter the decode wants happens
        inside its own `seed_kv()`.

        K is stored already rotated, V raw. The rotation is half-split --
        `out[i] = x[i]*cos[i] - x[i+32]*sin[i]`, pairing i with i+32 rather
        than adjacent lanes -- and its table carries llama3 frequency scaling
        (factor 32, low 1, high 4, old context 8192, theta 500000), which is
        why `config.rope_lut` re-exports mlir-air's generator instead of
        rebuilding one: a reimplementation that drops the scaling rotates K on
        the wrong frequencies and only long prompts show it.
        """
        c = self.current_context_length
        K = np.stack([self.kv_k[L][:c] for L in range(self.n_layers)])
        V = np.stack([self.kv_v[L][:c] for L in range(self.n_layers)])
        np.savez(
            path,
            k=K.astype(np.float32),
            v=V.astype(np.float32),
            first=first,
            prompt=np.array(prompt),
        )
        return K, V
