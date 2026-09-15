# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Operator backends for the Llama-3.2-1B prefill.

`TorchOps` is the reference: plain torch on the CPU, in float32 with bf16
weights, matching what the NPU kernels are expected to compute. `NpuOps`
(ops_npu.py) overrides one op at a time, so any divergence can be attributed
to a single kernel rather than to the model.
"""

import torch


class TorchOps:
    """CPU reference. Every method takes and returns float32 [..., D] torch."""

    name = "cpu"

    def rms_norm(self, x, weight, eps):
        """x: [N, D] -> normalized by RMS over D, scaled by `weight` [D]."""
        v = x.to(torch.float32)
        inv = torch.rsqrt((v * v).mean(-1, keepdim=True) + eps)
        return v * inv * weight

    def matmul(self, x, w):
        """x: [N, K] float32, w: [K, M] bfloat16 -> [N, M] float32.

        bf16 inputs with f32 accumulation, as the NPU GEMM does. Rounding x to
        bf16 here is not cosmetic: it is what the device sees, so keeping the
        reference in full f32 would hide a real error source.
        """
        return (x.to(torch.bfloat16).to(torch.float32) @ w.to(torch.float32)).to(
            torch.float32
        )

    def rope(self, x, lut, n_heads):
        """Half-split RoPE (HuggingFace Llama convention).

        x:   [N, n_heads*64]
        lut: [N, 64] = [cos_0..cos_31, sin_0..sin_31]

            out[i]      = x[i]*cos[i] - x[i+32]*sin[i]
            out[i+32]   = x[i]*sin[i] + x[i+32]*cos[i]

        Pairs (i, i+32), NOT adjacent (2i, 2i+1).
        """
        N = x.shape[0]
        half = lut.shape[-1] // 2
        cos = lut[:, :half].unsqueeze(1)  # [N, 1, 32]
        sin = lut[:, half:].unsqueeze(1)
        v = x.reshape(N, n_heads, 2 * half)
        x1, x2 = v[..., :half], v[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).reshape(
            N, -1
        )

    def attention(self, q, k, v, n_q, n_kv, dh):
        """Causal GQA. q: [N, n_q*dh], k/v: [N, n_kv*dh] -> [N, n_q*dh]."""
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

    def swiglu(self, gate, up):
        """SiLU(gate) * up, elementwise."""
        return torch.nn.functional.silu(gate) * up

    def lm_head(self, x, w):
        """x: [N, D] -> logits [N, VOCAB]. w is [VOCAB, D] (tied embed).

        Kept in bf16. `w.to(torch.float32)` would materialize a fresh 1 GB
        copy of the tied embedding on every call, which cost more than the
        matmul itself.
        """
        return torch.matmul(x.to(torch.bfloat16), w.t()).to(torch.float32)
