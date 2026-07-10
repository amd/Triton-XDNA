# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Fused multi-head attention kernel (FlashAttention-style) for GPT-2.

Fuses Q@K^T scaling, causal masking, softmax, and attn@V into a single kernel
using the online softmax algorithm. Eliminates 3 intermediate tensors and reduces
attention from 5 kernel launches to 1 per layer.
"""

import torch
import triton
import triton.language as tl


# Single config: the autotune sweep leaks into the first decode step (dominates
# TPOT). At decode seq_len=1 a small query block suffices; one config removes the
# sweep with negligible steady-state cost (gpt2-loop lesson).
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4),
    ],
    key=["seq_len"],
)
@triton.jit
def fused_attention_kernel(
    Q,
    K,
    V,
    Out,
    stride_qb,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vd,
    stride_ob,
    stride_os,
    stride_od,
    seq_len,
    total_len,
    scale,
    pos_offset,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Fused attention with online softmax.

    Grid: (cdiv(seq_len, BLOCK_M), batch * num_heads)
    Each program handles one query block for one batch-head pair.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Offset pointers to this batch-head
    Q = Q + pid_bh * stride_qb
    K = K + pid_bh * stride_kb
    V = V + pid_bh * stride_vb
    Out = Out + pid_bh * stride_ob

    # Query row indices for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_d = tl.arange(0, HEAD_DIM)  # (HEAD_DIM,)

    # Load Q block: (BLOCK_M, HEAD_DIM) — keep in bf16 for tl.dot hardware acceleration
    q_ptrs = Q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < seq_len
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online softmax state
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # running exp sum
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)  # running output

    # Iterate over K/V blocks
    for start_n in range(0, total_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

        # Load K block: (BLOCK_N, HEAD_DIM) — bf16 for tl.dot
        k_ptrs = K + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
        k_mask = offs_n[:, None] < total_len
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # S = Q @ K^T * scale: (BLOCK_M, BLOCK_N) — bf16 inputs, f32 accumulation
        s = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * scale

        # Bounds mask: positions beyond total_len
        s = tl.where(offs_n[None, :] < total_len, s, float("-inf"))

        # Causal mask: query at position (offs_m + pos_offset) can only see key at position offs_n
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None] + pos_offset) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        # Load V block: (BLOCK_N, HEAD_DIM) — bf16 for tl.dot
        v_ptrs = V + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=offs_n[:, None] < total_len, other=0.0)

        # Update accumulator: p (f32) cast to bf16 for dot, result accumulated in f32
        acc = alpha[:, None] * acc + tl.dot(p.to(tl.bfloat16), v_block)
        l_i = alpha * l_i + tl.sum(p, axis=1)
        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output: (BLOCK_M, HEAD_DIM) as bf16
    out_ptrs = Out + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    out_mask = offs_m[:, None] < seq_len
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def triton_fused_attention(q, k, v, scale, causal=True, pos_offset=0):
    """
    Fused multi-head attention using online softmax (FlashAttention-style).

    Fuses Q@K^T, scaling, causal masking, softmax, and attn@V into one kernel.

    Args:
        q: (B*num_heads, S, HEAD_DIM) query tensor
        k: (B*num_heads, T, HEAD_DIM) key tensor
        v: (B*num_heads, T, HEAD_DIM) value tensor
        scale: attention scale factor (1/sqrt(HEAD_DIM))
        causal: whether to apply causal masking
        pos_offset: position offset for causal mask (0 for prefill, past_len for decode)

    Returns:
        (B*num_heads, S, HEAD_DIM) output tensor in bf16
    """
    B_NH, S, HEAD_DIM = q.shape
    _, T, _ = k.shape

    device = "cuda"
    q = q.to(device=device, dtype=torch.bfloat16).contiguous()
    k = k.to(device=device, dtype=torch.bfloat16).contiguous()
    v = v.to(device=device, dtype=torch.bfloat16).contiguous()

    out = torch.empty((B_NH, S, HEAD_DIM), dtype=torch.bfloat16, device=device)

    def grid(META):
        return (triton.cdiv(S, META["BLOCK_M"]), B_NH)

    fused_attention_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        S,
        T,
        scale,
        pos_offset,
        HEAD_DIM=HEAD_DIM,
        IS_CAUSAL=causal,
    )

    return out
