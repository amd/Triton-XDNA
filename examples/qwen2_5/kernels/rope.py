# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Rotary position embeddings (RoPE) for Qwen2.5 attention.

RoPE rotates pairs of channels in the query/key vectors by a position-dependent
angle. Qwen2/Llama use the "half rotation" layout (rotate_half): the head_dim
is split into two halves and rotated together, rather than interleaved pairs.

RoPE is applied to small per-step tensors right before the fused attention
kernel (which runs on GPU in every backend), so a compact torch implementation
is used rather than a separate device kernel. precompute_rope_cache builds the
cos/sin tables once; apply_rope rotates q and k.
"""

import torch


def precompute_rope_cache(
    head_dim, max_seq_len, base=1000000.0, device="cpu", dtype=torch.float32
):
    """Precompute cos/sin tables of shape (max_seq_len, head_dim).

    base is rope_theta from the model config (Qwen2.5 uses 1e6).
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, half)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin, pos_offset=0):
    """Apply RoPE to q and k.

    Args:
        q: (B, n_heads, S, head_dim)
        k: (B, n_kv_heads, S, head_dim)
        cos, sin: (max_seq_len, head_dim) tables
        pos_offset: starting position index for this chunk

    Returns:
        q_rot, k_rot with the same shapes/dtypes as q, k.
    """
    S = q.shape[-2]
    cos_s = cos[pos_offset : pos_offset + S]  # (S, head_dim)
    sin_s = sin[pos_offset : pos_offset + S]
    # broadcast over (B, n_heads, S, head_dim)
    cos_b = cos_s.unsqueeze(0).unsqueeze(0).to(device=q.device, dtype=torch.float32)
    sin_b = sin_s.unsqueeze(0).unsqueeze(0).to(device=q.device, dtype=torch.float32)

    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    q_rot = q_f * cos_b + _rotate_half(q_f) * sin_b
    k_rot = k_f * cos_b + _rotate_half(k_f) * sin_b
    return q_rot.to(q.dtype), k_rot.to(k.dtype)
