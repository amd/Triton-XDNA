# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Matmul / Linear layer kernel for GPT-2 inference.
GPU: K-tiled matmul with masking and L2 cache swizzling.
NPU: Single-block matmul (BLOCK_SIZE_K = K), requires aligned dims.
"""

import os
import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# GPU kernel: K-tiled matmul with masking and autotuning
# ---------------------------------------------------------------------------
# Single config: the per-shape autotune sweep (running every candidate on first
# call of each M,N,K) leaks into the first decode step and dominates TPOT. Decode
# matmuls are M=1 memory-bound GEMVs where the block config barely matters, so one
# reasonable config removes the sweep with no steady-state loss (gpt2-loop lesson).
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_gpu(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        a = tl.load(
            A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ---------------------------------------------------------------------------
# NPU kernel: single-block matmul (no K-tiling)
# ---------------------------------------------------------------------------
@triton.jit
def matmul_kernel_npu(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_block = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_block = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    c_block = tl.dot(a_block, b_block)

    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c_block)


# ---------------------------------------------------------------------------
# Wrapper: triton_linear
# ---------------------------------------------------------------------------
def _pad_to_multiple(x, multiple):
    """Pad last two dims of x to nearest multiple. Returns padded tensor and original shape."""
    if x.ndim == 1:
        n = x.shape[0]
        pad_n = math.ceil(n / multiple) * multiple - n
        if pad_n == 0:
            return x, n
        return torch.nn.functional.pad(x, (0, pad_n)), n
    m, n = x.shape[-2], x.shape[-1]
    pad_m = math.ceil(m / multiple) * multiple - m
    pad_n = math.ceil(n / multiple) * multiple - n
    if pad_m == 0 and pad_n == 0:
        return x, (m, n)
    return torch.nn.functional.pad(x, (0, pad_n, 0, pad_m)), (m, n)


from .backend_utils import CachedNPUKernel

_matmul_npu_cached = CachedNPUKernel()

# Cache of contiguous (K, N) bf16 weights for the GPU path, keyed by the source
# weight tensor's id. Weights are static and live for the model's lifetime, so
# the transpose + bf16 copy is done once instead of on every matmul call (decode
# runs it per token, per layer). The cached source tensor is held alongside the
# result so its id cannot be reused by a freed-then-reallocated tensor.
_gpu_weight_cache = {}


def _gpu_weight_bf16(weight, device):
    """Return a cached contiguous (K, N) bf16 copy of an (N, K) weight."""
    key = id(weight)
    cached = _gpu_weight_cache.get(key)
    if cached is not None and cached[1].device.type == device:
        return cached[1]
    w = weight.t().to(device=device, dtype=torch.bfloat16).contiguous()
    _gpu_weight_cache[key] = (weight, w)  # keep src ref so id stays unique
    return w


def triton_linear(x, weight, bias=None, backend="gpu", transform_script=None):
    """
    Compute x @ weight^T + bias using Triton matmul kernels.

    Args:
        x: Input tensor of shape (*, K)
        weight: Weight tensor of shape (N, K) — output dim first, like nn.Linear
        bias: Optional bias tensor of shape (N,)
        backend: "gpu" or "npu"
        transform_script: Path to transform script (NPU only)

    Returns:
        Output tensor of shape (*, N)
    """
    # Flatten batch dims
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])  # (M, K)
    M_orig, K = x_2d.shape
    N = weight.shape[0]

    if backend == "gpu":
        # GPU path: handles arbitrary shapes via masking + autotuning
        device = "cuda"
        # Skip transfer if already on GPU
        if x_2d.device.type == "cuda":
            x_bf16 = x_2d.to(dtype=torch.bfloat16).contiguous()
        else:
            x_bf16 = x_2d.to(device=device, dtype=torch.bfloat16).contiguous()
        # weight is (N, K), we need (K, N) for A @ B — cached once per weight
        w_bf16 = _gpu_weight_bf16(weight, device)
        c = torch.empty((M_orig, N), dtype=torch.bfloat16, device=device)

        # Lambda grid: autotuner selects block sizes, grid computed from them
        def grid(META):
            return (
                triton.cdiv(M_orig, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            )

        matmul_kernel_gpu[grid](
            x_bf16,
            w_bf16,
            c,
            M_orig,
            N,
            K,
            x_bf16.stride(0),
            x_bf16.stride(1),
            w_bf16.stride(0),
            w_bf16.stride(1),
            c.stride(0),
            c.stride(1),
        )
    else:
        # NPU path: pad to block-aligned dims, single launch (full K on-device).
        # BLOCK stays at 256 (the transform script's multi-core herd tiling assumes
        # a >=128 program block). The single-block kernel loads A (BLOCK_M x K) and
        # B (K x BLOCK_N) whole; the transform script's k_reduction_loop tiles the
        # K reduction across L3/L2/L1 on-device, so K is not chunked on the host.
        # K is padded to a power of two (tl.arange requires pow2) and zero-padded
        # for correctness.
        BLOCK_M = 256
        BLOCK_N = 256

        M_padded = math.ceil(M_orig / BLOCK_M) * BLOCK_M
        N_padded = math.ceil(N / BLOCK_N) * BLOCK_N
        K_padded = 1 << (K - 1).bit_length()

        x_bf16 = x_2d.to(torch.bfloat16)
        # Pad x: (M_orig, K) -> (M_padded, K_padded), zero-pad K dim for correctness
        pad_k = K_padded - K
        pad_m = M_padded - M_orig
        if pad_k > 0 or pad_m > 0:
            x_bf16 = torch.nn.functional.pad(x_bf16, (0, pad_k, 0, pad_m))
        x_bf16 = x_bf16.contiguous()

        # weight is (N, K), transpose to (K, N) then pad both K and N
        w_t = weight.t().to(torch.bfloat16)  # (K, N)
        pad_n = N_padded - N
        if pad_k > 0 or pad_n > 0:
            w_t = torch.nn.functional.pad(w_t, (0, pad_n, 0, pad_k))
        w_t = w_t.contiguous()

        # Set transform script
        old_script = os.environ.get("AIR_TRANSFORM_TILING_SCRIPT")
        if transform_script:
            os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = transform_script

        grid = (M_padded // BLOCK_M, N_padded // BLOCK_N)
        c = torch.empty((M_padded, N_padded), dtype=torch.float32)
        _matmul_npu_cached(
            matmul_kernel_npu,
            grid,
            x_bf16,
            w_t,
            c,
            M_padded,
            N_padded,
            K_padded,
            x_bf16.stride(0),
            x_bf16.stride(1),
            w_t.stride(0),
            w_t.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=K_padded,
        )

        # Restore env
        if old_script is not None:
            os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = old_script
        elif transform_script:
            del os.environ["AIR_TRANSFORM_TILING_SCRIPT"]

        # Slice to original dims
        c = c[:M_orig, :N]

    if bias is not None:
        c = c + bias.to(device=c.device, dtype=c.dtype)

    # Reshape back to batch dims
    out_shape = orig_shape[:-1] + (N,)
    return c.reshape(out_shape)


# ---------------------------------------------------------------------------
# GPU kernel: batched matmul for attention (Q@K^T, attn@V)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def bmm_kernel_gpu(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Batched matmul: C[b] = A[b] @ B[b]. Batch dim on program_id(1)."""
    pid_mn = tl.program_id(0)
    pid_b = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Offset pointers to this batch element
    A = A + pid_b * stride_ab
    B = B + pid_b * stride_bb
    C = C + pid_b * stride_cb

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        a = tl.load(
            A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ---------------------------------------------------------------------------
# Wrapper: triton_bmm
# ---------------------------------------------------------------------------
def triton_bmm(a, b):
    """
    Batched matrix multiply: C[i] = A[i] @ B[i].

    Args:
        a: (B, M, K) tensor in bf16
        b: (B, K, N) tensor in bf16

    Returns:
        (B, M, N) tensor in bf16
    """
    assert (
        a.ndim == 3 and b.ndim == 3
    ), f"Expected 3D tensors, got {a.ndim}D and {b.ndim}D"
    batch, M, K = a.shape
    _, K2, N = b.shape
    assert K == K2, f"Inner dims mismatch: {K} vs {K2}"

    device = "cuda"
    a_bf16 = a.to(device=device, dtype=torch.bfloat16).contiguous()
    b_bf16 = b.to(device=device, dtype=torch.bfloat16).contiguous()
    c = torch.empty((batch, M, N), dtype=torch.bfloat16, device=device)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            batch,
        )

    bmm_kernel_gpu[grid](
        a_bf16,
        b_bf16,
        c,
        M,
        N,
        K,
        a_bf16.stride(0),
        a_bf16.stride(1),
        a_bf16.stride(2),
        b_bf16.stride(0),
        b_bf16.stride(1),
        b_bf16.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
    )

    return c
