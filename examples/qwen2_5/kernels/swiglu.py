# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
SwiGLU activation for Qwen2.5 MLP.

  SwiGLU(gate, up) = SiLU(gate) * up
  SiLU(x) = x * sigmoid(x)

The MLP is:  down_proj( SiLU(gate_proj(x)) * up_proj(x) )

GPU: a single fused kernel computes SiLU(gate) * up.
NPU: two passes — a unary SiLU kernel (reuses the elementwise transform script)
followed by an elementwise multiply (reuses the add transform script). This
mirrors how the gpt2 example keeps NPU kernels to single fusible primitives.
"""

import os
import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# GPU kernel: fused SiLU(gate) * up
# ---------------------------------------------------------------------------
@triton.jit
def swiglu_kernel_gpu(
    Gate,
    Up,
    Y,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    g = tl.load(Gate + offsets, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(Up + offsets, mask=mask, other=0.0).to(tl.float32)

    silu = g * tl.sigmoid(g)
    y = silu * u

    tl.store(Y + offsets, y.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# NPU kernel: unary SiLU
# ---------------------------------------------------------------------------
@triton.jit
def silu_kernel_npu(
    X,
    Y,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets[:])
    x_f32 = x.to(tl.float32)
    y = (x_f32 * tl.sigmoid(x_f32)).to(x.dtype)
    tl.store(Y + offsets[:], y)


# ---------------------------------------------------------------------------
# NPU kernel: elementwise multiply
# ---------------------------------------------------------------------------
@triton.jit
def mul_kernel_npu(
    A,
    B,
    C,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(A + offsets[:])
    b = tl.load(B + offsets[:])
    tl.store(C + offsets[:], a * b)


# ---------------------------------------------------------------------------
# Wrapper: triton_swiglu
# ---------------------------------------------------------------------------
from .backend_utils import CachedNPUKernel

_silu_npu_cached = CachedNPUKernel()
_mul_npu_cached = CachedNPUKernel()


def triton_swiglu(gate, up, backend="gpu", silu_script=None, mul_script=None):
    """
    SwiGLU: SiLU(gate) * up.

    Args:
        gate: gate_proj(x) output, any shape.
        up: up_proj(x) output, same shape as gate.
        backend: "gpu" or "npu"
        silu_script: NPU transform script for the unary SiLU pass.
        mul_script: NPU transform script for the elementwise multiply pass.

    Returns:
        SiLU(gate) * up, same shape.
    """
    orig_shape = gate.shape
    gate_flat = gate.reshape(-1).contiguous()
    up_flat = up.reshape(-1).contiguous()
    n_elements = gate_flat.numel()

    if backend == "gpu":
        device = "cuda"
        g_dev = gate_flat.to(device=device, dtype=torch.bfloat16).contiguous()
        u_dev = up_flat.to(device=device, dtype=torch.bfloat16).contiguous()
        output = torch.empty_like(g_dev)
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        swiglu_kernel_gpu[grid](g_dev, u_dev, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return output.reshape(orig_shape)

    # NPU path: SiLU(gate) then * up.
    BLOCK_SIZE = 1024
    n_padded = math.ceil(n_elements / BLOCK_SIZE) * BLOCK_SIZE
    pad = n_padded - n_elements

    g_npu = gate_flat.to(torch.bfloat16)
    u_npu = up_flat.to(torch.bfloat16)
    if pad > 0:
        g_npu = torch.nn.functional.pad(g_npu, (0, pad))
        u_npu = torch.nn.functional.pad(u_npu, (0, pad))
    g_npu = g_npu.contiguous()
    u_npu = u_npu.contiguous()

    silu_out = torch.empty(n_padded, dtype=torch.bfloat16)
    mul_out = torch.empty(n_padded, dtype=torch.bfloat16)
    grid = (n_padded // BLOCK_SIZE,)

    old_script = os.environ.get("AIR_TRANSFORM_TILING_SCRIPT")

    if silu_script:
        os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = silu_script
    _silu_npu_cached(
        silu_kernel_npu, grid, g_npu, silu_out, n_padded, BLOCK_SIZE=BLOCK_SIZE
    )

    if mul_script:
        os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = mul_script
    _mul_npu_cached(
        mul_kernel_npu, grid, silu_out, u_npu, mul_out, n_padded, BLOCK_SIZE=BLOCK_SIZE
    )

    if old_script is not None:
        os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = old_script
    elif silu_script or mul_script:
        os.environ.pop("AIR_TRANSFORM_TILING_SCRIPT", None)

    output = mul_out[:n_elements].to(torch.float32)
    return output.reshape(orig_shape)
