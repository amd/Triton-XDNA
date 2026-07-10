# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Elementwise addition kernel for GPT-2 residual connections.
GPU: Standard elementwise add with masking.
NPU: Block-based add adapted from vec-add example.
"""

import os
import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# GPU kernel: elementwise add
# ---------------------------------------------------------------------------
@triton.jit
def add_kernel_gpu(
    A,
    B,
    C,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(A + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(C + offsets, (a + b).to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# NPU kernel: block-based add
# ---------------------------------------------------------------------------
@triton.jit
def add_kernel_npu(
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
    tl.store(C + offsets[:], a + b)


# ---------------------------------------------------------------------------
# Wrapper: triton_add
# ---------------------------------------------------------------------------
from .backend_utils import CachedNPUKernel

_add_npu_cached = CachedNPUKernel()


def triton_add(a, b, backend="gpu", transform_script=None):
    """
    Elementwise addition.

    Args:
        a: First input tensor.
        b: Second input tensor (must be broadcastable to a's shape).
        backend: "gpu" or "npu"
        transform_script: Path to transform script (NPU only)

    Returns:
        a + b
    """
    # Broadcast b to a's shape if needed
    if a.shape != b.shape:
        b = b.expand_as(a)

    orig_shape = a.shape
    a_flat = a.reshape(-1).contiguous()
    b_flat = b.reshape(-1).contiguous()
    n_elements = a_flat.numel()

    if backend == "gpu":
        device = "cuda"
        if a_flat.device.type == "cuda":
            a_dev = a_flat.to(dtype=torch.bfloat16).contiguous()
        else:
            a_dev = a_flat.to(device=device, dtype=torch.bfloat16)
        if b_flat.device.type == "cuda":
            b_dev = b_flat.to(dtype=torch.bfloat16).contiguous()
        else:
            b_dev = b_flat.to(device=device, dtype=torch.bfloat16)
        output = torch.empty_like(a_dev)
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        add_kernel_gpu[grid](a_dev, b_dev, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # NPU path
        from .backend_utils import CachedNPUKernel

        BLOCK_SIZE = 1024
        n_padded = math.ceil(n_elements / BLOCK_SIZE) * BLOCK_SIZE
        a_npu = a_flat.to(torch.bfloat16)
        b_npu = b_flat.to(torch.bfloat16)
        if n_padded != n_elements:
            a_npu = torch.nn.functional.pad(a_npu, (0, n_padded - n_elements))
            b_npu = torch.nn.functional.pad(b_npu, (0, n_padded - n_elements))
        a_npu = a_npu.contiguous()
        b_npu = b_npu.contiguous()
        output = torch.empty(n_padded, dtype=torch.bfloat16)

        old_script = os.environ.get("AIR_TRANSFORM_TILING_SCRIPT")
        if transform_script:
            os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = transform_script

        grid = (n_padded // BLOCK_SIZE,)
        _add_npu_cached(
            add_kernel_npu, grid, a_npu, b_npu, output, n_padded, BLOCK_SIZE=BLOCK_SIZE
        )

        if old_script is not None:
            os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = old_script
        elif transform_script:
            del os.environ["AIR_TRANSFORM_TILING_SCRIPT"]

        output = output[:n_elements].to(torch.float32)

    return output.reshape(orig_shape)
