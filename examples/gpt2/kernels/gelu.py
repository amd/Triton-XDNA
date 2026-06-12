# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
GELU activation kernel for GPT-2.
Uses the tanh approximation (gelu_new) to match HuggingFace GPT-2:
  GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Both GPU and NPU kernels use the same tanh-approximation formula, converted
to sigmoid via the identity tanh(z) = 2*sigmoid(2z) - 1, giving:
  GELU(x) = x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 * x^3))
"""

import os
import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# GPU kernel: tanh-approximation GELU (matches GPT-2 gelu_new)
# ---------------------------------------------------------------------------
@triton.jit
def gelu_kernel_gpu(
    X, Y,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)

    # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using identity: tanh(z) = 2 * sigmoid(2*z) - 1
    # So: 0.5 * x * (1 + tanh(z)) = x * sigmoid(2*z)
    k = 0.7978845608028654  # sqrt(2/pi)
    z = k * (x + 0.044715 * x * x * x)
    y = x * tl.sigmoid(2.0 * z)

    tl.store(Y + offsets, y.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# NPU kernel: tanh-approximation GELU (matches GPU kernel / HuggingFace)
# ---------------------------------------------------------------------------
@triton.jit
def gelu_kernel_npu(
    X, Y,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offsets[:])
    # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using identity: tanh(z) = 2 * sigmoid(2*z) - 1
    # So: 0.5 * x * (1 + tanh(z)) = x * sigmoid(2*z)
    x_f32 = x.to(tl.float32)
    k: tl.constexpr = 0.7978845608028654  # sqrt(2/pi)
    z = k * (x_f32 + 0.044715 * x_f32 * x_f32 * x_f32)
    y = (x_f32 * tl.sigmoid(2.0 * z)).to(x.dtype)
    tl.store(Y + offsets[:], y)


# ---------------------------------------------------------------------------
# Wrapper: triton_gelu
# ---------------------------------------------------------------------------
from .backend_utils import CachedNPUKernel
_gelu_npu_cached = CachedNPUKernel()


def triton_gelu(x, backend="gpu", transform_script=None):
    """
    GELU activation.

    Args:
        x: Input tensor of any shape.
        backend: "gpu" or "npu"
        transform_script: Path to transform script (NPU only)

    Returns:
        GELU(x) of same shape and dtype.
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1).contiguous()
    n_elements = x_flat.numel()

    if backend == "gpu":
        device = "cuda"
        if x_flat.device.type == "cuda":
            x_dev = x_flat.to(dtype=torch.bfloat16).contiguous()
        else:
            x_dev = x_flat.to(device=device, dtype=torch.bfloat16)
        output = torch.empty_like(x_dev)
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        gelu_kernel_gpu[grid](x_dev, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # NPU path
        BLOCK_SIZE = 1024
        n_padded = math.ceil(n_elements / BLOCK_SIZE) * BLOCK_SIZE
        x_npu = x_flat.to(torch.bfloat16)
        if n_padded != n_elements:
            x_npu = torch.nn.functional.pad(x_npu, (0, n_padded - n_elements))
        x_npu = x_npu.contiguous()
        output = torch.empty(n_padded, dtype=torch.bfloat16)

        old_script = os.environ.get("AIR_TRANSFORM_TILING_SCRIPT")
        if transform_script:
            os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = transform_script

        grid = (n_padded // BLOCK_SIZE,)
        _gelu_npu_cached(gelu_kernel_npu, grid, x_npu, output, n_padded, BLOCK_SIZE=BLOCK_SIZE)

        if old_script is not None:
            os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = old_script
        elif transform_script:
            del os.environ["AIR_TRANSFORM_TILING_SCRIPT"]

        output = output[:n_elements].to(torch.float32)

    return output.reshape(orig_shape)
