# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Matrix multiply dispatched through the HSA/ROCR runtime.

Selects the HSA launch runtime instead of XRT. Under HSA the backend produces
PDI + insts artifacts and dispatches them via the AIE agent path
(hsa_amd_aie_kernel_dispatch_packet_t).

Run it::

    python hsa_matmul.py

The tiling script defaults to the one matching the detected device (see below);
override it with ``AIR_TRANSFORM_TILING_SCRIPT=...`` if needed.

Requires a ROCR install with the HSA runtime; set AMD_NPU_ROCR_PATH / ROCM_PATH if it is
not in the default location. This example is validated end-to-end on npu2
(AIE2P). It selects the runtime programmatically below with ``NPUDriver("hsa")``;
you can equivalently export ``AMD_TRITON_NPU_RUNTIME=hsa`` and use ``NPUDriver()``.

Note: a transform tiling script is required (as with the XRT matmul examples).
The default elementwise tiling path currently has an npu2 aircc-legalization
limitation shared with the XRT backend, so a matmul + transform script is the
validated HSA path.
"""

import os
import torch
import triton
import triton.language as tl

from triton.backends.amd_triton_npu.config import npu_config
from triton.backends.amd_triton_npu.driver import NPUDriver, detect_npu_version

# Select the HSA/ROCR launch runtime.
triton.runtime.driver.set_active(NPUDriver("hsa"))

# Default the tiling script to the one matching the detected device (AIE2 for
# npu1, AIE2P for npu2) unless the caller supplied one explicitly.
_HERE = os.path.dirname(os.path.abspath(__file__))
if not os.environ.get("AIR_TRANSFORM_TILING_SCRIPT"):
    _script = (
        "transform_aie2.mlir"
        if detect_npu_version() == "npu1"
        else "transform_aie2p.mlir"
    )
    npu_config.transform_tiling_script = os.path.join(_HERE, _script)


@triton.jit
def bare_matmul(
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
    a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c = tl.dot(a, b)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)


def bench_matmul(M, N, K):
    a = torch.randn((M, K), dtype=torch.bfloat16)
    b = torch.randn((K, N), dtype=torch.bfloat16)
    c = torch.empty((M, N), dtype=torch.float32)
    c_ref = torch.matmul(a, b).to(torch.float32)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    bare_matmul[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=K,
    )
    torch.testing.assert_close(c, c_ref, atol=1e1, rtol=1e-1)
    print(
        f"HSA matmul {M}x{N}x{K}: PASS (max abs diff {(c - c_ref).abs().max().item():.3f})"
    )


if __name__ == "__main__":
    bench_matmul(256, 256, 256)
