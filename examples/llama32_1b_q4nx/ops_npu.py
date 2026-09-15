# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NPU operator backend: Triton kernels on XDNA, falling back to TorchOps.

Each op can be enabled independently (`NpuOps(ops="matmul,rope")` or the
`LLAMA_NPU_OPS` environment variable), so a regression can be bisected to one
kernel without rebuilding the model. Anything not enabled runs the CPU
reference from ops.py.

Attention and the LM head stay on the CPU for now -- there is no NPU attention
transform script yet, exactly as `examples/gpt2 --backend npu` does today.
"""

import os

import kernels
from ops import TorchOps

ALL_OPS = ("matmul", "rms_norm", "swiglu")

# RoPE stays on the CPU, the same position examples/gpt2 takes on attention,
# and is ~0.1% of prefill arithmetic.
DEFAULT_OPS = "all"


class NpuOps(TorchOps):
    """TorchOps with selected operators dispatched to the NPU."""

    name = "npu"

    def __init__(self, ops=None):
        spec = ops if ops is not None else os.environ.get("LLAMA_NPU_OPS", DEFAULT_OPS)
        if spec in ("all", "*"):
            enabled = set(ALL_OPS)
        else:
            enabled = {o.strip() for o in spec.split(",") if o.strip()}
        unknown = enabled - set(ALL_OPS)
        if unknown:
            raise ValueError(f"unknown ops {sorted(unknown)}; known: {ALL_OPS}")
        self.enabled = enabled

    def matmul(self, x, w):
        if "matmul" not in self.enabled:
            return super().matmul(x, w)
        return kernels.matmul(x, w)

    def rms_norm(self, x, weight, eps):
        if "rms_norm" not in self.enabled:
            return super().rms_norm(x, weight, eps)
        return kernels.rms_norm(x, weight, eps)

    def swiglu(self, gate, up):
        if "swiglu" not in self.enabled:
            return super().swiglu(gate, up)
        return kernels.swiglu(gate, up)
