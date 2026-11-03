# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4 vision attention on the NPU, through `tl.extra.npu.flash_attention`.

The vision tower's attention is multi-head (12 heads, head dim 64), runs over
every real patch with no mask, and is quadratic in the patch count: at the
256-token image budget it was two thirds of the encoder on the CPU. The op
builds mlir-air's non-causal flash attention for the patch count; this wraps
it for the tower:

* Scale. The kernel scales scores by 1/sqrt(head_dim) and Gemma4's vision
  attention uses 1.0, so Q is multiplied by `scale * sqrt(head_dim)`, which for
  head dim 64 is an exact power of two in bf16.
* Precision. The microkernel is built in block-float mode (`bfp16`), about 3x
  the bf16 build at 2304 rows. The vision gate (`gemma4_e2b_flash/vision.py
  --backend npu`) holds with it at both image budgets.
* Dispatch. Through Triton-XDNA's `MultiLaunchRunner`, not mlir-air's
  `KernelCache`, which takes `/tmp/npu.lock` around every dispatch and so would
  deadlock under a caller that already holds it.
"""

import math

import numpy as np
import torch

import airsrc

#: The microkernel's block-float mode; see the module docstring.
BFP16 = True


class NPUVisionAttention:
    """Non-causal multi-head attention over `n` rows on the NPU."""

    @staticmethod
    def supports(n, head_dim):
        import triton.language as tl

        try:
            tl.extra.npu.AttentionConfig(n, 2, head_dim=head_dim)
        except tl.extra.npu.AttentionConfigError:
            return False
        return True

    def __init__(self, n, n_heads, head_dim):
        import triton.language as tl
        from triton.backends.amd_triton_npu.multilaunch import MultiLaunchRunner

        npu = tl.extra.npu
        cfg = npu.AttentionConfig(n, n_heads, head_dim=head_dim, bfp16=BFP16)
        built = npu.flash_attention(cfg, airsrc.flash_attention_dir())
        self.n, self.d = n, head_dim
        self.q_rows = built["q_rows"]
        self._runner = MultiLaunchRunner(built["elf_path"], built["kernel"])
        self._key = f"vision_attn_{built['fingerprint']}"
        self._q = np.zeros((self.q_rows, n_heads * head_dim), np.uint16)
        self._out = np.zeros_like(self._q)

    def __call__(self, q, k, v, scale):
        """q/k/v: [n, heads * head_dim] float -> [n, heads * head_dim] float32."""
        bf = torch.bfloat16
        n = self.n

        def raw(t):
            return t.to(bf).contiguous().view(torch.int16).numpy().view(np.uint16)

        self._q[:n] = raw(q * (scale * math.sqrt(self.d)))
        got = self._runner.run(
            [self._q, raw(k), raw(v), self._out],
            bo_key=self._key,
            intermediate_indices={3},
            output_indices=[3],
        )[3]
        rows = np.asarray(got).reshape(self.q_rows, -1)[:n].view(np.int16)
        return torch.from_numpy(rows.copy()).view(bf).to(torch.float32)
