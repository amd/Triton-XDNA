# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The Phi-4-mini prefill: Llama's block with RoPE over part of each head.

Phi-4-mini is Llama-shaped everywhere the operator routing cares about -- one
norm pair per block, fused QKV, half-split RoPE, SwiGLU, GQA (3 q heads per kv
head), a tied LM head -- and differs in one: `partial_rotary_factor=0.75`, so
the rotation covers only the leading 96 of each head's 128 lanes and the
trailing 32 pass through untouched.

That is a `_rope` override rather than a `_layer` one. The block is Llama's; it
is the operator that changes, and it changes the same way for q and for k. Every
other model here rotates the whole head, and the inherited `_rope` derives the
head width from the LUT -- so handing it a 96-wide LUT would not rotate 96 lanes
of a 128-wide head, it would reinterpret each head as 96 lanes and corrupt the
head boundary. That failure is silent.

Two further Phi-4 facts, both resolved in the model's `config`:

* **LongRoPE.** The frequencies are not `1/theta**(2i/d)` but that divided by a
  per-frequency factor table shipped in the bundle, and both cos and sin are
  scaled by `sqrt(1 + ln(max/orig)/ln(orig))`. `config.rope_lut` takes the whole
  table from mlir-air, which also picks the short or long factor table by
  sequence length.
* **The LM head is tied**, as on Llama-3.2-1B/3B. The bundle does carry a
  separate quantized `lm_head`, and it is the wrong one to use -- it is a
  lossier copy of the embedding the checkpoint ties to.

Not handled here, because Phi-4-mini does not have them: Qwen3's qk-norm,
Qwen2.5's projection bias, Gemma's norm sandwich.
"""

import torch

from config import ROPE_DIM

from llama_prefill import LlamaPrefill


class Phi4Prefill(LlamaPrefill):
    """Q4NX Phi-4-mini prefill producing the decode's KV handoff."""

    #: How many of each head's lanes are rotated. The rest are copied through,
    #: which is what `rope_partial` in the decode's kernel does.
    ROPE_DIM = ROPE_DIM

    def _rope(self, x, lut, n_heads, backend=None):
        """Half-split RoPE over the leading ROPE_DIM lanes of each head.

        x:   [N, n_heads*DH]
        lut: [N, ROPE_DIM] = [cos_0..cos_47, sin_0..sin_47]

        Llama's rotation, applied to a slice. With R = ROPE_DIM and H = R // 2,
        within each head:

            out[i]     = x[i]*cos[i] - x[i+H]*sin[i]      for i < H
            out[i+H]   = x[i]*sin[i] + x[i+H]*cos[i]
            out[R:]    = x[R:]                            unrotated

        The pairing is (i, i+H) inside the rotated slice -- so (0, 48) for
        Phi-4-mini, not (0, 64). Pairing across the whole head instead would
        rotate lane 0 against lane 64, which is in the *unrotated* tail, and
        produce fluent wrong text rather than an error.

        Goes to the iGPU under `hetero`, like the rotation it overrides. The
        partial width is the kernel's `rot`; the tail it copies through is the
        same one the torch body concatenates unchanged.
        """
        with self.timer.track("rope"):
            N = x.shape[0]
            R = self.ROPE_DIM
            half = R // 2
            if lut.shape[-1] != R:
                raise ValueError(
                    f"partial RoPE needs a {R}-wide LUT (cos|sin of {half}); "
                    f"got {lut.shape[-1]}"
                )
            dev = self._gpu_device(backend)
            if dev is not None:
                import gpu_kernels

                with gpu_kernels.gpu_driver():
                    out = gpu_kernels.rope_batch(
                        x.to(dev), lut.to(dev), n_heads, x.shape[-1] // n_heads, rot=R
                    )
                return out.cpu()
            cos = lut[:, :half].unsqueeze(1)  # [N, 1, half]
            sin = lut[:, half:].unsqueeze(1)
            v = x.reshape(N, n_heads, -1)
            x1, x2, tail = v[..., :half], v[..., half:R], v[..., R:]
            return torch.cat(
                [x1 * cos - x2 * sin, x1 * sin + x2 * cos, tail], dim=-1
            ).reshape(N, -1)
