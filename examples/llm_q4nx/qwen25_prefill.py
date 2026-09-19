# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The Qwen2.5 prefill: Llama's block with a bias on the QKV projection.

Qwen2.5 is Llama-shaped everywhere else -- one norm pair per block, fused QKV,
half-split (NEOX) RoPE, SwiGLU, GQA, single-theta RoPE, causal attention with no
window -- and differs in exactly one place: q, k and v each carry a bias vector,
added to the raw projection output *before* RoPE. Every other model here has
`y = x @ W` and nothing else, so that one addition is what this file is for.

It is a whole overridden `_layer` rather than a flag inside Llama's, for the
reason `qwen3_prefill` gives: a branch there would be read by four models that
must not take it, and taking it wrongly is fluent wrong text rather than an
error.

Note what this is NOT. Qwen2.5 has **no qk-norm** -- that is the Qwen3
convention, and applying it here would be a different model. It also has no
post-norms, no second RoPE theta and no sliding window.

One fact is handled in the model's `config` rather than here: the three biases
arrive already concatenated as a single `qkv_bias`, because the projection they
correct is itself fused into one GEMM. `config.load_q4nx` documents it.

What is *not* about Qwen2.5 at all, and is handled here because this is the
first model to reach it, is the size of its MLP: `INTER` is 18944, half again
the widest before it. Both GEMMs that touch it fall outside what the shared
matmul defaults lower -- see `_layer`.
"""

import torch

from config import (
    DH,
    DK,
    DQ,
    DV,
    INTER,
    N_KV_HEADS,
    N_Q_HEADS,
    RMS_EPS,
)

from llama_prefill import LlamaPrefill


class Qwen25Prefill(LlamaPrefill):
    """Q4NX Qwen2.5 prefill producing the decode's KV handoff."""

    #: The q/k/v projection biases, concatenated to match the fused QKV GEMM,
    #: so length DQ+DK+DV. Float32 because the add happens before the bf16
    #: round that follows the projection -- mlir-air's reference forward is
    #: `_bf(_bf(h @ w) + b)`, and rounding the bias itself first would round
    #: twice. They are not quantized on either side; the reference design
    #: leaves them, like the norms, in bf16.
    EXTRA_LAYER_WEIGHTS = {"qkv_bias": torch.float32}

    #: The driver-generated matmul schedule rather than the hand-written one
    #: every other model here uses, for the three GEMMs that pad K to 4096.
    #:
    #: Forced by `gate_up`, whose fused output is 2*18944 = 37888 columns. The
    #: shared schedule emits a DMA descriptor with stride 2424832 for it,
    #: against a hardware range of [1, 1048576], and aiecc rejects it. The
    #: generated schedule lowers the same shape. Measured, not reasoned:
    #: N=32768 lowers under the shared schedule and N=37888 does not, so the
    #: boundary is somewhere between and is not simply "N is large".
    #:
    #: It applies to `qkv` and `o` as well, and that is not a preference. All
    #: three share (BLOCK_M=128, BLOCK_N=256, BLOCK_K=4096), Triton's cache
    #: keys on constexprs and not on the schedule, and so one binary serves all
    #: three: asking for two schedules across them means one of them silently
    #: gets the other's. `kernels._check_script` now raises rather than
    #: allowing it.
    WIDE_SCRIPT = None

    #: `down` contracts INTER=18944, which pads to 32768, and its weight tile
    #: is BLOCK_K x BLOCK_N elements. At the default 256 that is 8388608,
    #: twice Triton's 4194304 per-tensor maximum -- a frontend limit, not a
    #: device one. 128 is the largest that fits. It also puts `down` on its own
    #: constexprs, which is why it can keep the shared schedule (and should:
    #: the generated one did not finish lowering this shape in 25 minutes).
    DOWN_BLOCK_N = 128

    def _layer(self, x, L, N, keep=None):
        """One Qwen2.5 transformer block, on the prompt. x: [N, D] -> [N, D].

        "Block", not "decoder block" -- see `qwen3_prefill._layer` for why the
        usual name is avoided in these files: "decode" here means mlir-air's
        fused per-token kernel, the other half of the split, and this is the
        prefill.

        Llama's block with one line added. The bias goes on the raw projection
        output and before RoPE, which is the order `fused_decode/kernels/rope.cc`
        applies it in (`add_q_k_v_bias`, then `apply_rope`). RoPE is a rotation
        and the bias is a translation, so the two do not commute: adding it
        afterwards would be a different model that still produces fluent text.

        `keep` is how many leading rows are real; only those reach the KV
        cache. The rest are sequence padding (see `SEQ_BUCKET`).
        """
        keep = N if keep is None else keep
        w = self._w[L]
        h = self._rms_norm(x, w["attn_norm"], RMS_EPS)

        # The Qwen2.5 delta. One add on the fused [DQ+DK+DV] output rather than
        # three on the split halves: same arithmetic, and it keeps the bias
        # laid out the way the GEMM that produced the tensor already is.
        qkv = (
            self._matmul(h, w["qkv"], transform_script=self.WIDE_SCRIPT) + w["qkv_bias"]
        )
        q, k, v = qkv.split([DQ, DK, DV], dim=1)

        q = self._rope(q, self._lut[:N], N_Q_HEADS)
        k = self._rope(k, self._lut[:N], N_KV_HEADS)

        # The decode's handoff: roped K, raw V, head-major within a position.
        # Both carry their bias -- it was added before the split.
        self.kv_k[L][:keep] = k[:keep].to(torch.float32).numpy()
        self.kv_v[L][:keep] = v[:keep].to(torch.float32).numpy()

        a = self._attention(q, k, v, N_Q_HEADS, N_KV_HEADS, DH)  # [N, DQ]
        x = x + self._matmul(a, w["o"], transform_script=self.WIDE_SCRIPT)

        h = self._rms_norm(x, w["ffn_norm"], RMS_EPS)
        g, u = self._matmul(h, w["gate_up"], transform_script=self.WIDE_SCRIPT).split(
            [INTER, INTER], dim=1
        )
        return x + self._matmul(
            self._swiglu(g, u), w["down"], block_n=self.DOWN_BLOCK_N
        )
