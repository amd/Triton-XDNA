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

One thing here is not about Qwen2.5 at all: which matmul schedule the
model-dim GEMMs are lowered with. The shared hand-written schedule does not
lower every width, and Qwen2.5-7B's fused `gate_up` is past what it takes --
so the choice is read from each model's `config` (`MATMUL_GENERATED_SCHEDULE`)
rather than fixed for the family. Qwen2.5-3B, on the same forward, does not
need it. See `GENERATED_SCHEDULE`.
"""

import torch

import config as _config

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

    #: Whether this model's GEMMs need the driver-GENERATED matmul schedule in
    #: place of the hand-written one every other model here uses. A per-model
    #: fact, so it is read from the model's own `config` rather than fixed for
    #: the family: Qwen2.5-7B needs it and Qwen2.5-3B does not.
    #:
    #: What forces it, on the 7B: `gate_up`, whose fused output is 2*18944 =
    #: 37888 columns. The shared schedule emits a DMA descriptor with stride
    #: 2424832 for it, against a hardware range of [1, 1048576], and aiecc
    #: rejects it. Measured, not reasoned: N=32768 lowers under the shared
    #: schedule and N=37888 does not, so the boundary is between the two and is
    #: not simply "N is large". The 3B's fused `gate_up` is 22016 columns, under
    #: the largest width measured to lower.
    #:
    #: It is one flag for `qkv`, `o` and `gate_up` together, and that is not a
    #: preference. All three share (BLOCK_M, BLOCK_N, BLOCK_K), Triton's cache
    #: keys on constexprs and not on the schedule, and so one binary serves all
    #: three: asking for two schedules across them means one silently gets the
    #: other's. `kernels._check_script` raises rather than allowing it.
    #:
    #: `down` is excluded deliberately. It lands on its own constexprs (its K is
    #: the MLP dim, not the model dim), so it can and should keep the shared
    #: schedule -- on the 7B the generated one did not finish lowering it in 25
    #: minutes, where the shared one takes seconds. Its narrower weight tile is
    #: derived inside `kernels.triton_matmul`, not declared here.
    GENERATED_SCHEDULE = getattr(_config, "MATMUL_GENERATED_SCHEDULE", False)

    @property
    def _wide(self):
        """Schedule override for the three model-dim GEMMs, as kwargs.

        Empty when the shared schedule serves, so `triton_matmul` keeps its own
        default rather than this file restating it.
        """
        return {"transform_script": None} if self.GENERATED_SCHEDULE else {}

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
        qkv = self._matmul(h, w["qkv"], **self._wide) + w["qkv_bias"]
        q, k, v = qkv.split([DQ, DK, DV], dim=1)

        q = self._rope(q, self._lut[:N], N_Q_HEADS)
        k = self._rope(k, self._lut[:N], N_KV_HEADS)

        # The decode's handoff: roped K, raw V, head-major within a position.
        # Both carry their bias -- it was added before the split.
        self.kv_k[L][:keep] = k[:keep].to(torch.float32).numpy()
        self.kv_v[L][:keep] = v[:keep].to(torch.float32).numpy()

        a = self._attention(q, k, v, N_Q_HEADS, N_KV_HEADS, DH)  # [N, DQ]
        x = x + self._matmul(a, w["o"], **self._wide)

        h = self._rms_norm(x, w["ffn_norm"], RMS_EPS)
        g, u = self._matmul(h, w["gate_up"], **self._wide).split([INTER, INTER], dim=1)
        # `down` keeps the shared schedule and the default block_n; its tile
        # width is narrowed by triton_matmul itself where K demands it.
        return x + self._matmul(self._swiglu(g, u), w["down"])
