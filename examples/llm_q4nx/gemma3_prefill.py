# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The Gemma3 prefill: a norm sandwich, two RoPE tables, and a GELU GLU.

The furthest any model here gets from the Llama block, and the only one that
differs in all four of the places a transformer block can:

* **Four norms, not two.** Each sublayer is wrapped: `input_layernorm` before
  attention and `post_attention_layernorm` after it, `pre_feedforward_layernorm`
  before the MLP and `post_feedforward_layernorm` after. The two post-norms
  apply to the sublayer output *before* it is added back to the residual, which
  is what makes them a sandwich rather than a second pre-norm.
* **Dual-theta RoPE.** Five layers in six are local (theta 1e4); every sixth is
  global (theta 1e6 with positions divided by 8). The table is chosen per layer,
  so `load_weights` builds both.
* **A sliding window**, 1024 tokens, on exactly the local layers. Note what this
  means for the gate: the canonical prompt is six tokens, so the window never
  binds there and a wrong window would pass it. What covers the window is the
  decode comparison against mlir-air's own generation.
* **GELU-tanh, not SiLU**, in the GLU. That is a new Triton kernel
  (`kernels.triton_geglu`); the rest of this file reuses existing ones.

It also has Qwen3's per-head qk-norm and Qwen3's decoupled q dim, so it
subclasses `LlamaPrefill` the same way and for the same reason: everything that
is not the forward -- operator routing, the timer, the bucketed padding, the KV
handoff -- is shared, and the block itself is written out.

Four Gemma conventions are resolved in the weight bundle rather than here (the
(1+w) norm fold, the embedding scale, the separately-stored unscaled LM head,
and the folded qk-norm weights). `config.load_q4nx` documents them; re-applying
any produces fluent wrong text.
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
    SLIDING_WINDOW,
    is_global_layer,
    rope_lut,
)

from llama_prefill import LlamaPrefill, _t


class Gemma3Prefill(LlamaPrefill):
    """Q4NX Gemma3 prefill producing the decode's KV handoff."""

    #: `geglu` in place of `swiglu`: Gemma's GLU activation is GELU-tanh, and
    #: the SiLU kernel would be a different function rather than a different
    #: schedule. Named as its own operator so `--ops` can bisect it.
    NPU_OPS = ("matmul", "rms_norm", "geglu")

    #: The two post-norms and the per-head q/k norms. Float32 for the same
    #: reason Qwen3's are: they are applied inside an f32 RMSNorm.
    EXTRA_LAYER_WEIGHTS = {
        "post_attn_norm": torch.float32,
        "post_ffn_norm": torch.float32,
        "q_norm": torch.float32,
        "k_norm": torch.float32,
    }

    def load_weights(self, model=None):
        """Llama's loader, plus the second RoPE table.

        `LlamaPrefill.load_weights` builds `self._lut` from `config.rope_lut`
        with no argument, which is the local table. The global one has to be
        built here because there is no single table for this model.
        """
        super().load_weights(model)
        self._lut_local = self._lut
        self._lut_global = (
            _t(rope_lut(self.max_seq, global_layer=True))
            .to(torch.bfloat16)
            .to(torch.float32)
        )

    def _geglu(self, gate, up, backend=None):
        """gelu_tanh(gate) * up, elementwise."""
        with self.timer.track("geglu"):
            if self._on_npu("geglu", backend):
                import kernels

                return kernels.triton_geglu(gate, up)
            return torch.nn.functional.gelu(gate, approximate="tanh") * up

    def _qk_norm(self, x, weight, n_heads, backend=None):
        """RMSNorm over each head's DH lanes. x: [N, n_heads*DH] -> same shape.

        Identical to Qwen3's, and for the same reason: Gemma normalizes within
        a head, so folding the head axis into the row axis makes this the
        ordinary `rms_norm` operator at DH columns.
        """
        N = x.shape[0]
        flat = x.reshape(N * n_heads, DH)
        return self._rms_norm(flat, weight, RMS_EPS, backend=backend).reshape(
            N, n_heads * DH
        )

    def _layer(self, x, L, N, keep=None):
        """One Gemma3 decoder block. x: [N, D] float32. Returns [N, D].

        `keep` is how many leading rows are real; only those reach the KV
        cache. The rest are sequence padding (see `SEQ_BUCKET`).
        """
        keep = N if keep is None else keep
        w = self._w[L]
        is_global = is_global_layer(L)
        lut = self._lut_global if is_global else self._lut_local

        # ---- attention sublayer, norm-sandwiched ----
        residual = x
        h = self._rms_norm(x, w["attn_norm"], RMS_EPS)  # input_layernorm

        qkv = self._matmul(h, w["qkv"])  # [N, DQ+DK+DV]
        q, k, v = qkv.split([DQ, DK, DV], dim=1)

        q = self._qk_norm(q, w["q_norm"], N_Q_HEADS)
        k = self._qk_norm(k, w["k_norm"], N_KV_HEADS)

        q = self._rope(q, lut[:N], N_Q_HEADS)
        k = self._rope(k, lut[:N], N_KV_HEADS)

        # The decode's handoff: roped K, raw V, head-major within a position.
        # Roped with *this layer's* table -- the cache is per layer, so the
        # local and global layers hold differently-rotated K.
        self.kv_k[L][:keep] = k[:keep].to(torch.float32).numpy()
        self.kv_v[L][:keep] = v[:keep].to(torch.float32).numpy()

        a = self._attention(
            q,
            k,
            v,
            N_Q_HEADS,
            N_KV_HEADS,
            DH,
            window=None if is_global else SLIDING_WINDOW,
        )
        a = self._matmul(a, w["o"])  # o contracts DQ -> D
        # post_attention_layernorm: on the sublayer output, before the residual
        # add. Moving it after the add would normalize the residual stream too,
        # which is a different model.
        x = residual + self._rms_norm(a, w["post_attn_norm"], RMS_EPS)

        # ---- MLP sublayer, norm-sandwiched the same way ----
        residual = x
        h = self._rms_norm(x, w["ffn_norm"], RMS_EPS)  # pre_feedforward
        g, u = self._matmul(h, w["gate_up"]).split([INTER, INTER], dim=1)
        d = self._matmul(self._geglu(g, u), w["down"])
        return residual + self._rms_norm(d, w["post_ffn_norm"], RMS_EPS)
