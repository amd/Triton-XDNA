# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The Qwen3 prefill: Llama's block with a per-head norm on q and k.

Qwen3 is Llama-shaped everywhere the operator routing cares about -- one norm
pair per block, fused QKV, half-split (NEOX) RoPE, SwiGLU, GQA -- and differs
in exactly one place: between the QKV projection and RoPE, each head's 128
lanes are RMS-normalized by a per-model weight (`q_norm`, `k_norm`). That is
enough to need its own `_layer`, and not enough to need its own class, so this
subclasses `LlamaPrefill` for everything that is not the forward: the
backend-per-operator routing, the timer, `prefill()`'s bucketed padding and the
KV handoff.

The delta is deliberately a whole overridden `_layer` rather than a flag inside
Llama's. A branch there would be read by three models that must not take it,
and the failure mode of taking it wrongly is fluent wrong text, not an error.

Two further Qwen3 facts, both handled in the model's `config` rather than here:

* **The RoPE table is single-theta** (1e6), with none of llama3's frequency
  scaling. `config.rope_lut` builds it from mlir-air's own per-position
  generator, so the table this forward rotates K with is the one the fused
  decode rotates its new tokens with.
* **The LM head is tied** on Qwen3-4B (`tie_word_embeddings=true`), so the
  inherited `_lm_head` against the embedding matrix is correct. Qwen3-8B is
  *not* tied; a model added here must be checked rather than assumed.

Not handled here, because Qwen3 does not have them: Gemma's (1+w) norm fold,
its embedding scale, its dual-theta RoPE and its sliding window.
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


class Qwen3Prefill(LlamaPrefill):
    """Q4NX Qwen3 prefill producing the decode's KV handoff."""

    #: The per-head q/k norm weights, length DH each. Read as float32 and kept
    #: there: they are applied inside an RMSNorm whose own accumulation is
    #: f32, so rounding them to bf16 here would round twice.
    EXTRA_LAYER_WEIGHTS = {"q_norm": torch.float32, "k_norm": torch.float32}

    def _qk_norm(self, x, weight, n_heads, backend=None):
        """RMSNorm over each head's DH lanes. x: [N, n_heads*DH] -> same shape.

        Qwen3 normalizes *within* a head, not across the projection, so the
        rows handed to the norm are (position, head) pairs. Folding the head
        axis into the row axis is what makes this the ordinary `rms_norm`
        operator -- and so the same Triton kernel, at DH=128 columns -- rather
        than a new one.
        """
        N = x.shape[0]
        flat = x.reshape(N * n_heads, DH)
        return self._rms_norm(flat, weight, RMS_EPS, backend=backend).reshape(
            N, n_heads * DH
        )

    def _layer(self, x, L, N, keep=None):
        """One Qwen3 decoder block. x: [N, D] float32. Returns [N, D].

        Llama's block with the two `_qk_norm` calls added, and o_proj no longer
        square: Qwen3-4B has DQ=4096 against D=2560, so attention output and
        residual have different widths and only the projection reconciles them.

        `keep` is how many leading rows are real; only those reach the KV
        cache. The rest are sequence padding (see `SEQ_BUCKET`).
        """
        keep = N if keep is None else keep
        w = self._w[L]
        h = self._rms_norm(x, w["attn_norm"], RMS_EPS)

        qkv = self._matmul(h, w["qkv"])  # [N, DQ+DK+DV]
        q, k, v = qkv.split([DQ, DK, DV], dim=1)

        # The Qwen3 delta. Before RoPE, never after: the norm is scale-only and
        # the rotation is not, so swapping them changes the result.
        q = self._qk_norm(q, w["q_norm"], N_Q_HEADS)
        k = self._qk_norm(k, w["k_norm"], N_KV_HEADS)

        q = self._rope(q, self._lut[:N], N_Q_HEADS)
        k = self._rope(k, self._lut[:N], N_KV_HEADS)

        # The decode's handoff: roped K, raw V, head-major within a position.
        # V is *not* qk-normed -- only q and k are.
        self.kv_k[L][:keep] = k[:keep].to(torch.float32).numpy()
        self.kv_v[L][:keep] = v[:keep].to(torch.float32).numpy()

        a = self._attention(q, k, v, N_Q_HEADS, N_KV_HEADS, DH)  # [N, DQ]
        x = x + self._matmul(a, w["o"])  # o contracts DQ -> D

        h = self._rms_norm(x, w["ffn_norm"], RMS_EPS)
        g, u = self._matmul(h, w["gate_up"]).split([INTER, INTER], dim=1)
        return x + self._matmul(self._swiglu(g, u), w["down"])
