# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4-E2B's tensors in FastFlowLM's model.q4nx bundle, by layer.

The int4 projections come back as stored, (q, scale, min) with w = q *
scale + min; the norms, PLE matrices and tables as float32.
"""

import numpy as np

import config as C
from q4nx_bundle import Bundle, resolve

REPO = "FastFlowLM/Gemma4-E2B-IT-NPU2"


class Gemma4Weights:
    def __init__(self, model=None):
        self.bundle = Bundle(resolve(model, REPO))
        b = self.bundle
        for L in range(C.N_LAYERS):
            got = b.int4_rows(f"model.layers.{L}.mlp.gate_proj.weight", C.D)
            if got != C.mlp_inter(L):
                raise ValueError(f"layer {L}: FFN width {got}, not {C.mlp_inter(L)}")

    def fingerprint(self):
        return self.bundle.fingerprint()

    def projections(self, L):
        """Layer L's int4 matrices [out, in] by name: q, o, gate, up, down,
        and k, v on the layers that own their KV."""
        dq = C.N_Q_HEADS * C.head_dim(L)
        dkv = C.N_KV_HEADS * C.head_dim(L)
        inter = C.mlp_inter(L)
        p = f"model.layers.{L}."
        shapes = dict(
            q=("self_attn.q_proj", dq, C.D),
            o=("self_attn.o_proj", C.D, dq),
            gate=("mlp.gate_proj", inter, C.D),
            up=("mlp.up_proj", inter, C.D),
            down=("mlp.down_proj", C.D, inter),
        )
        if C.owns_kv(L):
            shapes |= dict(
                k=("self_attn.k_proj", dkv, C.D), v=("self_attn.v_proj", dkv, C.D)
            )
        return {
            k: self.bundle.int4(p + name + ".weight", rows, K)
            for k, (name, rows, K) in shapes.items()
        }

    def norms(self, L):
        p = f"model.layers.{L}."
        w = lambda name: self.bundle.bf16(p + name + ".weight")  # noqa: E731
        return dict(
            input=w("input_layernorm"),
            post_attn=w("post_attention_layernorm"),
            pre_ffn=w("pre_feedforward_layernorm"),
            post_ffn=w("post_feedforward_layernorm"),
            post_ple=w("post_layernorm"),
            q_norm=w("self_attn.q_norm"),
            k_norm=w("self_attn.k_norm"),
            out_scale=float(w("layer_output_scale")[0]),
        )

    def ple(self, L):
        """Layer L's PLE matrices [out, in]: gate [PLI_D, D], proj [D, PLI_D]
        and its rows of the model projection [PLI_D, D]. The bundle stores
        each as tiles of 32 output rows."""
        b, p, nt = self.bundle, f"model.layers.{L}.", C.PLI_D // 32

        def rows32(name, rows, cols):
            t = b.bf16(name).reshape(rows // 32, cols, 32)
            return np.ascontiguousarray(t.transpose(0, 2, 1)).reshape(rows, cols)

        return dict(
            gate=rows32(p + "inp_gate.weight", nt * 32, C.D),
            proj=rows32(p + "per_layer_projection.weight", C.D, C.PLI_D),
            model_proj=rows32(
                f"model.per_layer_model_proj.weight_layer{L}", nt * 32, C.D
            ),
        )

    def final_norm(self):
        return self.bundle.bf16("model.norm.weight")

    def ple_proj_norm(self):
        return self.bundle.bf16("model.per_layer_proj_norm.weight")

    def rope_freqs(self):
        """The full-attention layers' frequency divisors, or None."""
        b = self.bundle
        return b.bf16("rope_freqs.weight") if b.has("rope_freqs.weight") else None

    def embed_table(self):
        """The token embeddings as (int8 [VOCAB, D], group-32 scales); a row
        times its scales times EMBED_SCALE is a token's embedding."""
        return self.bundle.int8_table("model.embed_tokens.weight")

    def ple_embed_table(self):
        """The per-layer token embeddings as (int8 [VOCAB, N_LAYERS * PLI_D],
        group-32 scales)."""
        return self.bundle.int8_table("model.per_layer_token_embd.weight")

    def lm_head(self, r0, rows):
        """Rows [r0, r0 + rows) of the LM head as (q, scale, min)."""
        return self.bundle.int4("lm_head.weight", rows, C.D, r0=r0)
