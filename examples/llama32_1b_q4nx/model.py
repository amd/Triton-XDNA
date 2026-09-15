# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Llama-3.2-1B prefill: the model, with each op routed to a backend.

The forward pass here is the contract; `ops.py` decides whether a given op
runs in torch on the CPU or as a Triton kernel on the NPU. Keeping the two
apart means a wrong answer can be bisected op by op against the same
reference, which is the only practical way to debug a 16-layer model whose
only end-to-end signal is one token id.

Emits exactly what mlir-air's fused decode expects; `save_kv_npz`
below states that contract.
"""

import numpy as np
import torch

from config import (
    D,
    DQ,
    DH,
    DK,
    DV,
    INTER,
    N_KV_HEADS,
    N_LAYERS,
    N_Q_HEADS,
    Q_PER_KV,
    RMS_EPS,
    VOCAB,
    load_q4nx,
    rope_lut,
)


def _t(a, dtype=torch.float32):
    """numpy (possibly ml_dtypes bfloat16) -> torch tensor."""
    if a.dtype.kind == "V" or str(a.dtype) == "bfloat16":
        a = a.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(a)).to(dtype)


class LlamaPrefill:
    """Q4NX Llama-3.2-1B prefill producing the decode's KV handoff.

    `ops` is the operator backend (see ops.py). The default runs everything in
    torch on the CPU, which is the reference the NPU path is checked against.
    """

    def __init__(self, ops=None, n_layers=N_LAYERS, max_seq=2048, model=None):
        from ops import TorchOps

        self.ops = ops if ops is not None else TorchOps()
        self.n_layers = n_layers
        self.max_seq = max_seq
        self.model = model
        self.current_context_length = 0
        # Per-layer KV cache: roped K and raw V, [max_seq, 512], head-major.
        self.kv_k = [np.zeros((max_seq, DK), np.float32) for _ in range(n_layers)]
        self.kv_v = [np.zeros((max_seq, DV), np.float32) for _ in range(n_layers)]
        self._w = None

    # ---- weights ----
    def load_weights(self, model=None):
        raw = load_q4nx(model or self.model)
        self.fingerprint = raw["fingerprint"]
        self.embed = raw["embed"]  # float32 [VOCAB, D], kept as numpy (1 GB)
        self.final_norm = _t(raw["final_norm"])
        self.lm_head = _t(raw["lm_head"], torch.bfloat16)  # tied to embed
        self._w = []
        for L in raw["layers"]:
            w = {
                "attn_norm": _t(L["attn_norm"]),
                "ffn_norm": _t(L["ffn_norm"]),
                **{
                    k: _t(L[k], torch.bfloat16)
                    for k in ("q", "k", "v", "o", "gate", "up", "down")
                },
            }
            # An NPU dispatch costs ~24 ms of fixed overhead regardless of
            # size, so projections that share an input are concatenated along
            # their output dimension and issued as one GEMM: Q|K|V (one
            # normalized hidden in) and gate|up. Same arithmetic, 48 fewer
            # launches per prefill.
            w["qkv"] = torch.cat([w["q"], w["k"], w["v"]], dim=1).contiguous()
            w["gate_up"] = torch.cat([w["gate"], w["up"]], dim=1).contiguous()
            self._w.append(w)
        # bf16 cos/sin, as the device applies them.
        self._lut = _t(rope_lut(self.max_seq)).to(torch.bfloat16).to(torch.float32)

    # ---- forward ----
    def _layer(self, x, L, N, keep=None):
        """One transformer block. x: [N, D] float32. Returns [N, D].

        `keep` is how many leading rows are real; only those reach the KV
        cache. The rest are sequence padding (see SEQ_BUCKET).
        """
        keep = N if keep is None else keep
        w = self._w[L]
        h = self.ops.rms_norm(x, w["attn_norm"], RMS_EPS)

        qkv = self.ops.matmul(h, w["qkv"])  # [N, 2048+512+512]
        q, k, v = qkv.split([DQ, DK, DV], dim=1)

        q = self.ops.rope(q, self._lut[:N], N_Q_HEADS)
        k = self.ops.rope(k, self._lut[:N], N_KV_HEADS)

        # The decode's handoff: roped K, raw V, head-major within a position.
        self.kv_k[L][:keep] = k[:keep].to(torch.float32).numpy()
        self.kv_v[L][:keep] = v[:keep].to(torch.float32).numpy()

        a = self.ops.attention(q, k, v, N_Q_HEADS, N_KV_HEADS, DH)  # [N, 2048]
        x = x + self.ops.matmul(a, w["o"])

        h = self.ops.rms_norm(x, w["ffn_norm"], RMS_EPS)
        g, u = self.ops.matmul(h, w["gate_up"]).split([INTER, INTER], dim=1)
        return x + self.ops.matmul(self.ops.swiglu(g, u), w["down"])

    #: Prompt lengths are rounded up to a multiple of this before the forward
    #: pass. Every NPU kernel is compiled for constexpr shapes and cached on
    #: them, so a prompt of a new length otherwise recompiles RMSNorm and
    #: SwiGLU -- seconds of latency on the first turn at each new length, which
    #: in a chat REPL is most turns. 128 matches the matmul's M block, so the
    #: GEMMs see no extra work at all.
    SEQ_BUCKET = 128

    def prefill(self, ids):
        """Run the prompt. Returns logits [VOCAB] for the final position."""
        assert self._w is not None, "call load_weights() first"
        N = len(ids)
        assert N <= self.max_seq, (N, self.max_seq)
        # Pad to a bucket. Safe under causal masking: the padded rows sit after
        # every real one, so no real position attends to them, and their own
        # outputs are discarded.
        Nb = min(
            self.max_seq,
            ((N + self.SEQ_BUCKET - 1) // self.SEQ_BUCKET) * self.SEQ_BUCKET,
        )
        x = torch.zeros((Nb, D), dtype=torch.float32)
        x[:N] = _t(self.embed[np.asarray(ids)])
        for L in range(self.n_layers):
            x = self._layer(x, L, Nb, keep=N)
        self.current_context_length = N
        # Final norm on the prediction row only, then the LM head.
        xn = self.ops.rms_norm(x[N - 1 : N], self.final_norm, RMS_EPS)
        return self.ops.lm_head(xn, self.lm_head)[0]

    # ---- decode handoff (mirrors mlir-air's causal_lm interface) ----
    def kv_view(self, layer_idx):
        c = self.current_context_length
        return self.kv_k[layer_idx][:c], self.kv_v[layer_idx][:c]

    def get_current_context_length(self):
        return self.current_context_length

    def clear_context(self):
        self.current_context_length = 0
        for L in range(self.n_layers):
            self.kv_k[L][:] = 0
            self.kv_v[L][:] = 0

    def save_kv_npz(self, path, first, prompt):
        """Write the handoff mlir-air's `generate()` loads.

        The contract, derived from both ends -- what its `kv_view()` returns
        and what its `seed_kv()` assumes -- because getting any of it wrong
        degrades quality without failing:

            k, v : [16, P, 512] float32   (bf16-exact values)

        512 is 8 KV heads x 64, laid out as column `h*64 + d`, position-major,
        heads contiguous within a position. No permutation and no interleaving
        at this boundary: the region-major scatter the decode wants happens
        inside its own `seed_kv()`.

        K is stored already rotated, V raw. The rotation is half-split --
        `out[i] = x[i]*cos[i] - x[i+32]*sin[i]`, pairing i with i+32 rather
        than adjacent lanes -- and its table carries llama3 frequency scaling
        (factor 32, low 1, high 4, old context 8192, theta 500000), which is
        why `config.rope_lut` re-exports mlir-air's generator instead of
        rebuilding one: a reimplementation that drops the scaling rotates K on
        the wrong frequencies and only long prompts show it.
        """
        c = self.current_context_length
        K = np.stack([self.kv_k[L][:c] for L in range(self.n_layers)])
        V = np.stack([self.kv_v[L][:c] for L in range(self.n_layers)])
        np.savez(
            path,
            k=K.astype(np.float32),
            v=V.astype(np.float32),
            first=first,
            prompt=np.array(prompt),
        )
        return K, V
