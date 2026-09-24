# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The Gemma4 prefill: per-layer embeddings, and layers that differ in shape.

The largest departure from the Llama block in this directory. Gemma3 already
had the norm sandwich, dual-theta RoPE, a sliding window and GELU-tanh; Gemma4
keeps all four and adds three things none of the other models here have:

* **Per-layer embeddings (PLE).** Every layer gets its own 256-wide vector per
  token, injected after the MLP through a gated projection and a fifth norm.
  The vectors are computed **once, from the input embeddings**, before the
  layer loop -- not per layer from that layer's hidden state. Upstream measured
  the difference: feeding the layer's own `x` scores the projection at cosine
  0.30 on layer 8, feeding the embeddings scores 1.000000. It is nearly
  invisible when wrong, because the shape and scale are right either way and it
  only drifts with depth.
* **Layers that are not all the same shape.** Four sliding layers then one
  full, repeating: the sliding ones have a 256-wide head, a 512-token window
  and RoPE theta 1e4, the full ones a 512-wide head, no window and theta 1e6.
  And from layer 15 up the FFN doubles to 12288.
* **Twenty layers with no KV of their own.** Those read a lower layer's cache.
  They never project k or v -- `config.load_q4nx` does not even load the
  tensors -- so the handoff repeats the source layer's arrays.

Two smaller things that are quiet when wrong and so are worth naming here:
`ATTN_SCALE` is 1.0 rather than `head_dim**-0.5`, and each block's output is
multiplied by a per-layer scalar (`out_scale`) before it becomes the next
layer's input.

Because the per-layer shapes vary, this class owns its `load_weights`,
`__init__` and `kv_stack` rather than inheriting them -- the base class's fuse
the QKV and gate/up projections once at fixed widths, and allocate one KV shape
for every layer. Everything that is not shape-dependent -- operator routing,
the timer, the bucketed padding -- is still shared.

The partial rotary on the full layers is **not** implemented here. It lives in
the frequency table: the bundle ships a divisor of ~1e30 for the dead lanes,
which makes cos 1 and sin 0 there. So the ordinary half-split `_rope` is
correct and the pairing stays (i, i + dh/2). See `config.rope_lut`.

**Do not gate this model's K against mlir-air's `forward_prompt`.** Its
`apply_rope` aliases: `a` and `b` are numpy *views* into the output, so the
write to the first half lands before the second half is computed, and the
second half is rotated with the already-updated `a` rather than the original.
The device does not do this -- `fused_decode/kernels/rope.cc` loads both halves
into vectors first and documents the plain form, `C = A cos - B sin,
D = A sin + B cos` -- and neither does this file.

Measured, because the difference is easy to mistake for a bug on our side: our
K scores cosine 0.94 against that oracle and **0.999999 against the same
oracle with the aliasing removed**, on every layer, K and V alike (worst
0.999991 over the 15 layers that own a cache). The first token is 9079 either
way, which is why it survives their gate.

The consequence for verification is the part worth remembering:
`generate(..., numpy_prefill=True)` -- the cheap reference the other models
here use -- seeds the decode from that aliased oracle, so on **this** model it
is not a reference at all. Gate against mlir-air's own NPU prefill instead.
"""

import contextlib
import os

import numpy as np
import torch

from config import (
    ATTN_SCALE,
    D,
    FINAL_LOGIT_SOFTCAP,
    FIRST_KV_SHARED,
    N_LAYERS,
    N_KV_HEADS,
    N_Q_HEADS,
    PLE_INPUT_SCALE,
    PLE_MODEL_PROJ_SCALE,
    PLI_D,
    RMS_EPS,
    SLIDING_WINDOW,
    head_dim,
    is_sliding,
    kv_source_layer,
    load_q4nx,
    mlp_inter,
    rope_lut,
)

import gpu_kernels

from llama_prefill import LlamaPrefill, _as_bf16, _t


def _host_view(buf):
    """The numpy view of a KV slot, whether it is a shared buffer or an array."""
    if buf is None:
        return None
    return buf.numpy() if hasattr(buf, "numpy") else buf


class Gemma4Prefill(LlamaPrefill):
    """Q4NX Gemma4-E2B prefill producing the decode's KV handoff."""

    #: `geglu` in place of `swiglu`, as for Gemma3: the GLU activation is
    #: GELU-tanh, which is a different function rather than a different
    #: schedule. Named as its own operator so `--ops` can bisect it.
    NPU_OPS = ("matmul", "rms_norm", "geglu")

    #: `rms_norm` is not in the default. Its NPU kernel is correct, but this
    #: model issues one per sublayer plus the per-head q/k/v norms plus the PLE
    #: norm -- some hundreds per prefill, each over rows a few tokens wide. At
    #: that size the dispatch costs more than the arithmetic, and it splits the
    #: GEMMs either side of it into separate staged launches. Running it on the
    #: host is faster at every prompt length tried, with the same first token.
    #:
    #: `--ops all` puts it back, which is how that was measured. Re-measure
    #: before assuming it still holds: the answer turns on dispatch overhead,
    #: which is exactly what the XRT launcher work would change.
    DEFAULT_OPS = ("matmul", "geglu")

    #: Everything `load_weights` establishes. `_lut` is a LIST here, one table
    #: per layer, where the base class has a single tensor -- see
    #: `load_weights`. The PLE trio and the two extra globals join it so a
    #: second instance built for `--compare-cpu` shares them rather than
    #: reloading a multi-GB bundle.
    WEIGHT_ATTRS = LlamaPrefill.WEIGHT_ATTRS + ("ple_proj_norm", "_ple_rows")

    #: The per-layer fused MLP chains, or None for the torch path. Declared on
    #: the class, not just assigned in `_build_fused_mlp`: an instance built
    #: for `share_weights_from` never runs `load_weights`, and `_layer` reads
    #: this on every layer.
    _fused_mlp = None

    def __init__(self, *a, **kw):
        """Llama's, with a per-layer KV cache.

        The base class allocates one `[max_seq, DK]` pair per layer at a fixed
        width. Here the width is the layer's own head dim, and the 20 layers
        from `FIRST_KV_SHARED` up get **no cache at all** -- they read the one
        belonging to `kv_source_layer(L)`. Allocating for them would be 20
        arrays that are written by nobody and read by nobody, and would make a
        wrong sharing map look like a cache of zeros rather than an error.
        """
        super().__init__(*a, **kw)
        # Shared pages where they can be had, so the GPU decode reads the
        # prefill's KV where it lies instead of copying it across. `kv_k` and
        # `kv_v` stay numpy either way -- they are *views* over the shared
        # pages here -- so `kv_view`, `kv_stack`, `_layer` and `--compare-cpu`
        # are unchanged and do not need to know. `_kv_sb_*` holds the buffers
        # so `Gemma4GpuDecode` can ask for the torch view of the same pages.
        self._kv_sb_k, self._kv_sb_v = self._alloc_kv()
        self.kv_k = [_host_view(b) for b in self._kv_sb_k]
        self.kv_v = [_host_view(b) for b in self._kv_sb_v]

    def _alloc_kv(self):
        """One `[max_seq, dh]` pair per layer that owns a cache, shared if possible.

        Falls back to plain numpy when the interop is unavailable -- no pyxrt,
        no ROCm torch, no visible device. This is an optimisation, and a cache
        the decode has to copy is better than a model that will not load.
        """
        shapes = [
            (self.max_seq, head_dim(L)) if L < FIRST_KV_SHARED else None
            for L in range(self.n_layers)
        ]
        try:
            from triton.backends.amd_triton_npu import shared

            bufs = [
                [
                    (
                        shared.zeros(
                            *sh, dtype=torch.float32, device="xrt:0", share="hip:0"
                        )
                        if sh
                        else None
                    )
                    for sh in shapes
                ]
                for _ in range(2)
            ]
            return bufs[0], bufs[1]
        except Exception as e:  # noqa: BLE001 -- see the docstring
            if os.environ.get("AMD_TRITON_NPU_DEBUG"):
                print(f"[gemma4] KV stays host-only: {e}", flush=True)
            return (
                [np.zeros(sh, np.float32) if sh else None for sh in shapes],
                [np.zeros(sh, np.float32) if sh else None for sh in shapes],
            )

    # ---- weights ----
    def load_weights(self, model=None):
        """This model's own loader: the base class's assumes fixed widths.

        Three departures, each forced by the shapes:

        * the Q|K|V fusion the base class does once is done **per layer** and
          only where the layer owns its k/v -- 20 of the 35 have no k or v to
          fuse. The gate|up fusion is per layer too, at that layer's width.
        * the RoPE table is a **list**, one per layer, because the base and the
          width both depend on the layer's attention type.
        * the PLE trio, the fifth norm and the scalar `out_scale` ride along in
          the layer dict.

        `raw` is emptied as it goes, for the reason the base class's does: it
        holds the float32 originals and keeping them alive alongside the bf16
        copies doubles the resident set.
        """
        raw = load_q4nx(model or self.model)
        self.fingerprint = raw["fingerprint"]
        self.embed = _as_bf16(raw.pop("embed"))
        # NOT tied on this model -- its own 262144x1536 matrix, and by a wide
        # margin the largest tensor here.
        self.lm_head = _t(_as_bf16(raw.pop("lm_head")), torch.bfloat16)
        self.final_norm = _t(raw["final_norm"])
        self.ple_proj_norm = _t(raw["ple_proj_norm"])
        self._ple_rows = raw["ple_rows"]
        rope_freqs = raw["rope_freqs"]

        self._w = []
        layers = raw["layers"]
        layers.reverse()  # pop off the tail; the reverse restores layer order
        while layers:
            k = len(self._w)
            L = layers.pop()
            w = {
                nm: _t(L[nm])
                for nm in (
                    "attn_norm",
                    "post_attn_norm",
                    "ffn_norm",
                    "post_ffn_norm",
                    "post_ple_norm",
                    "q_norm",
                    "k_norm",
                )
            }
            w["out_scale"] = float(L["out_scale"])
            for nm in ("o", "down", "inp_gate", "per_layer_projection", "model_proj"):
                w[nm] = _t(L[nm], torch.bfloat16)
            # One dispatch costs ~24 ms of fixed overhead regardless of size,
            # so projections sharing an input are issued as one GEMM. Q|K|V
            # only where k and v exist: on a KV-shared layer `q` goes alone.
            q = _t(L["q"], torch.bfloat16)
            if "k" in L:
                w["qkv"] = torch.cat(
                    [q, _t(L["k"], torch.bfloat16), _t(L["v"], torch.bfloat16)], dim=1
                ).contiguous()
            else:
                w["qkv"] = q
            w["gate_up"] = torch.cat(
                [_t(L["gate"], torch.bfloat16), _t(L["up"], torch.bfloat16)], dim=1
            ).contiguous()
            self._w.append(w)
            del L, q

        # bf16 cos/sin, as the device applies them. One table per layer: the
        # theta and the width both follow the attention type, so there is no
        # single table for this model and no default to fall back on.
        self._lut = [
            _t(rope_lut(self.max_seq, L, rope_freqs=rope_freqs))
            .to(torch.bfloat16)
            .to(torch.float32)
            for L in range(self.n_layers)
        ]
        self._build_fused_mlp()

    def _build_fused_mlp(self):
        """Pad every layer's MLP weights into a per-width fused chain.

        One `FusedMLP` per distinct FFN width -- two here, since the width
        changes at `FIRST_KV_SHARED` -- each owning one chain shared by its
        layers. Done at load time because that is when the padding is paid for
        anyway; the point of the chain is that it is then never paid again.

        Only on the NPU path, and only when both operators the chain subsumes
        would have gone there: an `--ops` bisection that pins either one to
        torch is asking to compare against the unfused path, and silently
        running the chain would answer a different question.

        Leaves `_fused_mlp` None on failure, which sends `_layer` down the
        unfused path. Deliberately narrow in what it catches: a
        CompilationError or a missing transform script means the chain is
        broken, and swallowing that here would report ~10x slower numbers under
        the same label rather than saying so.
        """
        self._fused_mlp = None
        # `_NPU_BACKENDS`, not `== "npu"`: `hetero` puts these two operators on
        # the NPU exactly as `npu` does, so it wants the chain for exactly the
        # same reason. Spelled as the shared tuple so the next backend that
        # routes to the NPU does not have to remember this line exists.
        fused_ops = {"matmul", "geglu"}
        if self.backend not in self._NPU_BACKENDS or not fused_ops <= self.enabled:
            return
        if os.environ.get("Q4NX_FUSED_MLP", "1") != "1":
            print("[gemma4] fused MLP disabled by Q4NX_FUSED_MLP=0")
            return
        try:
            from fused_mlp import FusedMLP

            by_width = {}
            for L in range(self.n_layers):
                inter = mlp_inter(L)
                mlp = by_width.get(inter)
                if mlp is None:
                    mlp = by_width[inter] = FusedMLP(D, inter)
                mlp.add_layer(L, self._w[L]["gate_up"], self._w[L]["down"])
                # Drop the originals. The chain holds its own padded copies and
                # nothing on this path reads these again -- keeping them would
                # hold ~80% of the model's weight bytes alive twice, which took
                # peak RSS past the `min_host_gib` this model declares.
                # `kernels.ResidentWeight` makes the same trade for the same
                # reason. The unfused path still needs them, so this runs only
                # once the chain is known to be built.
                del self._w[L]["gate_up"], self._w[L]["down"]
            self._fused_mlp = {L: by_width[mlp_inter(L)] for L in range(self.n_layers)}
            print(
                f"[gemma4] fused MLP: {len(by_width)} chains "
                f"(inter={sorted(by_width)}) over {self.n_layers} layers",
                flush=True,
            )
        except ImportError as e:
            print(f"[gemma4] fused MLP unavailable ({e}); using the unfused path")

    def share_weights_from(self, other):
        """Llama's, plus the MLP weights the fused path drops.

        `_w` is shared by reference, so an instance adopting a fused NPU
        model's weights sees the `gate_up`/`down` that `_build_fused_mlp`
        deleted -- and being a torch-path instance, it is exactly the one that
        needs them. `--compare-cpu` is the caller, and it raised before this.

        Rebuilt from the chain's padded copies rather than reloaded: a reload
        costs minutes and would compare against a different dequantization,
        which is the reason this method exists at all.

        The restored tensors land in the shared dicts, so the fused instance
        gets them back too. That is memory on a debug path and nothing more --
        its own `_layer` still goes through the chain, and `_fused_mlp` is what
        decides that, not the presence of these keys.
        """
        super().share_weights_from(other)
        for L, mlp in (getattr(other, "_fused_mlp", None) or {}).items():
            w = self._w[L]
            if "gate_up" not in w:
                w["gate_up"], w["down"] = mlp.logical_weights(L)
        return self

    # ---- forward ----
    def _geglu(self, gate, up, backend=None):
        """gelu_tanh(gate) * up, elementwise. Gemma3's, for the same reason."""
        with self.timer.track("geglu"):
            if self._on_npu("geglu", backend):
                import kernels

                return kernels.triton_geglu(gate, up)
            return torch.nn.functional.gelu(gate, approximate="tanh") * up

    def _head_norm(self, x, weight, n_heads, dh, backend=None):
        """RMSNorm over each head's `dh` lanes. x: [N, n_heads*dh] -> same.

        Qwen3's and Gemma3's trick, with the head dim passed in rather than
        read from the config: normalizing within a head makes this the ordinary
        `rms_norm` operator once the head axis is folded into the row axis.

        `weight=None` is the weightless value norm -- the reference normalizes
        v with no scale at all, which is a shape this operator has to allow.
        """
        N = x.shape[0]
        flat = x.reshape(N * n_heads, dh)
        if weight is None:
            rms = torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + RMS_EPS)
            out = flat * rms
        else:
            out = self._rms_norm(flat, weight, RMS_EPS, backend=backend)
        return out.reshape(N, n_heads * dh)

    def per_layer_inputs(self, x, N):
        """The PLE vector for every layer: [N, N_LAYERS, PLI_D].

        Computed **once, from the input embeddings** `x`, before the layer
        loop. Not from each layer's hidden state -- see the module docstring
        for how quietly that fails.

        Per layer: project the embeddings down to 256, scale, normalize against
        the one shared `ple_proj_norm`, add this layer's slice of the token's
        own per-layer embedding, and scale again.
        """
        with self.timer.track("ple_inputs"):
            tbl = _t(self._ple_rows(self._ids))  # [n, N_LAYERS, PLI_D]
            out = torch.zeros((N, N_LAYERS, PLI_D), dtype=torch.float32)
            n = tbl.shape[0]
            for L in range(self.n_layers):
                proj = self._matmul(x, self._w[L]["model_proj"]) * PLE_MODEL_PROJ_SCALE
                proj = self._rms_norm(proj, self.ple_proj_norm, RMS_EPS)
                out[:n, L] = (proj[:n] + tbl[:, L]) * PLE_INPUT_SCALE
                out[n:, L] = proj[n:] * PLE_INPUT_SCALE
            return out

    def _gpu_scope(self):
        """Hold the GPU driver across a region, or do nothing on the host path."""
        if self._gpu_device() is None:
            return contextlib.nullcontext()
        return gpu_kernels.gpu_driver()

    def _layer(self, x, L, N, keep=None, pli=None):
        """One Gemma4 block, on the prompt. x: [N, D] -> [N, D].

        `pli` is this layer's slice of `per_layer_inputs`, [N, PLI_D].

        Which head dim, which RoPE table, which window and whose KV are all
        decided by `L`: see the module docstring.
        """
        keep = N if keep is None else keep
        w = self._w[L]
        dh = head_dim(L)
        sliding = is_sliding(L)
        lut = self._lut[L]
        dq, dkv = N_Q_HEADS * dh, N_KV_HEADS * dh

        # ---- attention sublayer, norm-sandwiched ----
        residual = x
        h = self._rms_norm(x, w["attn_norm"], RMS_EPS)  # input_layernorm

        # One GPU-driver scope over the sublayer rather than one per operator.
        # `gpu_driver` no-ops when the GPU backend is already active, so the
        # scopes inside `_rope` and `_attention` cost nothing. Switching does
        # cost: `set_active` drops Triton's kernel cache.
        #
        # The qkv GEMM inside still switches to the NPU and back. That is
        # unavoidable; what this removes is the two ropes and the attention
        # each doing it again.
        # The projections first, outside the GPU scope below. They are NPU
        # work, and an NPU launch inside that scope would switch the active
        # driver and drop the GPU kernels' compiled cache -- the thing the
        # scope exists to keep.
        if L < FIRST_KV_SHARED:
            qkv = self._matmul(h, w["qkv"])
            q, k, v = qkv.split([dq, dkv, dkv], dim=1)
            src = None
        else:
            # A KV-shared layer projects q alone and attends the cache of
            # `kv_source_layer(L)`, which -- being lower -- has already run.
            q = self._matmul(h, w["qkv"])
            k = v = None
            src = kv_source_layer(L)

        # Everything from here to the attention is GPU work under `hetero`, so
        # it runs under one scope instead of each operator opening its own.
        with self._gpu_scope():
            q = self._head_norm(q, w["q_norm"], N_Q_HEADS, dh)
            q = self._rope(q, lut[:N], N_Q_HEADS)
            if src is None:
                k = self._head_norm(k, w["k_norm"], N_KV_HEADS, dh)
                # The value norm is weightless: `with_scale=False` upstream.
                v = self._head_norm(v, None, N_KV_HEADS, dh)
                k = self._rope(k, lut[:N], N_KV_HEADS)
                # The decode's handoff: roped K, normalized V.
                self.kv_k[L][:keep] = k[:keep].to(torch.float32).numpy()
                self.kv_v[L][:keep] = v[:keep].to(torch.float32).numpy()
            else:
                k = _t(self.kv_k[src][:N])
                v = _t(self.kv_v[src][:N])

            a = self._attention(
                q,
                k,
                v,
                N_Q_HEADS,
                N_KV_HEADS,
                dh,
                window=SLIDING_WINDOW if sliding else None,
                scale=ATTN_SCALE,
            )
        a = self._matmul(a, w["o"])  # o contracts dq -> D
        x = residual + self._rms_norm(a, w["post_attn_norm"], RMS_EPS)

        # ---- MLP sublayer, norm-sandwiched the same way ----
        residual = x
        h = self._rms_norm(x, w["ffn_norm"], RMS_EPS)  # pre_feedforward
        if self._fused_mlp is not None:
            # gate | up | merge | down as one dispatch, with the two weights
            # that are 80% of this model's staged bytes already on the device.
            with self.timer.track("mlp_fused"):
                d = self._fused_mlp[L].run(L, h)
        else:
            inter = mlp_inter(L)
            g, u = self._matmul(h, w["gate_up"]).split([inter, inter], dim=1)
            d = self._matmul(self._geglu(g, u), w["down"])
        x = residual + self._rms_norm(d, w["post_ffn_norm"], RMS_EPS)

        # ---- per-layer embedding injection ----
        # A GELU-tanh gate against this layer's PLE vector, projected back up
        # to D and added through the fifth norm. `_geglu` is the same operator:
        # gelu_tanh(gate) * other.
        residual = x
        gate = self._geglu(self._matmul(x, w["inp_gate"]), pli)
        p = self._matmul(gate, w["per_layer_projection"])
        x = residual + self._rms_norm(p, w["post_ple_norm"], RMS_EPS)

        # Per-layer output scale, a scalar. Missing it leaves every block
        # slightly too large, and it compounds over 35 layers.
        return x * w["out_scale"]

    def prefill(self, ids):
        """Run the prompt. Returns logits [VOCAB] for the final position.

        The base class's, with the PLE pass inserted before the layer loop and
        the logit softcap after the head.
        """
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

        self._ids = list(ids)
        pli_all = self.per_layer_inputs(x, Nb)
        for L in range(self.n_layers):
            x = self._layer(x, L, Nb, keep=N, pli=pli_all[:, L])
        self.current_context_length = N

        xn = self._rms_norm(x[N - 1 : N], self.final_norm, RMS_EPS)
        logits = self._lm_head(xn, self.lm_head)[0]
        if FINAL_LOGIT_SOFTCAP:
            # Monotonic, so it cannot change the argmax. Applied because the
            # reference applies it, which is what makes the two comparable as
            # logits rather than only as a predicted token.
            logits = FINAL_LOGIT_SOFTCAP * torch.tanh(logits / FINAL_LOGIT_SOFTCAP)
        # Kept so a decode does not have to be handed this separately. It and
        # the KV are written by the same call, and a decode seeded from one
        # prefill's token over another's KV answers for neither.
        self.last_first_token = int(logits.argmax())
        return logits

    # ---- decode handoff ----
    def kv_view(self, layer_idx):
        c = self.current_context_length
        src = kv_source_layer(layer_idx)
        return self.kv_k[src][:c], self.kv_v[src][:c]

    def clear_context(self):
        self.current_context_length = 0
        for L in range(self.n_layers):
            if self.kv_k[L] is not None:
                self.kv_k[L][:] = 0
                self.kv_v[L][:] = 0

    def kv_stack(self):
        """The handoff as two LISTS of [P, head_dim], one entry per layer.

        Lists, where every other model here returns a pair of stacked arrays:
        the entries are not the same width (256 on a sliding layer, 512 on a
        full one), so there is no array to stack them into. That is also the
        shape mlir-air's own `_prefill_npu` returns for this model and what its
        `FusedDecoder.seed_kv` consumes -- one `(P, head_dim)` entry per layer
        with the MQA head axis already squeezed off.

        A KV-shared layer repeats its source layer's arrays rather than holding
        a copy, exactly as upstream's does.
        """
        c = self.current_context_length
        ks, vs = [], []
        for L in range(self.n_layers):
            src = kv_source_layer(L)
            ks.append(np.asarray(self.kv_k[src][:c], np.float32))
            vs.append(np.asarray(self.kv_v[src][:c], np.float32))
        return ks, vs


# ---------------------------------------------------------------------------
# GPU decode
# ---------------------------------------------------------------------------
#
# The other half of a hybrid run: Triton prefill on the NPU, then token-by-token
# decode in torch on the iGPU, instead of mlir-air's fused NPU superkernel.
#
# Named after `examples/qwen2_5`'s `hetero-fast`, which is the same split for
# that model -- hetero prefill, all-GPU decode -- and reached the same way, by
# running the ordinary forward at one token with a KV cache. The math below is
# `_layer`'s, at N=1, with three differences that only appear at N=1:
#
#   * the new token attends the CACHE, so K and V come from it rather than from
#     this step's projections, and the sliding window becomes a bound on how far
#     back into the cache a single row may look;
#   * there is no causal mask -- one query row, every cached key at or before it
#     is visible;
#   * the per-layer embeddings come from THIS token's embedding. That is the
#     same rule the prompt follows (`per_layer_inputs` takes the embeddings of
#     the tokens being processed), not a decode-specific special case.
#
# Why a decode at all, when the NPU has one: it is the only way to run this
# model's generation without the PLE decode template, and it is a reference the
# fused kernel can be scored against -- one that lives here rather than
# upstream. It is not faster; the NPU path is one dispatch for all 35 layers.


class Gemma4GpuDecode:
    """Greedy decode for Gemma4-E2B, in Triton, on the iGPU.

    Built from a `Gemma4Prefill` that has already run: it takes that object's
    weights and its KV cache, so the prompt is not re-run. The weights are
    moved once, not per token -- on this model that is several GiB, and moving
    them per step would dominate everything else.

    The cache is held on the device as `[max_L, dh]` per layer that owns one,
    with the twenty KV-shared layers holding no cache at all and reading the
    one belonging to `kv_source_layer(L)` -- the same arrangement
    `Gemma4Prefill.__init__` uses, and for the same reason.

    Every operator is a Triton kernel from `gpu_kernels`
    ----------------------------------------------------
    The projections, both norms, RoPE, the GLU and the attention. `README.md`
    is explicit that *"PyTorch is used by the examples as a CPU reference"*,
    and `examples/gpt2` and `examples/qwen2_5` have had `*_kernel_gpu` for
    precisely this since they grew a GPU path -- a torch forward here would
    mean the one path in this repository that runs on a GPU exercises none of
    the compiler the repository is for.

    What is still torch is glue, and stays glue on purpose: moving weights and
    the cache to the device, the row gather out of the embedding table,
    `argmax` over the logits, and three scalar multiplies inside `_ple` on
    256-wide rows. None of them is an operator the model defines; a kernel for
    any of them would be slower than the launch it saved.
    """

    def __init__(self, pf, max_L, device="cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "no ROCm device visible to torch. This decode is the GPU half "
                "of a hybrid run; use the NPU decode instead, or install a "
                "ROCm build of torch."
            )
        self.pf = pf
        self.dev = device
        self.max_L = max_L
        self.timer = pf.timer

        # Weights, once. bf16 as the prefill holds them; the norms and the
        # RoPE tables stay float32 because that is what they are applied in.
        # The MLP weights may not be in `_w` at all: the fused NPU path hands
        # `gate_up`/`down` to its chain and deletes them, because holding ~80%
        # of the model's weight bytes twice takes peak RSS past this model's
        # declared `min_host_gib`. `logical_weights` rebuilds them from the
        # chain's padded copies -- the same door `share_weights_from` uses, and
        # for the same reason: a reload would cost minutes and would compare
        # against a different dequantization.
        fused = getattr(pf, "_fused_mlp", None) or {}
        self.w = []
        for L, w in enumerate(pf._w):
            w = dict(w)
            if "gate_up" not in w and L in fused:
                w["gate_up"], w["down"] = fused[L].logical_weights(L)
            self.w.append(
                {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in w.items()
                }
            )
        self.embed = _t(pf.embed, torch.bfloat16).to(device)
        self.lm_head = pf.lm_head.to(device)
        self.final_norm = pf.final_norm.to(device)
        self.ple_proj_norm = pf.ple_proj_norm.to(device)
        self.lut = [t.to(device) for t in pf._lut]

        # Where the prefill allocated the cache in shared pages, the torch
        # view of those pages is the decode's cache: the prefill wrote through
        # the numpy view of the same memory, so there is nothing to copy and
        # nothing to keep in step. The buffer is already `max_seq` deep, so a
        # step never reallocates.
        #
        # The copy this replaces is small. It is here because it is what lets a
        # decoder outlive a turn -- a new prompt's context is simply already
        # visible -- not for its own sake.
        self.P = pf.current_context_length
        self.k = []
        self.v = []
        shared_kv = getattr(pf, "_kv_sb_k", None) is not None
        for L in range(pf.n_layers):
            if pf.kv_k[L] is None:
                self.k.append(None)
                self.v.append(None)
                continue
            sbk, sbv = pf._kv_sb_k[L], pf._kv_sb_v[L]
            if shared_kv and hasattr(sbk, "torch"):
                self.k.append(sbk.torch())
                self.v.append(sbv.torch())
                continue
            # Host-only fallback: no interop, so the cache has to be copied.
            dh = head_dim(L)
            k = torch.zeros((max_L, dh), dtype=torch.float32, device=device)
            v = torch.zeros((max_L, dh), dtype=torch.float32, device=device)
            k[: self.P] = _t(pf.kv_k[L][: self.P]).to(device)
            v[: self.P] = _t(pf.kv_v[L][: self.P]).to(device)
            self.k.append(k)
            self.v.append(v)
        # The buffers outlive this object only if something holds them; `pf`
        # does, and the decode is built from it, so the pages cannot go away
        # underneath the views above.
        self._kv_owner = pf

    def _mm(self, x, w):
        """One activation row against a projection. bf16 in, f32 accumulation.

        `gpu_kernels.gemv`, not `x @ w`: this is the hot path -- ~281 of these
        per token across 35 layers and the head -- and at one row the cost is
        launch count, not arithmetic. See `gpu_kernels` for why it reduces over
        K rather than calling `tl.dot`.
        """
        return gpu_kernels.gemv(x, w)

    def _norm(self, x, weight=None):
        """RMSNorm over the last axis. `weight=None` is the weightless value norm."""
        return gpu_kernels.rmsnorm(x, weight, RMS_EPS)

    def _head_norm(self, x, weight, n_heads, dh):
        return self._norm(x.reshape(n_heads, dh), weight).reshape(1, n_heads * dh)

    def _norm_residual(self, x, weight, residual, scale=1.0):
        """The sublayer tail, `(residual + norm(x) * weight) * scale`, fused.

        Every Gemma4 sublayer ends this way, so this is three launches a layer
        rather than nine.
        """
        return gpu_kernels.rmsnorm_residual(x, weight, RMS_EPS, residual, scale)

    def _rope(self, x, L, pos, n_heads, dh):
        """Half-split rotary on one row, at one position.

        The same pairing `LlamaPrefill._rope` uses -- (i, i + dh/2), not
        adjacent lanes -- against this layer's own table. The partial rotary on
        the full layers is in that table, not here; see `config.rope_lut`.
        """
        return gpu_kernels.rope(x, self.lut[L][pos], n_heads, dh)

    def _ple(self, x):
        """This token's per-layer vectors, [N_LAYERS, PLI_D].

        From the token's own embedding, which is the same rule the prompt
        follows -- see the note above this class.
        """
        tbl = _t(self.pf._ple_rows([self._tok])).to(self.dev)  # [1, N_LAYERS, PLI_D]
        out = torch.empty((N_LAYERS, PLI_D), dtype=torch.float32, device=self.dev)
        for L in range(self.pf.n_layers):
            proj = self._mm(x, self.w[L]["model_proj"]) * PLE_MODEL_PROJ_SCALE
            proj = self._norm(proj, self.ple_proj_norm)
            out[L] = (proj[0] + tbl[0, L]) * PLE_INPUT_SCALE
        return out

    def step(self, token, pos):
        """One token through all 35 layers. Returns its logits, [VOCAB].

        The driver scope is here rather than around each launch: a step issues
        several hundred, and every one of them has to reach the GPU backend
        rather than the NPU one that a preceding prefill leaves active. See
        `gpu_kernels.gpu_driver`.
        """
        with gpu_kernels.gpu_driver():
            return self._step(token, pos)

    def _step(self, token, pos):
        """`step`'s body, with a GPU driver already selected.

        `pos` is the position this token occupies -- the length of everything
        before it. The cache rows `[0, pos)` are the context; this step writes
        row `pos` on the layers that own a cache and then attends `[0, pos]`.
        """
        self._tok = int(token)
        dev = self.dev
        # `self.embed` is already a torch tensor on the device -- staged once
        # in __init__ -- so this is a row gather, not another conversion.
        x = self.embed[self._tok : self._tok + 1].to(torch.float32)  # [1, D]
        pli = self._ple(x)

        for L in range(self.pf.n_layers):
            w = self.w[L]
            dh = head_dim(L)
            dq, dkv = N_Q_HEADS * dh, N_KV_HEADS * dh
            src = kv_source_layer(L)

            residual = x
            h = self._norm(x, w["attn_norm"])

            if L < FIRST_KV_SHARED:
                qkv = self._mm(h, w["qkv"])
                q, k, v = qkv.split([dq, dkv, dkv], dim=1)
                q = self._head_norm(q, w["q_norm"], N_Q_HEADS, dh)
                k = self._head_norm(k, w["k_norm"], N_KV_HEADS, dh)
                v = self._head_norm(v, None, N_KV_HEADS, dh)  # weightless
                q = self._rope(q, L, pos, N_Q_HEADS, dh)
                k = self._rope(k, L, pos, N_KV_HEADS, dh)
                self.k[L][pos] = k[0]
                self.v[L][pos] = v[0]
            else:
                q = self._mm(h, w["qkv"])
                q = self._head_norm(q, w["q_norm"], N_Q_HEADS, dh)
                q = self._rope(q, L, pos, N_Q_HEADS, dh)

            # One query row against the cache. No causal mask -- every cached
            # key is at or before this position by construction. The sliding
            # window survives as a lower bound on how far back it may look.
            lo = max(0, pos - SLIDING_WINDOW + 1) if is_sliding(L) else 0
            kc = self.k[src][lo : pos + 1]  # [S, dh]
            vc = self.v[src][lo : pos + 1]
            a = gpu_kernels.attn_decode(q, kc, vc, N_Q_HEADS, dh, ATTN_SCALE)

            a = self._mm(a, w["o"])
            x = self._norm_residual(a, w["post_attn_norm"], residual)

            residual = x
            h = self._norm(x, w["ffn_norm"])
            inter = mlp_inter(L)
            g, u = self._mm(h, w["gate_up"]).split([inter, inter], dim=1)
            d = self._mm(gpu_kernels.geglu(g, u), w["down"])
            x = self._norm_residual(d, w["post_ffn_norm"], residual)

            # Per-layer embedding injection, then the per-layer output scale --
            # folded into the same launch as the norm that precedes it.
            residual = x
            gate = gpu_kernels.geglu(self._mm(x, w["inp_gate"]), pli[L])
            pr = self._mm(gate, w["per_layer_projection"])
            x = self._norm_residual(
                pr, w["post_ple_norm"], residual, scale=w["out_scale"]
            )

        x = self._norm(x, self.final_norm)
        logits = self._mm(x, self.lm_head.T)[0]
        if FINAL_LOGIT_SOFTCAP:
            logits = gpu_kernels.logit_softcap(logits, FINAL_LOGIT_SOFTCAP)
        return logits

    def generate(self, first=None, n_tokens=1, eos=()):
        """Greedy continuation, `first` included. Returns the ids.

        `first` is the prefill's own prediction, so it is emitted without a
        step: the step that produced it was the prefill. Each subsequent token
        is one `step` at the next position.
        """
        # Both taken from the prefiller unless overridden, and taken *now*
        # rather than at construction: the cached decoder outlives a turn, and
        # a caller that ran another prefill in between would otherwise decode
        # one prompt's token against another's cache.
        if first is None:
            first = self.pf.last_first_token
        self.P = self.pf.current_context_length
        out = [int(first)]
        pos = self.P
        # One driver scope for the whole loop. `step` takes one too, which
        # no-ops inside this; what must not happen is a *switch* per token,
        # because that drops Triton's kernel cache and recompiles every shape
        # the token touches -- 7.5 s of an 9.3 s, 64-token run before this was
        # hoisted. See `gpu_kernels.gpu_driver`.
        with gpu_kernels.gpu_driver():
            for _ in range(max(0, n_tokens - 1)):
                if pos >= self.max_L:
                    print(f"[gpu-decode] hit max_L={self.max_L}; stopping", flush=True)
                    break
                with self.timer.track("gpu_decode_step"):
                    logits = self.step(out[-1], pos)
                nxt = int(torch.argmax(logits))
                pos += 1
                if nxt in eos:
                    out.append(nxt)
                    break
                out.append(nxt)
        return out
