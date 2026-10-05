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
import kv_layout

from llama_prefill import LlamaPrefill, _as_bf16, _t


def _host_view(buf):
    """The numpy view of a KV slot, whether it is a shared buffer or an array."""
    if buf is None:
        return None
    return buf.numpy() if hasattr(buf, "numpy") else buf


def _bf16_np(t):
    """A numpy bfloat16 view of a torch tensor, converting exactly once.

    numpy has no bfloat16, so the round trip goes through int16 and is
    reinterpreted by `ml_dtypes` on the far side. Done here rather than at each
    assignment because the value is written twice -- once per attention CU --
    and letting numpy convert on assignment would convert twice.
    """
    from ml_dtypes import bfloat16

    return t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(bfloat16)


class Gemma4Prefill(LlamaPrefill):
    """Q4NX Gemma4-E2B prefill producing the decode's KV handoff."""

    #: `geglu` in place of `swiglu`, as for Gemma3: the GLU activation is
    #: GELU-tanh, which is a different function rather than a different
    #: schedule. Named as its own operator so `--ops` can bisect it.
    NPU_OPS = ("matmul", "rms_norm", "geglu", "attention")

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
    DEFAULT_OPS = ("matmul", "geglu", "attention")

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
        """Llama's, with ONE KV cache in mlir-air's device layout.

        The base class allocates one `[max_seq, DK]` pair per layer at a fixed
        width. This model has two head widths and 20 layers that own no cache
        at all -- and, more to the point, three readers that used to disagree
        about the layout: the prefill and the iGPU decode held `[max_seq, dh]`
        f32 while the NPU decode wanted a scattered bf16 slab, with a per-turn
        rearrangement bridging them. They now share the slab. `kv_layout` is
        the single definition of it; nothing here computes an offset.

        Two things are kept alongside it, and both are load-bearing:

        * `_kv_src` -- dense f32 copies for the layers some *other* layer reads
          back (layer 13 and 14, derived rather than named). The 20 KV-shared
          layers attend their source's cache, and reading it out of the slab
          would mean the prefill's own attention ran on bf16 and on a padded
          head. Keeping these makes the prefill bit-identical to before this
          change: a layer that owns its cache uses the `k`/`v` in hand and
          never reads the slab at all.
        * `_kv_fanout` -- which slabs each source layer must also fill. The
          decode template on disk is identity-mapped (see `kv_layout`), so all
          35 slabs are read and a shared layer's slab has to carry a copy.
        """
        super().__init__(*a, **kw)
        self.kv_attn_maxl = self._resolve_attn_maxl()
        self._kv_fanout = {}
        for L in range(self.n_layers):
            self._kv_fanout.setdefault(kv_source_layer(L), []).append(L)
        self._kv_slab = self._alloc_kv()
        self._kv_host = _host_view(self._kv_slab)
        self._kv_src = {
            src: (
                np.zeros((self.max_seq, head_dim(src)), np.float32),
                np.zeros((self.max_seq, head_dim(src)), np.float32),
            )
            for src, users in self._kv_fanout.items()
            if users != [src]
        }

    def _alloc_base_kv(self, n_layers, max_seq):
        """No base cache: this model's lives in `_kv_slab`, in another shape.

        `kv_k`/`kv_v` are left unset rather than set to None, so anything that
        still reaches for them fails by name at the point of the mistake.
        """

    def _resolve_attn_maxl(self):
        """The slab's row count: the decode template's ATTN_MAXL, or `max_seq`.

        See `airsrc.decode_attn_maxl` for why both ends agree on a window from
        `max_seq` rather than each deriving one from the generation length.
        With no decode build present -- `--prefill-only`, `--decode gpu` --
        there is no template to agree with and `max_seq` is the right size.

        When a template IS present its size wins even if it is SMALLER than
        `max_seq`. Deferring to `max_seq` there looks like the safe choice and
        is the opposite: the slab would be laid out for rows the template does
        not have, so every offset past the first layer would be wrong. The
        decode cannot serve a context past its ATTN_MAXL in any case, so a
        prompt that needs more rows is a prompt this build cannot run --
        `prefill` says so rather than writing past the slab.
        """
        import airsrc

        got = airsrc.decode_attn_maxl("ple", want=self.max_seq)
        return self.max_seq if got is None else got

    def _alloc_kv(self):
        """The `[n_layers, layer_elems]` bf16 slab, in shared pages if possible.

        Falls back to plain numpy when the interop is unavailable -- no pyxrt,
        no ROCm torch, no visible device. The distinction that matters is that
        only *zero-copy* is lost there, not the layout: the slab is canonical
        either way, so the fallback path still skips the rearrangement and the
        decoder merely has to `write()` an already-correct buffer. That is
        deliberate -- shared buffers silently fall back in CI (#136), so a
        design that only existed on the shared path would never be gated.
        """
        shape = kv_layout.slab_shape(self.n_layers, self.kv_attn_maxl)
        try:
            from triton.backends.amd_triton_npu import shared

            return shared.zeros(
                *shape, dtype=torch.bfloat16, device="xrt:0", share="hip:0"
            )
        except Exception as e:  # noqa: BLE001 -- see the docstring
            if os.environ.get("AMD_TRITON_NPU_DEBUG"):
                print(f"[gemma4] KV stays host-only: {e}", flush=True)
            from ml_dtypes import bfloat16

            return np.zeros(shape, dtype=bfloat16)

    # ---- the canonical cache ----
    def _region(self, layer_idx, region):
        """One layer's K or V region as `[attn_maxl, REGION_W]`."""
        return kv_layout.region_view(
            self._kv_host, layer_idx, region, self.kv_attn_maxl
        )

    def _store_kv(self, layer_idx, k, v, keep):
        """Write this layer's roped K / normed V into every slab that reads it.

        `layer_idx` owns the cache; `_kv_fanout` names the KV-shared layers
        whose own slab has to carry the same rows, because the built template
        is identity-mapped.
        """
        dh = head_dim(layer_idx)
        pair = (
            (kv_layout.K_REGION, _bf16_np(k[:keep])),
            (kv_layout.V_REGION, _bf16_np(v[:keep])),
        )
        for L in self._kv_fanout[layer_idx]:
            for region, src in pair:
                kv_layout.scatter_rows(self._region(L, region)[:keep], src, dh, region)
        if layer_idx in self._kv_src:
            kf, vf = self._kv_src[layer_idx]
            kf[:keep] = k[:keep].to(torch.float32).numpy()
            vf[:keep] = v[:keep].to(torch.float32).numpy()

    def _source_kv(self, src, n):
        """The dense f32 `[n, dh]` K/V a KV-shared layer attends.

        Out of `_kv_src`, not out of the slab: see `__init__`. This is what
        keeps the prefill's arithmetic unchanged by the layout switch.
        """
        kf, vf = self._kv_src[src]
        return _t(kf[:n]), _t(vf[:n])

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
        """The PLE vector for every layer: [N, n_layers, PLI_D].

        Computed **once, from the input embeddings** `x`, before the layer
        loop. Not from each layer's hidden state -- see the module docstring
        for how quietly that fails.

        Per layer: project the embeddings down to 256, scale, normalize against
        the one shared `ple_proj_norm`, add this layer's slice of the token's
        own per-layer embedding, and scale again.

        All 35 projections run as ONE `[N, D] x [D, n_layers*PLI_D]` GEMM, the
        same reason `qkv` and `gate_up` are single GEMMs: they share an input.
        Here the input is shared across LAYERS rather than within one, which is
        why it was missed -- the loop read as per-layer work when only the
        weight is per-layer. It matters more here than anywhere else in the
        model, because `triton_matmul`'s chain stages the weight once
        (`static_indices={1}`) but re-stages the INPUT on every call: 35 copies
        of a [2048, 2048] bf16 `a` is 280 MiB, which at this link's ~4 GB/s is
        the whole of the 70 ms this used to cost against 7 ms of arithmetic.

        The norm batches for the same reason -- `ple_proj_norm` is one shared
        weight and RMSNorm is per-row, so 35 calls over [N, 256] are one call
        over [N*n_layers, 256] with the rows in the order the reshape gives.
        """
        with self.timer.track("ple_inputs"):
            tbl = _t(self._ple_rows(self._ids))  # [n, all layers, PLI_D]
            n = tbl.shape[0]
            nl = self.n_layers
            # Concatenated once and cached, not per prefill: it is the weight
            # the chain holds resident under `bo_key`, so a fresh tensor each
            # call would re-stage 37 MiB and give back what this saves.
            if getattr(self, "_mp_all", None) is None:
                self._mp_all = torch.cat(
                    [self._w[L]["model_proj"] for L in range(nl)], dim=1
                ).contiguous()
            proj = self._matmul(x, self._mp_all, stage_key="mp_all")
            proj = proj.reshape(N * nl, PLI_D) * PLE_MODEL_PROJ_SCALE
            proj = self._rms_norm(proj, self.ple_proj_norm, RMS_EPS)
            proj = proj.reshape(N, nl, PLI_D)
            # `tbl` covers only the real tokens; the padded tail has no table
            # row and takes the projection alone, exactly as the loop did.
            out = proj * PLE_INPUT_SCALE
            out[:n] = (proj[:n] + tbl[:, :nl]) * PLE_INPUT_SCALE
            return out

    def _attention(self, q, k, v, n_q, n_kv, dh, window=None, scale=None, backend=None):
        """The base class's causal attention, with its GEMMs on the NPU.

        `attn_npu` runs Q.K^T and P.V on the array and keeps the masked
        softmax on the host, over pages both sides address directly. Taken
        only where it applies -- one KV head, which is every Gemma4 layer, and
        no iGPU in play: under `hetero` the iGPU's flash kernel already does
        the whole operator, and does it faster.
        """
        if (
            n_kv != 1
            or not self._on_npu("attention", backend)
            or self._gpu_device(backend) is not None
        ):
            return super()._attention(
                q, k, v, n_q, n_kv, dh, window=window, scale=scale, backend=backend
            )
        import attn_npu

        with self.timer.track("attention"):
            cache = self.__dict__.setdefault("_npu_attn", {})
            att = cache.get((n_q, dh, window))
            if att is None:
                att = cache[(n_q, dh, window)] = attn_npu.NPUAttention(n_q, dh, window)
            return att(q, k, v, scale=dh**-0.5 if scale is None else scale)

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
        # `stage_key` keeps this layer's weight on the device between calls.
        # It is the weight's identity, not the shape's: the chain behind it is
        # shared by everything of the same shape, and this is what gives each
        # layer its own buffers on it. The two branches hold DIFFERENT weights
        # under one name -- a KV-shared layer's is q alone -- so they land on
        # two chains, which the shape in the key makes visible.
        if L < FIRST_KV_SHARED:
            qkv = self._matmul(h, w["qkv"], stage_key=f"qkv_L{L}")
            q, k, v = qkv.split([dq, dkv, dkv], dim=1)
            src = None
        else:
            # A KV-shared layer projects q alone and attends the cache of
            # `kv_source_layer(L)`, which -- being lower -- has already run.
            q = self._matmul(h, w["qkv"], stage_key=f"qkv_L{L}")
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
                # The decode's handoff: roped K, normalized V, written straight
                # into the layout both decoders read. No rearrangement follows.
                self._store_kv(L, k, v, keep)
            else:
                k, v = self._source_kv(src, N)

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
        a = self._matmul(a, w["o"], stage_key=f"o_L{L}")  # o contracts dq -> D
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
        gate = self._geglu(self._matmul(x, w["inp_gate"], stage_key=f"ig_L{L}"), pli)
        p = self._matmul(gate, w["per_layer_projection"], stage_key=f"plp_L{L}")
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
        if N > self.kv_attn_maxl:
            # The slab is the decode template's shape, so this is the same
            # limit `FusedDecoder.seed_kv` enforces -- raised here instead,
            # before `_store_kv` writes past the rows the layout has.
            raise ValueError(
                f"a {N}-token prompt needs {N} KV rows, but the decode "
                f"template on disk is ATTN_MAXL={self.kv_attn_maxl}. Rebuild "
                f"it larger: make compile-decode DECODE_L=<n>."
            )
        # Pad to a bucket. Safe under causal masking: the padded rows sit after
        # every real one, so no real position attends to them, and their own
        # outputs are discarded.
        Nb = min(
            self.max_seq,
            ((N + self.SEQ_BUCKET - 1) // self.SEQ_BUCKET) * self.SEQ_BUCKET,
        )
        x = torch.zeros((Nb, D), dtype=torch.float32)
        x[:N] = _t(self.embed[np.asarray(ids)])

        # Before any layer writes: the padded lanes inside a written row, and
        # the rows past the prompt, both have to be zero for the decode to read
        # them. `_store_kv` writes only the real lanes, by design.
        self._zero_slab()
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
        return (
            kv_layout.gather_rows(
                self._region(src, kv_layout.K_REGION)[:c],
                head_dim(src),
                kv_layout.K_REGION,
            ),
            kv_layout.gather_rows(
                self._region(src, kv_layout.V_REGION)[:c],
                head_dim(src),
                kv_layout.V_REGION,
            ),
        )

    def release_npu_decoder(self):
        """Close the NPU decoder kept across turns, if one was built."""
        dec = self.__dict__.pop("_npu_decoder", None)
        if dec is not None:
            dec.release()

    def clear_context(self):
        self.current_context_length = 0
        self._zero_slab()
        for kf, vf in self._kv_src.values():
            kf[:] = 0
            vf[:] = 0

    def _zero_slab(self):
        """Zero the whole slab before a prefill writes into it.

        Not narrowed to the rows this prompt uses, and the reason is the decode
        rather than the prefill: the attention bound is patched per token in
        16-row blocks, so at context L the kernel reads up to `roundup16(L)`
        and the rows in `[L, roundup16(L))` -- which no append has reached yet
        -- have to be zero. That bound grows with generation, which the prefill
        cannot see. Zeroing everything is what upstream's `seed_kv` does and is
        correct for the same reason.

        It costs ~43 ms at ATTN_MAXL=2048, inside a prefill that is ~18 s. The
        narrower version is a real optimisation (it measures 2.1 ms at a
        six-token prompt) but it needs the generation length, so it belongs
        with whatever learns that, not here.
        """
        self._kv_host[:] = 0

    # ---- the in-place handoff ----
    #
    # Both of these are how `harness._install_shared_kv` recognises a prefiller
    # whose cache the decode can read where it lies. A model without them takes
    # the `kv_stack()` route unchanged.
    def make_npu_decoder_class(self, air, prefiller=None):
        return make_npu_decoder_class(air, prefiller or self)

    def kv_placeholders(self, p):
        """What to hand `_prefill_npu` when nothing will read the K/V.

        Shaped, because their `generate()` derives `P` from `ks[0].shape[0]`;
        unreadable, because anything that gets past that is a bug -- see
        `UnusedKV`.
        """
        shapes = [head_dim(kv_source_layer(L)) for L in range(self.n_layers)]
        return (
            [UnusedKV(p, dh) for dh in shapes],
            [UnusedKV(p, dh) for dh in shapes],
        )

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

        Now a DE-SCATTER out of the canonical slab, which makes it O(P) work
        that the NPU decode no longer needs: `Gemma4NpuDecode` reads the slab
        in place. This stays for `--compare-cpu`, the self-check in
        `llama_prefill`, and any decoder that has not been taught the layout --
        callers that want the cache, not callers that want the handoff.
        """
        c = self.current_context_length
        ks, vs = [], []
        for L in range(self.n_layers):
            src = kv_source_layer(L)
            dh = head_dim(src)
            ks.append(
                kv_layout.gather_rows(
                    self._region(src, kv_layout.K_REGION)[:c], dh, kv_layout.K_REGION
                )
            )
            vs.append(
                kv_layout.gather_rows(
                    self._region(src, kv_layout.V_REGION)[:c], dh, kv_layout.V_REGION
                )
            )
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

    The cache is the prefill's slab, in the layout mlir-air's fused NPU decode
    also reads -- see `kv_layout`. So this reads the prompt's K/V where the
    prefill left it, and a step's own K/V is appended to the same rows the NPU
    decode would have appended to. A KV-shared layer still attends
    `kv_source_layer(L)`, and still carries a copy in its own slab, because the
    built decode template is identity-mapped.

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

    #: What the projections are read as. `q4nx` streams the bundle's Codec B
    #: chunks and dequantizes in the GEMV (W4A16); `bf16` streams the host's
    #: dequantized copies. The decode is weight-bandwidth bound, so this is
    #: the difference between 1.48 and 4.56 GB read per token. `q4k` is
    #: `q4nx` re-encoded at load to `gpu_kernels`' Codec K -- Q4_K-style 6-bit
    #: scales and mins, 4.5 bits a weight -- trading a small, measured loss
    #: of accuracy for 10% fewer projection bytes. The default: on the real
    #: model it is 6% faster than `q4nx` in the decode, passes the
    #: decode-vs-prefill gate token for token, and moves perplexity on a text
    #: passage within noise (85.5 -> 83.7; KL 2.3e-3 against `q4nx`).
    #: `Q4NX_GPU_WEIGHTS=q4nx` is the bundle-exact path.
    WEIGHTS = ("q4k", "q4nx", "bf16")

    def __init__(self, pf, max_L, device="cuda", weights=None):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "no ROCm device visible to torch. This decode is the GPU half "
                "of a hybrid run; use the NPU decode instead, or install a "
                "ROCm build of torch."
            )
        weights = weights or os.environ.get("Q4NX_GPU_WEIGHTS", "q4k")
        if weights not in self.WEIGHTS:
            raise ValueError(f"weights={weights!r}; expected one of {self.WEIGHTS}")
        self.weights = weights
        self.pf = pf
        self.dev = device
        # The slab is what bounds a run, not the caller's ask: appending past
        # its rows would write into the next layer's region rather than fail.
        self.max_L = min(max_L, pf.kv_attn_maxl)
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
        # The MLP weights are the chain's, read where they lie. Only where that
        # is impossible -- no interop, so they were never shared -- is a
        # private copy rebuilt, which is the path `logical_weights` exists for.
        #
        # Under `q4nx` / `q4k` none of that applies: the projections come packed from
        # the bundle instead, the bf16 copies of them never reach the device,
        # and the MLP runs unfused against its own packed gate|up and down.
        packed = None
        if weights in ("q4nx", "q4k"):
            from config import load_q4nx_packed

            packed = load_q4nx_packed(pf.model)
        fused = getattr(pf, "_fused_mlp", None) or {}
        self._mlp_w = None
        if fused and packed is None:
            try:
                self._mlp_w = {L: m.device_weights(L) for L, m in fused.items()}
            except RuntimeError:
                self._mlp_w = None
        self.w = []
        for L, w in enumerate(pf._w):
            w = dict(w)
            if packed is not None:
                for nm, (data, N, K) in packed["layers"][L].items():
                    w[nm] = self._packed(data, N, K)
            elif self._mlp_w is None and "gate_up" not in w and L in fused:
                w["gate_up"], w["down"] = fused[L].logical_weights(L)
            elif self._mlp_w is not None:
                w.pop("gate_up", None), w.pop("down", None)
            self.w.append(
                {
                    k: (
                        v.to(device)
                        if isinstance(v, (torch.Tensor, gpu_kernels.Q4NXWeight))
                        else v
                    )
                    for k, v in w.items()
                }
            )
        self.embed = _t(pf.embed, torch.bfloat16).to(device)
        # `[K, N]`, the orientation `_mm` takes. The bf16 head is reached as a
        # transposed view of `[VOCAB, D]` -- see `gpu_kernels.gemv` for why it
        # is not made contiguous.
        if packed is not None:
            data, N, K = packed["lm_head"]
            self.lm_head = self._packed(data, N, K).to(device)
        else:
            self.lm_head = pf.lm_head.to(device).T
        self.final_norm = pf.final_norm.to(device)
        self.ple_proj_norm = pf.ple_proj_norm.to(device)
        # Every layer's `model_proj` side by side, `[D, n_layers * PLI_D]`: they
        # all project the same embedding, so `_ple` issues them as one GEMV.
        self._model_proj_all = torch.cat(
            [w["model_proj"] for w in self.w], dim=1
        ).contiguous()
        for w in self.w:
            del w["model_proj"]
        self.lut = [t.to(device) for t in pf._lut]

        # The cache: region views of the prefill's slab, in the layout the NPU
        # decode also reads. Where the slab is in shared pages, `torch()` gives
        # the iGPU an alias of the very memory the prefill wrote through, so
        # there is nothing to copy and nothing to keep in step; where it is
        # not, one copy brings the same layout across. Either way this decode
        # and the NPU one now agree about where a row is, which is what the
        # layout unification is for.
        #
        # Views, not copies, are also what lets a decoder outlive a turn: a new
        # prompt's context is simply already visible.
        self.P = pf.current_context_length
        self.attn_maxl = pf.kv_attn_maxl
        slab = pf._kv_slab
        if hasattr(slab, "torch"):
            slab_t = slab.torch()
        else:
            # Host-only fallback: no interop, so the slab has to be copied. The
            # int16 hop is because numpy has no bfloat16 for torch to adopt.
            host = pf._kv_host
            slab_t = (
                torch.from_numpy(host.view(np.int16)).view(torch.bfloat16).to(device)
            )
        self._slab_t = slab_t
        self._slab_shared = hasattr(slab, "torch")
        self.k = [
            kv_layout.region_view(slab_t, L, kv_layout.K_REGION, self.attn_maxl)
            for L in range(pf.n_layers)
        ]
        self.v = [
            kv_layout.region_view(slab_t, L, kv_layout.V_REGION, self.attn_maxl)
            for L in range(pf.n_layers)
        ]
        # The buffers outlive this object only if something holds them; `pf`
        # does, and the decode is built from it, so the pages cannot go away
        # underneath the views above.
        self._kv_owner = pf
        # Per owning layer, the layers whose slabs take its K/V row, as the
        # device int32 list `gpu_kernels.qkv_post` walks. The same fan-out the
        # prefill does, for the same reason: the decode template on disk is
        # identity-mapped, so a KV-shared layer reads its own slab and that
        # slab has to carry the row too.
        self._fan = {
            L: torch.tensor(users, dtype=torch.int32, device=device)
            for L, users in pf._kv_fanout.items()
        }
        self._region_stride = kv_layout.region_stride(self.attn_maxl)

    def _packed(self, data, N, K):
        """A bundle matrix as the decode reads it, still on the host.

        Re-encoded to Codec K here, on the CPU, under `q4k`. Not on the iGPU:
        the encoder is a dozen torch ops, and on this stack torch's ROCm
        kernels are gfx1102 code run on gfx1103 through `HSA_OVERRIDE` --
        encoding on the device reset the GPU in 2 of 2 decode-vs-prefill runs,
        where the same runs with Codec B and a host-side encode are clean.
        """
        w = gpu_kernels.Q4NXWeight(torch.from_numpy(data), N, K)
        return w.to_codec_k() if self.weights == "q4k" else w

    def _mm(self, x, w):
        """One activation row against a projection. bf16 in, f32 accumulation.

        `gpu_kernels.gemv`, not `x @ w`: this is the hot path -- ~281 of these
        per token across 35 layers and the head -- and at one row the cost is
        launch count, not arithmetic. See `gpu_kernels` for why it reduces over
        K rather than calling `tl.dot`.

        A `Q4NXWeight` goes to the W4A16 GEMV instead; see `WEIGHTS`.
        """
        if isinstance(w, gpu_kernels.Q4NXWeight):
            return gpu_kernels.gemv_q4nx(x, w)
        return gpu_kernels.gemv(x, w)

    def _norm(self, x, weight=None):
        """RMSNorm over the last axis. `weight=None` is the weightless value norm."""
        return gpu_kernels.rmsnorm(x, weight, RMS_EPS)

    def _norm_residual(self, x, weight, residual, scale=1.0, next_norm=None):
        """The sublayer tail, `(residual + norm(x) * weight) * scale`, fused.

        Every Gemma4 sublayer ends this way, so this is three launches a layer
        rather than nine. `next_norm` also returns the following sublayer's
        pre-norm of the result, from the same launch.
        """
        return gpu_kernels.rmsnorm_residual(
            x, weight, RMS_EPS, residual, scale, next_norm
        )

    def _ffn(self, h, L, w):
        """The FFN sublayer body, from its pre-normed input `h`.

        Under `q4nx` / `q4k`, two launches: gate|up, then `down` with the GeGLU fused
        into its load of the activation. A persistent cooperative kernel doing
        both with a grid barrier was measured and rejected -- see
        IGPU_DECODE_PERF.md, "Persistent megakernel".
        """
        # `.get`: in `bf16` mode with the NPU chain resident, `__init__` pops
        # `gate_up`/`down` so `_mlp` reads the chain's shared pages instead.
        if isinstance(w.get("down"), gpu_kernels.Q4NXWeight):
            gu = self._mm(h, w["gate_up"])
            return gpu_kernels.gemv_q4nx(gu, w["down"], glu_in=True)
        return self._mlp(h, L, w)

    def _mlp(self, h, L, w):
        """`down(gelu_tanh(gate(h)) * up(h))` for one token.

        Reads the *chain's* weights where they are shared -- the same pages the
        NPU prefill dispatches on -- rather than a device-resident copy of its
        own. That is the whole reason they are allocated shared: the model
        exists once instead of three times.

        They come back padded, so the activation is zeroed out to `K_pad`
        before the projections. The padding contributes nothing: the weights
        are zero past the logical extent, so the tail of `gate`/`up` is zero,
        the merge maps zero to zero, and `down` contracts it against its own
        zero rows. The result is bit-identical to the unpadded form.
        """
        if self._mlp_w is None:
            inter = mlp_inter(L)
            g, u = self._mm(h, w["gate_up"]).split([inter, inter], dim=1)
            return self._mm(gpu_kernels.geglu(g, u), w["down"])
        Bg, Bu, Bd = self._mlp_w[L]
        K_pad = Bg.shape[0]
        hp = torch.zeros(1, K_pad, dtype=torch.float32, device=self.dev)
        hp[0, : h.shape[-1]] = h.reshape(-1)
        y = gpu_kernels.geglu(self._mm(hp, Bg), self._mm(hp, Bu))
        return self._mm(y, Bd)[:, : self.pf._w[L]["post_ffn_norm"].shape[-1]]

    def _ple(self, x):
        """This token's per-layer vectors, [n_layers, PLI_D].

        From the token's own embedding, which is the same rule the prompt
        follows -- see the note above this class.

        Sized by the prefill's layer count for the same reason
        `per_layer_inputs` is: `--n-layers` truncates.

        One GEMV and one kernel for all layers, rather than five launches per
        layer: every layer's `model_proj` reads the same `x`, so they are one
        `[D, n_layers * PLI_D]` projection, and the scale/norm/add/scale that
        follows is row-local. See `gpu_kernels.ple_combine`.
        """
        tbl = _t(self.pf._ple_rows([self._tok])).to(self.dev)  # [1, all, PLI_D]
        proj = self._mm(x, self._model_proj_all)
        return gpu_kernels.ple_combine(
            proj,
            self.ple_proj_norm,
            tbl[0, : self.pf.n_layers],
            PLE_MODEL_PROJ_SCALE,
            PLE_INPUT_SCALE,
            RMS_EPS,
        )

    def prefill(self, ids):
        """The prompt, on the iGPU, into the same KV slab. Returns its logits.

        The same forward as `Gemma4Prefill._layer`, run on this decoder's
        device-resident weights so the model is not held twice. The numerics
        differ from the host prefill's: the projections quantize activations
        to int8 (`gpu_kernels.gemm_w4a8`) and attention takes fp16 operands,
        so the logits are close to the host's, not equal.

        K/V go through the batched `qkv_post`, the kernel each decode step
        appends with, into `kv_layout`'s shared slab, so a decoder on either
        device continues from it as it would from the host prefill. Leaves
        `current_context_length` and `last_first_token` on the prefiller,
        which is what `generate` and the NPU decode read.
        """
        if not isinstance(self.w[0]["qkv"], gpu_kernels.Q4NXWeight):
            raise ValueError("the iGPU prefill reads packed weights: q4k or q4nx")
        pf = self.pf
        N = len(ids)
        if N > self.attn_maxl:
            raise ValueError(
                f"a {N}-token prompt needs {N} KV rows; ATTN_MAXL={self.attn_maxl}"
            )
        # The RoPE tables have `max_seq` rows, and the slab may have more.
        if N > pf.max_seq:
            raise ValueError(f"a {N}-token prompt exceeds max_seq={pf.max_seq}")
        with gpu_kernels.gpu_driver():
            logits = self._prefill(list(ids))
            if not self._slab_shared:
                # Without interop the device slab is a copy; the host one is
                # what `kv_stack` and the NPU decode read.
                pf._kv_host.view(np.int16)[...] = (
                    self._slab_t.view(torch.int16).cpu().numpy()
                )
        pf._ids = list(ids)
        pf.current_context_length = N
        pf.last_first_token = int(torch.argmax(logits))
        return logits

    def _prefill(self, ids):
        N, n = len(ids), self.pf.n_layers
        dev = self.dev
        # The slab's padded lanes and the rows past the prompt must read as
        # zero -- the reason `Gemma4Prefill._zero_slab` zeroes all of it.
        self._slab_t.zero_()
        tok = torch.tensor(ids, device=dev)
        x = self.embed[tok].to(torch.float32)  # [N, D]

        # Per-layer inputs for every token and layer at once: one GEMM for
        # every layer's `model_proj`, one `ple_combine` over all N * n rows.
        tbl = _t(self.pf._ple_rows(ids)).to(dev)[:, :n]  # [N, n, PLI_D]
        proj = gpu_kernels.matmul(x, self._model_proj_all)  # [N, n * PLI_D]
        pli = gpu_kernels.ple_combine(
            proj,
            self.ple_proj_norm,
            tbl.reshape(N * n, PLI_D),
            PLE_MODEL_PROJ_SCALE,
            PLE_INPUT_SCALE,
            RMS_EPS,
        )
        pli = pli.view(N, n, PLI_D).transpose(0, 1).contiguous()  # [n, N, PLI_D]

        dense = {}  # owning layer -> its dense fp16 K/V, for itself and sharers
        h = gpu_kernels.rmsnorm(x, self.w[0]["attn_norm"], RMS_EPS)
        for L in range(n):
            w = self.w[L]
            dh = head_dim(L)
            residual = x
            qkv = gpu_kernels.gemm_w4a8(h, w["qkv"])
            if L < FIRST_KV_SHARED:
                q, kd, vd = gpu_kernels.qkv_post(
                    qkv,
                    w["q_norm"],
                    w["k_norm"],
                    self.lut[L][:N],
                    N_Q_HEADS,
                    dh,
                    RMS_EPS,
                    kv=(self._slab_t, self._fan[L], 0, self._region_stride),
                    dense_kv=True,
                    out_dtype=torch.float16,
                )
                dense[L] = (kd, vd)
            else:
                q = gpu_kernels.qkv_post(
                    qkv,
                    w["q_norm"],
                    None,
                    self.lut[L][:N],
                    N_Q_HEADS,
                    dh,
                    RMS_EPS,
                    out_dtype=torch.float16,
                )
                kd, vd = dense[kv_source_layer(L)]
            a = gpu_kernels.attn_prefill_fa(
                q,
                kd,
                vd,
                N_Q_HEADS,
                dh,
                window=SLIDING_WINDOW if is_sliding(L) else None,
                scale=ATTN_SCALE,
            )
            a = gpu_kernels.gemm_w4a8(a, w["o"])
            x, h = gpu_kernels.rmsnorm_residual(
                a, w["post_attn_norm"], RMS_EPS, residual, next_norm=w["ffn_norm"]
            )

            residual = x
            # gate|up, GeGLU and the down GEMM's int8 input in one kernel.
            gq = gpu_kernels.gemm_w4a8_glu_q8(h, w["gate_up"])
            d = gpu_kernels.gemm_w4a8(gq, w["down"])
            x = gpu_kernels.rmsnorm_residual(d, w["post_ffn_norm"], RMS_EPS, residual)

            residual = x
            g = gpu_kernels.geglu(gpu_kernels.matmul(x, w["inp_gate"]), pli[L])
            pr = gpu_kernels.matmul(g.view(N, PLI_D), w["per_layer_projection"])
            nxt = self.w[L + 1]["attn_norm"] if L + 1 < n else self.final_norm
            x, h = gpu_kernels.rmsnorm_residual(
                pr,
                w["post_ple_norm"],
                RMS_EPS,
                residual,
                scale=w["out_scale"],
                next_norm=nxt,
            )

        # Only the last position's logits: the head is a GEMV, as in a step.
        logits = self._mm(h[N - 1 : N], self.lm_head)[0]
        if FINAL_LOGIT_SOFTCAP:
            logits = gpu_kernels.logit_softcap(logits, FINAL_LOGIT_SOFTCAP)
        return logits

    def step(self, token, pos):
        """One token through all 35 layers. Returns its logits, [VOCAB].

        The driver scope is here rather than around each launch: a step issues
        several hundred, and every one of them has to reach the GPU backend
        rather than the NPU one that a preceding prefill leaves active. See
        `gpu_kernels.gpu_driver`.
        """
        if not 0 <= pos < self.max_L:
            raise ValueError(
                f"position {pos} is outside the decode's {self.max_L} rows"
            )
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
        n = self.pf.n_layers
        # Every later pre-norm comes out of the launch before it; see
        # `_norm_residual`. Only the first layer's has nothing to ride on.
        h = self._norm(x, self.w[0]["attn_norm"])

        for L in range(n):
            w = self.w[L]
            dh = head_dim(L)
            src = kv_source_layer(L)

            # `h` is this layer's attn_norm of `x`, from the launch that ended
            # the previous layer. The head norms, RoPE and the cache append
            # are then one launch.
            residual = x
            qkv = self._mm(h, w["qkv"])
            owns = L < FIRST_KV_SHARED
            q = gpu_kernels.qkv_post(
                qkv,
                w["q_norm"],
                w["k_norm"] if owns else None,
                self.lut[L][pos],
                N_Q_HEADS,
                dh,
                RMS_EPS,
                kv=(
                    (self._slab_t, self._fan[L], pos, self._region_stride)
                    if owns
                    else None
                ),
            )

            # One query row against the cache. No causal mask -- every cached
            # key is at or before this position by construction. The sliding
            # window survives as a lower bound on how far back it may look.
            lo = max(0, pos - SLIDING_WINDOW + 1) if is_sliding(L) else 0
            kc = self.k[src][lo : pos + 1]  # [S, REGION_W], padded head
            vc = self.v[src][lo : pos + 1]
            a = gpu_kernels.attn_decode(
                q,
                kc,
                vc,
                N_Q_HEADS,
                dh,
                ATTN_SCALE,
                k_shift=kv_layout.lane_shift(dh, kv_layout.K_REGION),
                v_shift=kv_layout.lane_shift(dh, kv_layout.V_REGION),
            )

            a = self._mm(a, w["o"])
            x, h = self._norm_residual(
                a, w["post_attn_norm"], residual, next_norm=w["ffn_norm"]
            )

            residual = x
            d = self._ffn(h, L, w)
            x = self._norm_residual(d, w["post_ffn_norm"], residual)

            # Per-layer embedding injection, then the per-layer output scale --
            # folded into the same launch as the norm that precedes it.
            residual = x
            # `inp_gate` is 1536 -> 256: four programs unsplit, so it is split
            # four ways over K, and the partials are summed -- and the GeGLU
            # with this layer's per-layer input applied -- as the next GEMV
            # loads them. 53 -> 23 us, 35 times a token.
            gate = gpu_kernels.gemv(x, w["inp_gate"], split_k=4)
            pr = gpu_kernels.gemv(gate, w["per_layer_projection"], glu_up=pli[L])
            # ...and the next layer's attn_norm, or after the last layer the
            # final norm, from the same launch.
            nxt = self.w[L + 1]["attn_norm"] if L + 1 < n else self.final_norm
            x, h = self._norm_residual(
                pr, w["post_ple_norm"], residual, scale=w["out_scale"], next_norm=nxt
            )

        logits = self._mm(h, self.lm_head)[0]
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


# ---------------------------------------------------------------------------
# NPU decode
# ---------------------------------------------------------------------------
def make_npu_decoder_class(air, prefiller):
    """mlir-air's `FusedDecoder`, reading the prefill's cache where it lies.

    The same shape as `hsa_decode.make_hsa_decoder_class`: everything host-side
    stays theirs -- weight load, the dispatch loop, sampling -- and one thing is
    replaced. There it is the dispatch; here it is `seed_kv`.

    `seed_kv` exists to rearrange the prefill's `[P, head_dim]` K/V into the
    device slab. With `Gemma4Prefill` writing that slab directly there is
    nothing left to rearrange, so the override points the decoder's cache at
    the prefill's pages and syncs. Measured at a 2048-token prompt, that takes
    the step from 484 ms to a cache flush.

    Two things this must get right, both of which fail silently if it does not:

    * **The window.** A template serves every L in [1, ATTN_MAXL], and the slab
      is sized for one particular ATTN_MAXL. Upstream picks the smallest window
      covering `P + n_tokens`, which the prefill could not have known, so this
      pins the decoder to the largest calibrated one -- the same rule
      `airsrc.decode_attn_maxl` gave the prefill. Checked afterwards anyway: a
      slab built for another window is not an error, it is every row at the
      wrong offset.
    * **Who owns the pages.** In the shared case the decoder's KV argument
      becomes a userptr BO over the prefill's pages, so the in-place appends
      the kernel makes during generation are visible to whoever else holds
      them. Without the interop the slab is written once instead -- still no
      rearrangement, just not zero-copy.
    """

    # `_install_shared_kv` replaces `air.FusedDecoder` with this class, so a
    # later call would otherwise subclass the previous prefiller's adapter and
    # reach its cached decoder through `super().__new__`. Always derive from
    # mlir-air's own class.
    if not hasattr(air, "_triton_fused_decoder_base"):
        air._triton_fused_decoder_base = air.FusedDecoder
    base = air._triton_fused_decoder_base

    class Gemma4NpuDecode(base):
        # One decoder per prefiller, kept across turns. mlir-air's `generate()`
        # constructs a `FusedDecoder` on every call and closes it at the end.
        # Construction loads the template, the weights and a hardware context,
        # and none of that depends on the turn: `max_L` is pinned to the slab
        # below and `seed_kv` is only a sync.
        def __new__(cls, *a, **kw):
            cached = getattr(prefiller, "_npu_decoder", None)
            # An instance of an earlier turn's class is not an instance of this
            # one, so Python does not run `__init__` on it again.
            return cached if cached is not None else super().__new__(cls)

        def close(self):
            if getattr(prefiller, "_npu_decoder", None) is not self:
                super().close()

        def release(self):
            """Close this decoder even though it is the one kept for reuse."""
            base.close(self)

        def __init__(self, *a, **kw):
            if getattr(self, "_built", False):
                return
            # The prefiller comes from the closure, not the signature: upstream
            # `generate()` constructs this itself, as `FusedDecoder(model=...,
            # max_L=...)`, and knows nothing to pass.
            self._pf = prefiller
            # The window the SLAB was laid out for, not the one this prompt
            # would have chosen. It is itself calibrated, so upstream's
            # "smallest covering" lands exactly on it. Passing the generation
            # reach instead is what would pick a different window, and a
            # different window is a different layout.
            kw["max_L"] = prefiller.kv_attn_maxl
            try:
                super().__init__(*a, **kw)
            except RuntimeError as e:
                # The device grants a limited number of hardware contexts, and
                # the Triton prefill's chains may hold them all. Chains reopen
                # on their next run, so release them and open this one again.
                from triton.backends.amd_triton_npu.multilaunch import NPUChain

                if "HWCTX" not in str(e) or not NPUChain._open:
                    raise
                NPUChain.close_all()
                super().__init__(*a, **kw)
            if self.ATTN_MAXL != prefiller.kv_attn_maxl:
                raise RuntimeError(
                    f"the KV slab was built for ATTN_MAXL="
                    f"{prefiller.kv_attn_maxl} but the decode template "
                    f"resolved to {self.ATTN_MAXL}. Every row would be read at "
                    f"the wrong offset and the decode would produce fluent "
                    f"nonsense rather than fail. Rebuild the templates, or "
                    f"remove the stray pair from "
                    f"{os.environ.get('Q4NX_GEMMA4_DECODE_DIR', 'the decode dir')}."
                )
            if self.KV.shape != prefiller._kv_host.shape:
                raise RuntimeError(
                    f"KV slab shape {prefiller._kv_host.shape} != the decoder's "
                    f"{self.KV.shape}"
                )
            # The layers whose V the decoder keeps at the front of its slot.
            # The slab is written with `kv_layout`'s choice; a decoder that
            # disagrees reads every such layer's V from the wrong lanes.
            ours = {
                L
                for L in range(self.UNI)
                if kv_layout.lane_shift(head_dim(L), kv_layout.K_REGION)
                != kv_layout.lane_shift(head_dim(L), kv_layout.V_REGION)
            }
            theirs = getattr(self, "SWA", None)
            if theirs != ours:
                raise RuntimeError(
                    f"the decoder places V at the front of the slot on layers "
                    f"{sorted(theirs) if theirs is not None else 'none'}, but "
                    f"kv_layout does on {sorted(ours)}. The mlir-air in use "
                    f"does not match the KV layout this prefill writes."
                )
            self._bind_slab()
            self._built = True
            prefiller._npu_decoder = self

        def _bind_slab(self):
            """Point `kvc`/`KV` at the prefill's slab, zero-copy where possible."""
            bo = getattr(self._pf._kv_slab, "bo", None)
            if bo is None:
                self._shared = False
                return
            self.kvc = bo
            self.KV = self._pf._kv_host
            self._shared = True

        def seed_kv(self, ks, vs, P):
            """No rearrangement: the prefill already wrote this layout.

            `ks`/`vs` are ignored -- deliberately, and the caller knows: the
            harness hands this path a placeholder that raises if anything tries
            to read it, rather than real arrays that would quietly go unused.
            """
            if P > self.ATTN_MAXL:
                raise ValueError(f"prompt of {P} exceeds ATTN_MAXL={self.ATTN_MAXL}")
            TO = self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            if not self._shared:
                # Same layout, so this is one contiguous write, not a scatter.
                self.KV[:] = self._pf._kv_host
                self.kvc.write(
                    self.np.ascontiguousarray(self.KV).reshape(-1).view(self.np.int16),
                    0,
                )
            self.kvc.sync(TO)

    return Gemma4NpuDecode


class UnusedKV:
    """A stand-in for the `[P, head_dim]` arrays `seed_kv` no longer reads.

    mlir-air's `generate` takes the prefill's K/V and derives `P` from
    `ks[0].shape[0]` before handing them to `seed_kv`. Under the shared layout
    `seed_kv` ignores them, so building them would be an O(P) de-scatter whose
    result nothing reads.

    Passing zeros instead would be worse than wasteful. If the decoder override
    ever failed to apply, upstream's `seed_kv` would seed a cache of zeros and
    the run would produce a correct first token -- that comes from the prefill
    -- followed by fluent nonsense, which is the exact failure mlir-air records
    for a mis-seeded cache. So this carries the shape and nothing else, and
    raises the moment anyone reads a value.
    """

    __slots__ = ("shape",)

    def __init__(self, p, dh):
        self.shape = (p, dh)

    def __array__(self, *a, **kw):
        raise RuntimeError(
            "the prefill's K/V were read after the shared-KV decode path said "
            "nothing would read them -- the Gemma4NpuDecode.seed_kv override "
            "is not in effect. Refusing to seed: upstream's seed_kv would fill "
            "the cache from this placeholder and decode fluent nonsense."
        )
