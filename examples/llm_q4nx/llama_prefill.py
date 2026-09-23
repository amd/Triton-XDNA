# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The Llama prefill: the model, with each op routed to a backend.

Shared by every Llama-architecture Q4NX example -- 3.2-1B, 3.2-3B, 3.1-8B.
They differ only in the constants below, never in the forward: one norm pair
per block, fused QKV, half-split RoPE, SwiGLU, no qk-norm. A family that breaks
any of those (Gemma's norm sandwich, Qwen3's qk-norm) writes its own `_layer`
and must not add a branch to this one -- `qwen3_prefill.py` is that, and it
subclasses `LlamaPrefill` only to reuse what is *not* the forward: the operator
routing, the timer, the KV handoff and `prefill()`'s padding. The two hooks
that exist for it, `EXTRA_LAYER_WEIGHTS` and `kv_stack`, are marked below; a
family needing more than a hook needs its own class, not a third hook.

**Which model this is, is decided by `sys.path`.** The constants come from
`config`, and each example directory -- which holds its own `config.py` -- is
put ahead of this one on `sys.path` by its entry point. That keeps the forward
free of dimension plumbing, at the cost of an import that reads as ambiguous.
`bound_model()` below makes the binding checkable, and a second model in the
same process fails loudly rather than silently running on the first one's dims.

Structured like examples/gpt2 and examples/qwen2_5: the model carries a
`backend`, and each operator is a `_op(..., backend=None)` method that falls
back to it. The torch body of each method is the reference its NPU kernel is
checked against, so a wrong answer can be bisected op by op -- the only
practical way to debug a 28-layer model whose end-to-end signal is one token
id. `--ops` narrows which operators go to the NPU; `--compare-cpu` builds a
second model with backend="cpu" and diffs the KV cache layer by layer.

This forward is much shorter than gpt2's for two reasons, both structural:
there is no decode here (it is mlir-air's fused superkernel, reached through
the npz `save_kv_npz` describes), and there is no iGPU path, so no per-op
device placement.
"""

import os
import time
from collections import defaultdict
from contextlib import contextmanager

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


def bound_model():
    """The model whose constants this forward is running on.

    The names above are resolved once, at first import. A process that later
    put a different model's directory on `sys.path` would keep using these, and
    the symptom would be a plausible-looking wrong token rather than an error --
    so the binding is made checkable, and `LlamaPrefill` asserts it.
    """
    import config as _cfg

    return _cfg.MODEL_NAME


def _t(a, dtype=torch.float32):
    """numpy (possibly ml_dtypes bfloat16) -> torch tensor.

    bf16 is reinterpreted rather than converted. torch cannot borrow an
    ml_dtypes bfloat16 buffer directly, and the obvious workaround -- go via
    float32 -- doubles every weight in transit, which on an 8B is 13 GiB of
    transient float32 for a result that ends up bf16 again. The two types are
    the same two bytes, so view them as `uint16` and relabel: no copy, no
    intermediate, bit-identical to the float32 route.
    """
    if a.dtype.kind == "V" or str(a.dtype) == "bfloat16":
        t = torch.from_numpy(np.ascontiguousarray(a).view(np.uint16))
        t = t.view(torch.bfloat16)
        return t if dtype == torch.bfloat16 else t.to(dtype)
    return torch.from_numpy(np.ascontiguousarray(a)).to(dtype)


def _as_bf16(a):
    """A numpy array as ml_dtypes bfloat16, without a copy when it already is.

    mlir-air's readers hand back float32 -- their `bf16()` upcasts on the way
    out, and `dequant()` produces float32 by construction -- which is the right
    default for a reference forward and the wrong one for a table we only ever
    read a few rows of. Narrowing here means `_t` can then reinterpret rather
    than convert, so a tied LM head aliases the embedding instead of copying it.
    """
    from ml_dtypes import bfloat16

    if a.dtype == bfloat16:
        return a
    return a.astype(bfloat16)


class OpTimer:
    """Per-op wall-clock timer. Zero overhead when disabled.

    Same shape as examples/gpt2's, without its CUDA synchronise: there is no
    iGPU path here, so the wall time around a call already bounds the work.
    """

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.records = []  # (op_name, duration_ms)

    def reset(self):
        self.records.clear()

    @contextmanager
    def track(self, op_name):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        yield
        self.records.append((op_name, (time.perf_counter() - t0) * 1000))

    def summary(self):
        """Aggregate by op name, total time descending."""
        agg = defaultdict(float)
        for op, ms in self.records:
            agg[op] += ms
        return sorted(agg.items(), key=lambda kv: -kv[1])

    def report(self, total_s):
        total_ms = total_s * 1000
        for op, ms in self.summary():
            print(f"[profile] {op:10s} {ms:8.1f} ms  {100 * ms / total_ms:5.1f}%")
        print(f"[profile] {'TOTAL':10s} {total_ms:8.1f} ms")


class LlamaPrefill:
    """Q4NX Llama prefill producing the decode's KV handoff.

    backend: "cpu" runs every operator in torch, which is the reference the NPU
    path is checked against; "npu" runs the operators named by `ops` as Triton
    kernels and leaves the rest in torch.

    Which Llama is decided by which example's `config` is on `sys.path` -- see
    the module docstring and `bound_model`.
    """

    #: Operators with an NPU kernel. RoPE and attention are absent because
    #: neither has a transform script yet -- the same position
    #: examples/gpt2 --backend npu takes on attention, not a Triton limit.
    NPU_OPS = ("matmul", "rms_norm", "swiglu")

    #: Per-layer tensors a subclass's forward needs that Llama's does not, as
    #: `name -> torch dtype`. `load_weights` reads them out of `load_q4nx`'s
    #: layer dicts; this family needs none. Qwen3's per-head q/k norms are the
    #: reason it exists -- see qwen3_prefill.py.
    EXTRA_LAYER_WEIGHTS = {}

    #: Everything `load_weights` establishes, which is what a second instance
    #: needs to share rather than reload. Declared here so `share_weights_from`
    #: can copy it, because the alternative -- each caller listing the
    #: attributes it remembers -- is a trap: a subclass that adds one gets an
    #: AttributeError deep in the forward, from every call site that was
    #: written before it existed. Gemma3's second RoPE table did exactly that.
    WEIGHT_ATTRS = ("_w", "embed", "final_norm", "lm_head", "_lut", "fingerprint")

    def __init__(
        self,
        backend="cpu",
        ops="all",
        n_layers=N_LAYERS,
        max_seq=2048,
        model=None,
        expect_model=None,
    ):
        # A mismatch here means this forward resolved one model's dims while
        # the caller meant another, which produces fluent wrong text rather
        # than an error. Cheap to check once, expensive to debug later.
        if expect_model is not None and bound_model() != expect_model:
            raise RuntimeError(
                f"this forward is bound to {bound_model()!r} but {expect_model!r} "
                "was asked for; each model runs in its own process, with its own "
                "example directory first on sys.path"
            )
        self.backend = backend
        self.enabled = self._resolve_ops(ops)
        self.timer = OpTimer(enabled=False)
        self.n_layers = n_layers
        self.max_seq = max_seq
        self.model = model
        self.current_context_length = 0
        # Per-layer KV cache: roped K and raw V, [max_seq, 512], head-major.
        self.kv_k = [np.zeros((max_seq, DK), np.float32) for _ in range(n_layers)]
        self.kv_v = [np.zeros((max_seq, DV), np.float32) for _ in range(n_layers)]
        self._w = None

    @classmethod
    def _resolve_ops(cls, spec):
        """Which operators may go to the NPU: "all", or a comma-separated list.

        LLAMA_NPU_OPS is consulted when the caller passes nothing, so a
        bisection can be driven from the environment without touching argv.
        """
        if spec is None:
            spec = os.environ.get("LLAMA_NPU_OPS", "all")
        if spec in ("all", "*"):
            return set(cls.NPU_OPS)
        enabled = {o.strip() for o in spec.split(",") if o.strip()}
        unknown = enabled - set(cls.NPU_OPS)
        if unknown:
            raise ValueError(f"unknown ops {sorted(unknown)}; known: {cls.NPU_OPS}")
        return enabled

    #: Backends that put the NPU-capable operators on the NPU. `hetero` does
    #: too -- it differs only in where the REST go, which is `_gpu_device`.
    _NPU_BACKENDS = ("npu", "hetero")

    def _on_npu(self, op, backend):
        """True when `op` should run as a Triton kernel for this call."""
        return (backend or self.backend) in self._NPU_BACKENDS and op in self.enabled

    def _gpu_device(self, backend=None):
        """Where an operator that is NOT on the NPU should run.

        `None` means torch on the CPU, which is what `cpu` and `npu` both do:
        under `npu` the operators without an NPU kernel -- RoPE and attention --
        stay on the host. Under `hetero` they go to the iGPU instead, which is
        the whole of what that backend means here. The same split
        `examples/qwen2_5` calls hetero, named the same way on purpose.

        Returns None rather than raising when there is no ROCm device, so a
        host without one degrades to the CPU path instead of failing: this is a
        placement decision, not a correctness one, and CI has no iGPU.
        """
        if (backend or self.backend) != "hetero":
            return None
        if not hasattr(self, "_gpu_ok"):
            self._gpu_ok = torch.cuda.is_available()
            if not self._gpu_ok:
                print(
                    "[hetero] no ROCm device visible to torch; the operators "
                    "without an NPU kernel stay on the CPU",
                    flush=True,
                )
        return "cuda" if self._gpu_ok else None

    # ---- operators ----
    # Each is torch by default and Triton on the NPU when enabled. The torch
    # body is the reference --compare-cpu diffs against, so it models what the
    # device actually computes rather than the ideal.

    def _rms_norm(self, x, weight, eps, backend=None):
        """x: [N, D] -> normalized by RMS over D, scaled by `weight` [D]."""
        with self.timer.track("rms_norm"):
            if self._on_npu("rms_norm", backend):
                import kernels

                return kernels.triton_rms_norm(x, weight, eps)
            v = x.to(torch.float32)
            inv = torch.rsqrt((v * v).mean(-1, keepdim=True) + eps)
            return v * inv * weight

    def _matmul(self, x, w, backend=None, **mm):
        """x: [N, K] float32, w: [K, M] bfloat16 -> [N, M] float32.

        bf16 inputs with f32 accumulation, as the NPU GEMM does. Rounding x to
        bf16 here is not cosmetic: it is what the device sees, so keeping the
        reference in full f32 would hide a real error source.

        `mm` is forwarded to `kernels.triton_matmul` -- `block_n` and
        `transform_script`, both of which a wide-MLP model has to move off
        their defaults (see `qwen25_prefill._layer`). The torch path ignores
        them: they select a device schedule, and it has none.
        """
        with self.timer.track("matmul"):
            if self._on_npu("matmul", backend):
                import kernels

                return kernels.triton_matmul(x, w, **mm)
            if not hasattr(w, "to"):
                # A ResidentWeight: padded for the device, original freed. The
                # torch path cannot run against it, and saying so here beats an
                # AttributeError three frames down.
                raise RuntimeError(
                    "this weight was made NPU-resident, which frees the "
                    "unpadded copy the torch path needs. Build the model "
                    "without make_npu_resident() to run matmul on the CPU "
                    "(--compare-cpu and --ops without matmul both do)."
                )
            return (x.to(torch.bfloat16).to(torch.float32) @ w.to(torch.float32)).to(
                torch.float32
            )

    def _swiglu(self, gate, up, backend=None):
        """SiLU(gate) * up, elementwise."""
        with self.timer.track("swiglu"):
            if self._on_npu("swiglu", backend):
                import kernels

                return kernels.triton_swiglu(gate, up)
            return torch.nn.functional.silu(gate) * up

    def _rope(self, x, lut, n_heads, backend=None):
        """Half-split RoPE (HuggingFace Llama convention). CPU only, see NPU_OPS.

        x:   [N, n_heads*DH]
        lut: [N, DH] = [cos_0..cos_{DH/2-1}, sin_0..sin_{DH/2-1}]

        With H = DH // 2:

            out[i]      = x[i]*cos[i] - x[i+H]*sin[i]
            out[i+H]    = x[i]*sin[i] + x[i+H]*cos[i]

        Pairs (i, i+H), NOT adjacent (2i, 2i+1). H is 32 for the 1B's 64-wide
        heads and 64 for the 3B's 128-wide ones.
        """
        with self.timer.track("rope"):
            N = x.shape[0]
            # Under `hetero` this runs on the iGPU: it has no NPU kernel, so
            # the choice is host or GPU, not NPU or GPU. The result comes back
            # to the CPU because the next operator is an NPU GEMM, which reads
            # host memory -- the same round trip examples/qwen2_5 makes around
            # its GPU attention.
            dev = self._gpu_device(backend)
            if dev is not None:
                x, lut = x.to(dev), lut.to(dev)
            half = lut.shape[-1] // 2
            cos = lut[:, :half].unsqueeze(1)  # [N, 1, 32]
            sin = lut[:, half:].unsqueeze(1)
            v = x.reshape(N, n_heads, 2 * half)
            x1, x2 = v[..., :half], v[..., half:]
            out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).reshape(
                N, -1
            )
            return out.cpu() if dev is not None else out

    def _attention(self, q, k, v, n_q, n_kv, dh, window=None, scale=None, backend=None):
        """Causal GQA. q: [N, n_q*dh], k/v: [N, n_kv*dh] -> [N, n_q*dh].

        `window` bounds how far back a position may attend: with it set, j is
        visible to i only while `i - window < j <= i`. None is unbounded, which
        is every Llama and Qwen3 layer; Gemma3 sets it on five layers in six.
        A parameter rather than a subclass override because it is the same
        operator with a different mask, exactly as `n_q`/`n_kv` are the same
        operator at different head counts.

        `scale` multiplies the scores. `None` is `dh**-0.5`, the usual
        attention scale and what every model here but Gemma4 wants -- that one
        configures 1.0, which is not a normalization anyone would guess from
        the head dim, so it is passed rather than derived. Wrong here is quiet:
        the softmax still normalizes, it just runs at the wrong temperature.

        CPU only, see NPU_OPS.
        """
        with self.timer.track("attention"):
            N = q.shape[0]
            rep = n_q // n_kv
            scale = dh**-0.5 if scale is None else scale
            # The other op with no NPU kernel, and the one that pays for the
            # transfer: the scores are [n_q, N, N], so this is where a long
            # prompt spends its prefill. See `_gpu_device`.
            dev = self._gpu_device(backend)
            if dev is not None:
                q, k, v = q.to(dev), k.to(dev), v.to(dev)
            qh = q.reshape(N, n_q, dh).transpose(0, 1)  # [n_q, N, dh]
            kh = k.reshape(N, n_kv, dh).transpose(0, 1)  # [n_kv, N, dh]
            vh = v.reshape(N, n_kv, dh).transpose(0, 1)
            kh = kh.repeat_interleave(rep, dim=0)  # GQA broadcast
            vh = vh.repeat_interleave(rep, dim=0)
            scores = (qh @ kh.transpose(1, 2)) * scale  # [n_q, N, N]
            mask = torch.full((N, N), float("-inf")).triu(1)
            if window is not None:
                # `tril(-window)` is -inf exactly where j <= i - window, which
                # is what falls out of the window. Added to the causal mask
                # rather than replacing it: a position must satisfy both, and
                # j == i always survives, so no row is fully masked.
                mask = mask + torch.full((N, N), float("-inf")).tril(-window)
            scores = scores + mask.to(scores.device)
            p = torch.softmax(scores, dim=-1)
            out = (p @ vh).transpose(0, 1).reshape(N, n_q * dh)
            return out.cpu() if dev is not None else out

    def _lm_head(self, x, w, backend=None):
        """x: [N, D] -> logits [N, VOCAB]. w is [VOCAB, D] (tied embed).

        Kept in bf16: `w.to(torch.float32)` would materialize a fresh 1 GB copy
        of the tied embedding on every call, costing more than the matmul.
        CPU only -- one GEMV, off the hot path.
        """
        with self.timer.track("lm_head"):
            return torch.matmul(x.to(torch.bfloat16), w.t()).to(torch.float32)

    #: The projections `_matmul` sends to the NPU, and so the ones worth making
    #: resident. The norms and the per-head q/k norms are kilobytes and stay as
    #: they are.
    NPU_WEIGHTS = ("qkv", "o", "gate_up", "down")

    # ---- weights ----
    def make_npu_resident(self):
        """Pad every projection for the device now, and free the unpadded copy.

        `kernels.triton_matmul` pads each weight on first use and caches the
        result for the process lifetime, so after one prefill both forms are
        resident: 6.8 GiB unpadded plus 10.6 GiB padded on Qwen3-4B. On a run
        that only ever goes to the NPU the unpadded form is dead the moment the
        padded one exists -- but the cache cannot drop it, because it is keyed
        on that tensor's address and weakref'd to it.

        So the conversion is done here instead, one weight at a time, each
        original released as its padded form is built. That also moves the work
        off the first prefill, where it was the whole of an 11 GiB step.

        Not the default: it is one-way (see `_matmul`), and the torch reference
        path, `--ops` subsets that leave matmul on the CPU, and `--compare-cpu`
        all need the original. The entry points that only ever run on the
        device opt in.
        """
        import kernels

        for w in self._w:
            for name in self.NPU_WEIGHTS:
                if name in w:
                    w[name] = kernels.resident_weight(w[name])
        return self

    def share_weights_from(self, other):
        """Adopt `other`'s loaded weights instead of loading them again.

        For `--compare-cpu`, which runs the same weights through a second
        instance on the CPU backend: a reload would cost minutes and, worse,
        would compare against a different dequantization rather than the same
        one. `WEIGHT_ATTRS` is read off `other`'s class, so a subclass that adds
        an attribute is covered by declaring it there and nowhere else.
        """
        for attr in type(other).WEIGHT_ATTRS:
            setattr(self, attr, getattr(other, attr))
        return self

    def load_weights(self, model=None):
        raw = load_q4nx(model or self.model)
        self.fingerprint = raw["fingerprint"]
        # The embedding table and the LM head, in bf16 rather than the float32
        # mlir-air's readers return. Two savings, and the second is the larger:
        #
        #  * the table itself halves -- 1.45 -> 0.72 GiB on Qwen3-4B, 2.5 ->
        #    1.25 on Gemma3's 262208-row one. Nothing here wants the extra
        #    precision: `prefill` gathers a handful of rows and immediately
        #    rounds them through bf16 anyway, because that is what the device
        #    sees.
        #  * where the head is tied, it becomes an *alias* of the table instead
        #    of a second array. `_t` reinterprets a bf16 buffer rather than
        #    converting it, so `self.lm_head` and `self.embed` end up sharing
        #    one allocation; from float32 they could not, and the copy cost a
        #    second full-size tensor.
        #
        # `raw` is emptied as we go: it holds the float32 originals, and the
        # point is not to keep them alive alongside the bf16 ones.
        # `raw` is emptied as we go: it holds the float32 originals, and the
        # point is not to keep them alive alongside the bf16 ones.
        embed = raw.pop("embed")
        lm_head = raw.pop("lm_head")
        # Tied to `embed` on the 1B, 3B and Qwen3-4B, and its own dequantized
        # tensor on the 8B and Gemma3 -- `config.load_q4nx` decides which, so
        # this must not assume either. Getting it wrong is silent: the 8B ran
        # on the tied assumption and generated 57618 instead of " Paris".
        tied = lm_head is embed
        self.embed = _as_bf16(embed)
        del embed
        self.lm_head = _t(self.embed if tied else _as_bf16(lm_head), torch.bfloat16)
        del lm_head
        self.final_norm = _t(raw["final_norm"])
        self._w = []
        # Pop as we go: `raw` holds every layer's numpy arrays at once, and
        # holding those alongside the torch copies doubles the resident set for
        # the length of the loop. Each layer's source is dead once converted.
        #
        # Reversed so the pop comes off the tail. `pop(0)` shifts the whole list
        # each time, which is O(n^2) for no reason; `pop()` is O(1) and the
        # reverse restores the order the forward indexes `self._w` by. That
        # order is load-bearing -- layer k's weights must land at `_w[k]`.
        layers = raw["layers"]
        layers.reverse()
        while layers:
            L = layers.pop()
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
            for nm, dt in self.EXTRA_LAYER_WEIGHTS.items():
                w[nm] = _t(L[nm], dt)
            w["qkv"] = torch.cat([w["q"], w["k"], w["v"]], dim=1).contiguous()
            w["gate_up"] = torch.cat([w["gate"], w["up"]], dim=1).contiguous()
            # The unfused halves are dead once concatenated -- the forward only
            # ever reads `qkv` and `gate_up` -- and keeping them doubles the
            # host cost of everything that was just copied. That is 8.5 GiB on
            # Llama-3.1-8B, a third of its footprint, and enough on its own to
            # get the prefill OOM-killed on a small runner.
            for dead in ("q", "k", "v", "gate", "up"):
                del w[dead]
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
        h = self._rms_norm(x, w["attn_norm"], RMS_EPS)

        qkv = self._matmul(h, w["qkv"])  # [N, 2048+512+512]
        q, k, v = qkv.split([DQ, DK, DV], dim=1)

        q = self._rope(q, self._lut[:N], N_Q_HEADS)
        k = self._rope(k, self._lut[:N], N_KV_HEADS)

        # The decode's handoff: roped K, raw V, head-major within a position.
        self.kv_k[L][:keep] = k[:keep].to(torch.float32).numpy()
        self.kv_v[L][:keep] = v[:keep].to(torch.float32).numpy()

        a = self._attention(q, k, v, N_Q_HEADS, N_KV_HEADS, DH)  # [N, 2048]
        x = x + self._matmul(a, w["o"])

        h = self._rms_norm(x, w["ffn_norm"], RMS_EPS)
        g, u = self._matmul(h, w["gate_up"]).split([INTER, INTER], dim=1)
        return x + self._matmul(self._swiglu(g, u), w["down"])

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
        xn = self._rms_norm(x[N - 1 : N], self.final_norm, RMS_EPS)
        return self._lm_head(xn, self.lm_head)[0]

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

    def kv_stack(self):
        """The handoff as a pair of arrays: k, v float32 [N_LAYERS, P, DK].

        The layout is stated in `save_kv_npz` below, which is the same data
        written to a file. It is split out because mlir-air's drivers do not
        agree on how the handoff arrives -- the 1B reads the npz, the 3B is
        handed the prefiller, and Qwen3's takes exactly these two arrays (see
        `ModelSpec.driver_api`).
        """
        c = self.current_context_length
        return (
            np.stack([self.kv_k[L][:c] for L in range(self.n_layers)]).astype(
                np.float32
            ),
            np.stack([self.kv_v[L][:c] for L in range(self.n_layers)]).astype(
                np.float32
            ),
        )

    def save_kv_npz(self, path, first, prompt):
        """Write the handoff mlir-air's `generate()` loads.

        The contract, derived from both ends -- what its `kv_view()` returns
        and what its `seed_kv()` assumes -- because getting any of it wrong
        degrades quality without failing:

            k, v : [N_LAYERS, P, DK] float32   (bf16-exact values)

        DK is N_KV_HEADS x DH, laid out as column `h*DH + d`, position-major,
        heads contiguous within a position. That is [16, P, 512] for the 1B
        and [28, P, 1024] for the 3B -- the shape follows the model, so read it
        from `config` rather than from this line. No permutation and no interleaving
        at this boundary: the region-major scatter the decode wants happens
        inside its own `seed_kv()`.

        K is stored already rotated, V raw. The rotation is half-split --
        `out[i] = x[i]*cos[i] - x[i+DH/2]*sin[i]`, pairing i with i+DH/2 rather
        than adjacent lanes -- and its table carries llama3 frequency scaling
        (factor 32, low 1, high 4, old context 8192, theta 500000), which is
        why `config.rope_lut` re-exports mlir-air's generator instead of
        rebuilding one: a reimplementation that drops the scaling rotates K on
        the wrong frequencies and only long prompts show it.
        """
        K, V = self.kv_stack()
        np.savez(path, k=K, v=V, first=first, prompt=np.array(prompt))
        return K, V
