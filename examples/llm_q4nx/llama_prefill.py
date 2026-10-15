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

The NPU decode is mlir-air's fused superkernel, reached through the handoff
`save_kv_npz` describes or the shared cache `bind_kv_sink` takes. The iGPU
prefill and decode are `LlamaGpuDecode` at the end of this file.
"""

import os
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
from triton.backends.amd_triton_npu.driver_scope import driver_scope

from dense_kv_layout import K_REGION, V_REGION, DenseKVGeometry, gather

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


_HEAP_KEPT = False


def _keep_freed_blocks_in_heap():
    """Stop glibc from unmapping large freed blocks (once per process).

    Host ops allocate activation-sized tensors on every call, and with the
    default mmap threshold each one page-faults again. Q4NX_KEEP_HEAP=0 keeps
    the glibc defaults.
    """
    global _HEAP_KEPT
    if _HEAP_KEPT or os.environ.get("Q4NX_KEEP_HEAP", "1") == "0":
        return
    _HEAP_KEPT = True
    import ctypes

    try:
        mallopt = ctypes.CDLL("libc.so.6").mallopt
    except (OSError, AttributeError):
        return
    M_TRIM_THRESHOLD, M_MMAP_THRESHOLD = -1, -3
    ok = mallopt(M_MMAP_THRESHOLD, ctypes.c_int(1 << 30))
    ok &= mallopt(M_TRIM_THRESHOLD, ctypes.c_int(2**31 - 1))
    if not ok and os.environ.get("AMD_TRITON_NPU_DEBUG"):
        print("[prefill] mallopt refused the heap thresholds", flush=True)


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

    backend: a row of `PLACEMENT` -- "cpu" runs every operator in torch, which
    is the reference the device paths are checked against; "npu" runs the
    operators `ops` enables as Triton kernels on the NPU and leaves the rest in
    torch; "hetero" is "npu" with RoPE and attention on the iGPU.

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

    #: Whether `save_kv_npz` can write this model's cache as one npz, which
    #: `prefill.py --kv-out` offers. It needs every layer's cache the same width.
    SAVES_KV_NPZ = True

    def __init__(
        self,
        backend="cpu",
        ops=None,
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
        _keep_freed_blocks_in_heap()
        if backend not in self.PLACEMENT:
            raise ValueError(
                f"backend {backend!r}: {type(self).__name__} places its operators "
                f"for {sorted(self.PLACEMENT)}"
            )
        self.backend = backend
        self.enabled = self._resolve_ops(ops)
        self.timer = OpTimer(enabled=False)
        self.n_layers = n_layers
        self.max_seq = max_seq
        self.model = model
        self.current_context_length = 0
        self._alloc_base_kv(n_layers, max_seq)
        self._w = None
        self._kv_sink = None

    #: Per-layer KV cache: roped K and raw V, `[max_seq, 512]`, head-major.
    #:
    #: A method rather than two assignments because a subclass may hold its
    #: cache in another shape entirely -- Gemma4-E2B keeps one slab in the
    #: decode's own layout -- and would otherwise pay for these and drop them.
    #: At `max_seq=2048` that is ~294 MiB allocated and immediately orphaned.
    def _alloc_base_kv(self, n_layers, max_seq):
        self.kv_k = [np.zeros((max_seq, DK), np.float32) for _ in range(n_layers)]
        self.kv_v = [np.zeros((max_seq, DV), np.float32) for _ in range(n_layers)]

    def bind_kv_sink(self, sink):
        """Divert `_kv_store` into `sink` (a decoder's region-major KV buffer)."""
        self._kv_sink = sink

    def _kv_store(self, L, k, v, keep):
        """Write the layer's KV to the host cache, or the bound sink."""
        if self._kv_sink is not None:
            self._kv_store_direct(L, k, v, keep)
            return
        self.kv_k[L][:keep] = k[:keep].to(torch.float32).cpu().numpy()
        self.kv_v[L][:keep] = v[:keep].to(torch.float32).cpu().numpy()

    def _kv_store_direct(self, L, k, v, keep):
        """Write the layer's K/V into the sink, in `dense_kv_layout`'s layout."""
        from dense_kv_layout import K_REGION, V_REGION, DenseKVGeometry, scatter

        s = self._kv_sink
        geom = DenseKVGeometry.from_sink(s)
        # uint16 view: a bit copy, since torch and ml_dtypes bf16 would else convert.
        buf = s["buf"].view(np.uint16)
        slab = buf[: geom.n_layers * geom.lreg].reshape(geom.slab_shape())
        for region, x in ((K_REGION, k), (V_REGION, v)):
            rows = x[:keep].to(torch.bfloat16).view(torch.uint16).cpu().numpy()
            scatter(slab, L, region, rows, geom)

    #: What `--ops` selects when nothing is asked for. `None` means every
    #: operator with an NPU kernel, which is right wherever the NPU is faster
    #: at all of them. A subclass narrows it where measurement says otherwise;
    #: `NPU_OPS` stays the full set either way, so `--ops rms_norm` can still
    #: put one back for a bisection.
    DEFAULT_OPS = None

    #: Query rows per chunk in the host attention body. It bounds the score
    #: block at `[n_q, ATTN_CHUNK, window + ATTN_CHUNK]`, so smaller is a
    #: tighter bound on a windowed layer and more Python iterations; at a
    #: prompt of a few thousand the iterations are far below the GEMMs they
    #: carry. The result does not depend on it -- see `_attention`.
    ATTN_CHUNK = 128

    @classmethod
    def _resolve_ops(cls, spec):
        """Which operators may go to the NPU: "all", or a comma-separated list.

        LLAMA_NPU_OPS is consulted when the caller passes nothing, so a
        bisection can be driven from the environment without touching argv.
        "all" is literal -- every operator in `NPU_OPS` -- where passing
        nothing takes the class's `DEFAULT_OPS`.
        """
        if spec is None:
            spec = os.environ.get("LLAMA_NPU_OPS") or (
                ",".join(cls.DEFAULT_OPS) if cls.DEFAULT_OPS else "all"
            )
        if spec in ("all", "*"):
            return set(cls.NPU_OPS)
        enabled = {o.strip() for o in spec.split(",") if o.strip()}
        unknown = enabled - set(cls.NPU_OPS)
        if unknown:
            raise ValueError(f"unknown ops {sorted(unknown)}; known: {cls.NPU_OPS}")
        return enabled

    #: Where each operator runs, per `--backend`: "npu" is a Triton kernel on
    #: the NPU, "gpu" a Triton kernel on the iGPU, "cpu" torch on the host
    #: (the reference `--compare-cpu` checks against). Operators look their
    #: device up through `_device`; a subclass with other operators replaces
    #: the table.
    #:
    #: `_device` adjusts a cell in two cases:
    #:
    #: * an "npu" operator that `--ops` does not enable runs on the CPU;
    #: * a "gpu" cell, when torch sees no iGPU, takes the "npu" row's value,
    #:   so `hetero` without an iGPU runs as `npu`.
    #:
    #: RoPE and attention have no NPU kernel here, so they run on the host or
    #: the iGPU. The LM head is always torch on the host and is not listed.
    PLACEMENT = {
        "cpu": dict(
            matmul="cpu", rms_norm="cpu", swiglu="cpu", rope="cpu", attention="cpu"
        ),
        "npu": dict(
            matmul="npu", rms_norm="npu", swiglu="npu", rope="cpu", attention="cpu"
        ),
        "hetero": dict(
            matmul="npu", rms_norm="npu", swiglu="npu", rope="gpu", attention="gpu"
        ),
    }

    def _device(self, op, backend=None):
        """Where `op` runs for this call: "npu", "gpu" or "cpu". See PLACEMENT."""
        table = self.PLACEMENT
        dev = table[backend or self.backend][op]
        if dev == "gpu" and not self._has_gpu():
            dev = table["npu"][op]
        if dev == "npu" and op not in self.enabled:
            dev = "cpu"
        return dev

    def _has_gpu(self):
        """Whether torch sees an iGPU. Checked once; prints a note if not."""
        if not hasattr(self, "_gpu_ok"):
            self._gpu_ok = torch.cuda.is_available()
            if not self._gpu_ok:
                print(
                    "[hetero] no ROCm device visible to torch; the operators "
                    "placed on the iGPU run as under --backend npu",
                    flush=True,
                )
        return self._gpu_ok

    # ---- operators ----
    # Each is torch by default and Triton on the NPU when enabled. The torch
    # body is the reference --compare-cpu diffs against, so it models what the
    # device actually computes rather than the ideal.

    def _rms_norm(self, x, weight, eps, backend=None):
        """x: [N, D] -> normalized by RMS over D, scaled by `weight` [D]."""
        with self.timer.track("rms_norm"):
            if self._device("rms_norm", backend) == "npu":
                import kernels

                return kernels.triton_rms_norm(x, weight, eps)
            v = x.to(torch.float32)
            if hasattr(torch.nn.functional, "rms_norm"):
                return torch.nn.functional.rms_norm(v, (v.shape[-1],), weight, eps)
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
            if self._device("matmul", backend) == "npu":
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
            if self._device("swiglu", backend) == "npu":
                import kernels

                return kernels.triton_swiglu(gate, up)
            return torch.nn.functional.silu(gate) * up

    def _rope(self, x, lut, n_heads, backend=None):
        """Half-split RoPE (HuggingFace Llama convention). Host or iGPU.

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
            # Under `hetero` this runs on the iGPU. The result comes back to
            # the CPU because the next operator is an NPU GEMM, which reads
            # host memory -- the same round trip examples/qwen2_5 makes around
            # its GPU attention.
            if self._device("rope", backend) == "gpu":
                dev = "cuda"
                # A Triton kernel, not torch on a GPU tensor: torch is this
                # repository's CPU reference (README), and `examples/gpt2` and
                # `examples/qwen2_5` have run their GPU half as `*_kernel_gpu`
                # since they grew one.
                import gpu_kernels

                with driver_scope("gpu"):
                    out = gpu_kernels.rope_batch(
                        x.to(dev), lut.to(dev), n_heads, lut.shape[-1]
                    )
                return out.cpu()
            half = lut.shape[-1] // 2
            cos = lut[:, :half].unsqueeze(1)  # [N, 1, 32]
            sin = lut[:, half:].unsqueeze(1)
            v = x.reshape(N, n_heads, 2 * half)
            x1, x2 = v[..., :half], v[..., half:]
            return torch.cat(
                [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
            ).reshape(N, -1)

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

        Host or iGPU, see PLACEMENT.
        """
        with self.timer.track("attention"):
            N = q.shape[0]
            rep = n_q // n_kv
            scale = dh**-0.5 if scale is None else scale
            # The scores are [n_q, N, N], which is where a long prompt spends
            # its prefill.
            if self._device("attention", backend) == "gpu":
                dev = "cuda"
                # Flash-style, so the [n_q, N, N] score matrix the torch body
                # below materializes never exists -- that matrix is what makes
                # a long prompt expensive here. See `_rope` for why it is a
                # kernel rather than torch on a GPU tensor.
                import gpu_kernels

                with driver_scope("gpu"):
                    out = gpu_kernels.attn_prefill(
                        q.to(dev), k.to(dev), v.to(dev), n_q, n_kv, dh, window, scale
                    )
                return out.cpu()
            qh = q.reshape(N, n_q, dh).transpose(0, 1)  # [n_q, N, dh]
            kh = k.reshape(N, n_kv, dh).transpose(0, 1)  # [n_kv, N, dh]
            vh = v.reshape(N, n_kv, dh).transpose(0, 1)
            kh = kh.repeat_interleave(rep, dim=0)  # GQA broadcast
            vh = vh.repeat_interleave(rep, dim=0)
            out = torch.empty((n_q, N, dh), dtype=qh.dtype, device=qh.device)
            # A chunk of queries at a time, against only the keys its mask can
            # let through: causal stops at the chunk's LAST row, and a window
            # starts `window` back from its FIRST -- the two ends of the chunk,
            # not one of them, or the earliest rows lose keys they can see. Everything outside that span is
            # -inf for every row in the chunk, and -inf contributes exactly
            # zero to both the softmax denominator and `p @ v`, so leaving it
            # out is the same arithmetic on fewer elements rather than an
            # approximation of it. Whole-prompt scores are `[n_q, N, N]`; this
            # is what a long prompt used to spend its prefill on.
            for m0 in range(0, N, self.ATTN_CHUNK):
                m1 = min(m0 + self.ATTN_CHUNK, N)
                lo = 0 if window is None else max(0, m0 - window + 1)
                s = (qh[:, m0:m1] @ kh[:, lo:m1].transpose(1, 2)) * scale
                i = torch.arange(m0, m1, device=s.device)[:, None]
                j = torch.arange(lo, m1, device=s.device)[None, :]
                keep = j <= i
                if window is not None:
                    # `j <= i` is the causal half and `i - j < window` the
                    # other; a position must satisfy both. `j == i` survives
                    # both, so no row comes out fully masked.
                    keep = keep & (i - j < window)
                s = s.masked_fill(~keep, float("-inf"))
                out[:, m0:m1] = torch.softmax(s, dim=-1) @ vh[:, lo:m1]
            return out.transpose(0, 1).reshape(N, n_q * dh)

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

    #: Of those, the ones every call dispatches through a chain (`stage_key`).
    #: Only these may be padded to their exact K: the others are also reached
    #: by the unfused MLP path, which compiles a Triton kernel and so needs a
    #: power-of-two reduction. `kernels.resident_weight` fixes the padding at
    #: load time, so the choice cannot be deferred to the call.
    CHAIN_WEIGHTS = ("qkv", "o")

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
                    w[name] = kernels.resident_weight(
                        w[name], exact=name in self.CHAIN_WEIGHTS
                    )
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
        self._kv_store(L, k, v, keep)

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


# ---------------------------------------------------------------------------
# iGPU prefill and decode
# ---------------------------------------------------------------------------
def _q4_0_as_codec_b(data):
    """Signed Q4_0 chunks as Codec B, exactly.

    Same chunk layout, but the nibbles are two's-complement and the mins zero,
    so `w = s * n`. With `u = n ^ 8 = n + 8` that is `w = s * u - 8s`: Codec B
    with `min = -8 * scale`, which bf16 holds exactly.
    """
    import gpu_kernels as gk

    c = data.reshape(-1, gk.Q4NX_CHUNK_BYTES).copy()
    if c[:, 512:1024].any():
        raise RuntimeError("Q4_0 chunks with nonzero mins")
    sc = torch.from_numpy(c[:, :512].view(np.uint16).copy()).view(torch.bfloat16)
    c[:, 512:1024] = (sc * -8).view(torch.uint16).numpy().view(np.uint8)
    c[:, 1024:] ^= 0x88
    return c.reshape(-1)


def _quantize_pack(w):
    """A float `[N, K]` matrix as Codec B chunks, through mlir-air's own
    quantizer and block packer (what its decode is built from)."""
    from proj_qmm_pack import GROUP, pack_q4k_blocks
    from qwen25_7b_q4nx_requant import _requant_q4k

    q, sc, mn = _requant_q4k(np.asarray(w, np.float32), GROUP)
    return pack_q4k_blocks(q, sc, mn).view(np.uint8).reshape(-1)


def packed_bundle(qm, cfg, n_layers=None, inter=None, q4_0=False):
    """The projections in Codec B, still packed, and the host tensors the iGPU
    forward needs. `qm` is mlir-air's weight reader for the model.

    From a `model.q4nx` bundle the chunks are read as stored (`q4_0` converts
    the signed variant). From a reader with no bundle behind it, one that
    quantizes an HF checkpoint, they are quantized here.

    `inter` is the GLU width as stored, where the bundle pads it with zeros.

    Returns a dict with
        layers: per layer, `qkv`, `o`, `gate_up`, `down` as
            `(uint8 chunks, N, K)` for y = W x, q|k|v and gate|up stacked
            along N (the chunks are block-major over (N/32, K/256), so
            stacking is concatenation); the norms as `attn_norm`, `ffn_norm`
            and, for Gemma's sandwich, `post_attn_norm`, `post_ffn_norm`;
            `q_norm`, `k_norm` and `qkv_bias` where the model has them
        embed: bfloat16 `[VOCAB, D]` torch tensor
        final_norm: float32
        lm_head: `(uint8 chunks, VOCAB, D)`, or None without a packed head

    Only the I8 header (`w = s * q + min`) is read. The older "Q4NX" header
    subtracts its zero point and is refused rather than misread.
    """
    D, DQ, DK, V = cfg.D, cfg.DQ, cfg.DK, cfg.VOCAB
    I = inter or cfg.INTER
    bundle = hasattr(qm, "_hdr")

    def raw(name, N, K):
        if not bundle:
            return _quantize_pack(qm.bf16(name))
        e = qm._hdr[name]
        if e.get("dtype") != "I8":
            raise RuntimeError(f"{name}: {e.get('dtype')} header; expected I8")
        if e["shape"][0] != (N // 32) * (K // 256):
            raise RuntimeError(f"{name}: {e['shape'][0]} chunks is not {N}x{K}")
        data = qm._raw_i16(name).view(np.uint8)
        return _q4_0_as_codec_b(data) if q4_0 else data

    layers = []
    for k in range(n_layers or cfg.N_LAYERS):
        p = f"model.layers.{k}."
        a, m = p + "self_attn.", p + "mlp."
        qkv = [
            raw(a + "q_proj.weight", DQ, D),
            raw(a + "k_proj.weight", DK, D),
            raw(a + "v_proj.weight", DK, D),
        ]
        gate_up = [raw(m + "gate_proj.weight", I, D), raw(m + "up_proj.weight", I, D)]
        L = dict(
            qkv=(np.concatenate(qkv), DQ + 2 * DK, D),
            o=(raw(a + "o_proj.weight", D, DQ), D, DQ),
            gate_up=(np.concatenate(gate_up), 2 * I, D),
            down=(raw(m + "down_proj.weight", D, I), D, I),
        )
        norms = [np.asarray(n, np.float32) for n in qm.layer_rms(k)]
        if len(norms) == 2:
            L["attn_norm"], L["ffn_norm"] = norms
        else:
            L["attn_norm"], L["post_attn_norm"], L["ffn_norm"], L["post_ffn_norm"] = (
                norms
            )
        if hasattr(qm, "layer_qk_norm"):
            L["q_norm"], L["k_norm"] = (
                np.asarray(n, np.float32) for n in qm.layer_qk_norm(k)
            )
        if hasattr(qm, "layer_qkv_bias"):
            L["qkv_bias"] = np.concatenate(qm.layer_qkv_bias(k)).astype(np.float32)
        layers.append(L)

    if bundle:
        embed = torch.from_numpy(qm._raw_i16("model.embed_tokens.weight"))
        embed = embed.view(torch.bfloat16).view(V, D)
        has_head = qm._hdr.get("lm_head.weight", {}).get("dtype") == "I8"
    else:
        embed = torch.from_numpy(qm.bf16("model.embed_tokens.weight")).bfloat16()
        has_head = qm.has("lm_head.weight")
    return dict(
        layers=layers,
        embed=embed,
        final_norm=np.asarray(qm.bf16("model.norm.weight"), np.float32),
        lm_head=(raw("lm_head.weight", V, D), V, D) if has_head else None,
    )


class LlamaGpuDecode:
    """Prefill and greedy decode of a Llama-family model, in Triton, on the iGPU.

    `Gemma4GpuDecode` for the models mlir-air's shared decode engine builds.
    Built from a prefiller (`pf`): it reads the model and layer count from it,
    and leaves `current_context_length` and `last_first_token` on it after its
    own prefill, as the host prefill does.

    The weights come packed from the bundle (`config.load_q4nx_packed`), not
    from `pf`: the projections are read in place by the W4A16 GEMV (decode)
    and the W4A8 GEMM (prefill), and nothing is dequantized on the host.

    The KV cache is a `[n_layers, lreg]` slab in `dense_kv_layout`'s layout,
    which is the one mlir-air's NPU decode reads. It is allocated in pages the
    NPU and the iGPU share where the interop exists, or adopted from an NPU
    decoder (`bind_kv_sink`). Either way a prefill on one device and a decode
    on the other meet in the same rows:

    * iGPU prefill, iGPU decode: `prefill`, then `generate`;
    * host/NPU prefill, iGPU decode: the prefiller bound to `kv_sink()`;
    * iGPU prefill, NPU decode: this object bound to the decoder's sink.

    The forward covers the Llama family by what each layer carries, not by
    model name: `q_norm`/`k_norm` (per-head RMSNorm before RoPE),
    `qkv_bias`, `post_attn_norm`/`post_ffn_norm` (Gemma's sandwich), a RoPE
    table narrower than the head (partial rotary), and from `config`,
    `is_global_layer` with `SLIDING_WINDOW` (local layers attend a window and
    use the local table), `GLU_ACT` and `TIED_LM_HEAD`.
    """

    #: What the projections are read as. `q4k` re-encodes each matrix to
    #: `gpu_kernels`' Codec K on the host at load, as `Gemma4GpuDecode` does;
    #: `q4nx` streams the bundle's chunks. `Q4NX_GPU_WEIGHTS` overrides.
    WEIGHTS = ("q4k", "q4nx")
    #: The LM head: `packed` is the bundle's `lm_head`; `tied` is the bf16
    #: embedding, the default for a model that ties them (`TIED_LM_HEAD`),
    #: where the packed copy is a lossier rendering of the same matrix.
    #: `Q4NX_GPU_HEAD` overrides.
    HEADS = ("packed", "tied")

    def __init__(self, pf, max_L, device="cuda", weights=None, head=None):
        import config as cfg

        if not torch.cuda.is_available():
            raise RuntimeError(
                "no ROCm device visible to torch; use the NPU decode instead, "
                "or install a ROCm build of torch"
            )
        weights = weights or os.environ.get("Q4NX_GPU_WEIGHTS", "q4k")
        if weights not in self.WEIGHTS:
            raise ValueError(f"weights={weights!r}; expected one of {self.WEIGHTS}")
        tied = getattr(cfg, "TIED_LM_HEAD", False)
        head = head or os.environ.get("Q4NX_GPU_HEAD") or ("tied" if tied else "packed")
        if head not in self.HEADS:
            raise ValueError(f"head={head!r}; expected one of {self.HEADS}")
        if head == "tied" and not tied:
            raise ValueError(f"{cfg.MODEL_NAME} does not tie its LM head")
        self.cfg, self.pf, self.dev = cfg, pf, device
        self.weights, self.head = weights, head
        self.timer = pf.timer
        self.n_layers = pf.n_layers
        self.nq, self.nkv, self.dh = cfg.N_Q_HEADS, cfg.N_KV_HEADS, cfg.DH
        self.eps = cfg.RMS_EPS
        self.act = getattr(cfg, "GLU_ACT", "silu")
        self.max_L = max_L
        self.geom = DenseKVGeometry.for_model(
            cfg.MODEL_NAME, self.nkv, self.dh, self.n_layers, max_L
        )
        self._slab = None  # the shared buffer, where there is one
        self._slab_t = None  # its `[n_layers, lreg]` iGPU view
        self._load(pf.model)

    def _load(self, model):
        import gpu_kernels as gk

        packed = self.cfg.load_q4nx_packed(model, n_layers=self.n_layers)
        dev = self.dev

        def proj(entry):
            data, N, K = entry
            w = gk.Q4NXWeight(torch.from_numpy(data), N, K)
            if self.weights == "q4k":
                try:
                    w = w.to_codec_k()
                except ValueError:
                    pass  # a positive min, which Codec K cannot hold
            return w.to(dev)

        projs = ("qkv", "o", "gate_up", "down")
        self.w = [
            {k: proj(v) if k in projs else _t(v).to(dev) for k, v in L.items()}
            for L in packed.pop("layers")
        ]
        self.embed = packed["embed"]
        self.final_norm = _t(packed["final_norm"]).to(dev)
        if self.head == "tied":
            # `[D, VOCAB]` as a transposed view; see `gpu_kernels.gemv`.
            self.lm_head = self.embed.to(dev).T
        elif packed["lm_head"] is None:
            raise RuntimeError("the bundle has no packed lm_head")
        else:
            self.lm_head = proj(packed["lm_head"])
        self.lut = _t(self.cfg.rope_lut(self.max_L)).to(dev)
        self.lut_global = None
        if hasattr(self.cfg, "is_global_layer"):
            self.lut_global = _t(self.cfg.rope_lut(self.max_L, global_layer=True)).to(
                dev
            )

    # ---- the cache ----
    def bind_kv_sink(self, sink):
        """Use a decoder's resident KV buffer as the cache.

        `sink` is what `harness._xrt_kv_sink` builds, with `torch`, the
        buffer's iGPU view. Its geometry must be this model's, or every row
        would land at the wrong offset without an error.
        """
        geom = DenseKVGeometry.from_sink(sink, self.n_layers)
        if (geom.ngrp, geom.region_w) != (self.geom.ngrp, self.geom.region_w):
            raise ValueError(
                f"sink has {geom.ngrp} groups of {geom.region_w}, the model "
                f"{self.geom.ngrp} of {self.geom.region_w}"
            )
        t = sink.get("torch")
        if t is None or not t.is_cuda:
            raise ValueError("the sink has no iGPU view")
        self.geom = geom
        self._slab = sink.get("shared")
        n = geom.n_layers * geom.lreg
        self._slab_t = t.reshape(-1)[:n].view(*geom.slab_shape())

    def kv_sink(self):
        """The cache as a sink a host or NPU prefiller can bind
        (`LlamaPrefill.bind_kv_sink`), for this object to decode from."""
        self._ensure_kv()
        if self._slab is None:
            raise RuntimeError("the KV cache is not in shared pages here")
        return dict(self.geom.sink(self._slab.numpy()), torch=self._slab_t)

    def _ensure_kv(self):
        """Allocate the slab, in XRT/HIP-shared pages if the interop works.

        Without it the slab is plain iGPU memory, and only the iGPU-only split
        is available.
        """
        if self._slab_t is not None:
            return
        shape = self.geom.slab_shape()
        try:
            import kernels
            from triton.backends.amd_triton_npu import shared

            self._slab = shared.zeros(
                *shape, dtype=torch.bfloat16, device="xrt:0", share=kernels.igpu_share()
            )
            self._slab_t = self._slab.torch()
            if not self._slab_t.is_cuda:
                raise RuntimeError("no iGPU view of the shared buffer")
        except Exception as e:  # noqa: BLE001 -- see the docstring
            if os.environ.get("AMD_TRITON_NPU_DEBUG"):
                print(f"[gpu] KV not in shared pages: {e}", flush=True)
            self._slab = None
            self._slab_t = torch.zeros(shape, dtype=torch.bfloat16, device=self.dev)

    @property
    def kv_slab(self):
        """The shared buffer holding the cache, or None."""
        return self._slab

    def kv_stack(self, P):
        """Rows `[0, P)` as `(k, v)` float32 `[n_layers, P, DK]`. A copy."""
        return tuple(
            torch.stack(
                [gather(self._slab_t, L, r, self.geom, P) for L in range(self.n_layers)]
            )
            .float()
            .cpu()
            .numpy()
            for r in (K_REGION, V_REGION)
        )

    # ---- forward ----
    def _window(self, L):
        g = getattr(self.cfg, "is_global_layer", None)
        return None if g is None or g(L) else self.cfg.SLIDING_WINDOW

    def _lut(self, L):
        g = getattr(self.cfg, "is_global_layer", None)
        return self.lut_global if g is not None and g(L) else self.lut

    def _qkv_post(self, qkv, L, rows, pos, **kw):
        import gpu_kernels as gk

        w = self.w[L]
        return gk.qkv_post_gqa(
            qkv,
            rows,
            self.nq,
            self.nkv,
            self.dh,
            self.eps,
            q_norm=w.get("q_norm"),
            k_norm=w.get("k_norm"),
            bias=w.get("qkv_bias"),
            kv=(self._slab_t[L], pos, self.geom),
            **kw,
        )

    def _tail(self, x, y, L, post, nxt):
        """End a sublayer: `(x', rmsnorm(x') * nxt)`. Pre-norm adds `y` to
        `x`; Gemma's sandwich norms `y` with `post` first."""
        import gpu_kernels as gk

        w = self.w[L]
        if post in w:
            return gk.rmsnorm_residual(y, w[post], self.eps, x, next_norm=nxt)
        return gk.add_rmsnorm(x, y, nxt, self.eps)

    def _next_norm(self, L):
        return self.w[L + 1]["attn_norm"] if L + 1 < self.n_layers else self.final_norm

    def _head(self, h):
        import gpu_kernels as gk

        if isinstance(self.lm_head, gk.Q4NXWeight):
            return gk.gemv_q4nx(h, self.lm_head)[0]
        return gk.gemv(h, self.lm_head)[0]

    def prefill(self, ids):
        """The prompt into cache rows `[0, N)`. Returns its last logits.

        The projections quantize activations to int8 (`gpu_kernels.gemm_w4a8`)
        and attention takes fp16 operands, so the logits are close to the host
        prefill's, not equal.
        """
        ids = [int(t) for t in np.asarray(ids).reshape(-1)]
        N = len(ids)
        if N > min(self.max_L, self.geom.attn_maxl):
            raise ValueError(
                f"a {N}-token prompt exceeds the cache's "
                f"{min(self.max_L, self.geom.attn_maxl)} rows"
            )
        self._ensure_kv()
        with driver_scope("gpu"):
            logits = self._prefill(ids).cpu()
        self.pf._ids = ids
        self.pf.current_context_length = N
        self.pf.last_first_token = int(torch.argmax(logits))
        return logits

    def _prefill(self, ids):
        import gpu_kernels as gk

        N = len(ids)
        nq, nkv, dh, eps = self.nq, self.nkv, self.dh, self.eps
        x = self.embed[torch.tensor(ids)].float().to(self.dev)
        h = gk.rmsnorm(x, self.w[0]["attn_norm"], eps)
        for L in range(self.n_layers):
            w = self.w[L]
            q, kd, vd = self._qkv_post(
                gk.gemm_w4a8(h, w["qkv"]),
                L,
                self._lut(L)[:N],
                0,
                dense_kv=True,
                out_dtype=torch.float16,
            )
            a = gk.attn_prefill_fa(q, kd, vd, nq, dh, window=self._window(L), n_kv=nkv)
            o = gk.gemm_w4a8(a, w["o"])
            x, h = self._tail(x, o, L, "post_attn_norm", w["ffn_norm"])
            gq = gk.gemm_w4a8_glu_q8(h, w["gate_up"], act=self.act)
            d = gk.gemm_w4a8(gq, w["down"])
            x, h = self._tail(x, d, L, "post_ffn_norm", self._next_norm(L))
        return self._head(h[N - 1 : N])

    def step(self, token, pos):
        """One token at position `pos`: appends its K/V at row `pos` and
        attends `[0, pos]`. Returns its logits, on the iGPU."""
        if not 0 <= pos < min(self.max_L, self.geom.attn_maxl):
            raise ValueError(f"position {pos} is outside the cache")
        with driver_scope("gpu"):
            return self._step(token, pos)

    def _step(self, token, pos):
        import gpu_kernels as gk

        nq, nkv, dh, eps = self.nq, self.nkv, self.dh, self.eps
        x = self.embed[token : token + 1].float().to(self.dev)
        h = gk.rmsnorm(x, self.w[0]["attn_norm"], eps)
        for L in range(self.n_layers):
            w = self.w[L]
            q = self._qkv_post(gk.gemv_q4nx(h, w["qkv"]), L, self._lut(L)[pos], pos)
            win = self._window(L)
            lo = 0 if win is None else max(0, pos - win + 1)
            a = gk.attn_decode_gqa(
                q, self._slab_t[L], self.geom, nq, nkv, dh, pos + 1, lo=lo
            )
            o = gk.gemv_q4nx(a, w["o"])
            x, h = self._tail(x, o, L, "post_attn_norm", w["ffn_norm"])
            gu = gk.gemv_q4nx(h, w["gate_up"])
            d = gk.gemv_q4nx(gu, w["down"], glu_in=True, act=self.act)
            x, h = self._tail(x, d, L, "post_ffn_norm", self._next_norm(L))
        return self._head(h)

    def generate(self, first=None, n_tokens=1, eos=()):
        """Greedy continuation, `first` included. Returns the ids.

        Continues from the prefiller's `current_context_length` rows, read at
        the call: whichever prefill wrote them, the decode reads them in place.
        """
        self._ensure_kv()
        if first is None:
            first = self.pf.last_first_token
        out = [int(first)]
        pos = self.pf.current_context_length
        limit = min(self.max_L, self.geom.attn_maxl)
        with driver_scope("gpu"):
            for _ in range(max(0, n_tokens - 1)):
                if pos >= limit:
                    print(f"[gpu-decode] hit the cache's {limit} rows", flush=True)
                    break
                with self.timer.track("gpu_decode_step"):
                    nxt = int(torch.argmax(self._step(out[-1], pos)))
                pos += 1
                out.append(nxt)
                if nxt in eos:
                    break
        return out


class _IgpuPrefill:
    """Mixed in ahead of a `LlamaPrefill` class; see `igpu_prefill_class`."""

    ENGINE = "igpu"

    def __init__(self, *a, **kw):
        kw["backend"] = "cpu"
        super().__init__(*a, **kw)

    def _alloc_base_kv(self, n_layers, max_seq):
        """No host cache: the decoder's slab is the cache."""

    def load_weights(self, model=None):
        """The decoder's packed weights only; no host copy."""
        if model is not None:
            self.model = model
        self.gpu_decoder()

    def make_npu_resident(self):
        """Nothing to convert: no host weights are loaded."""

    def gpu_decoder(self):
        dec = getattr(self, "_gpu_decoder", None)
        if dec is None:
            dec = LlamaGpuDecode(self, max_L=self.max_seq)
            self._gpu_decoder = dec
        return dec

    def bind_kv_sink(self, sink):
        super().bind_kv_sink(sink)
        self.gpu_decoder().bind_kv_sink(sink)

    def prefill(self, ids):
        return self.gpu_decoder().prefill(ids)

    def clear_context(self):
        self.current_context_length = 0

    def kv_view(self, layer_idx):
        dec = self.gpu_decoder()
        return tuple(
            gather(dec._slab_t, layer_idx, r, dec.geom, self.current_context_length)
            .float()
            .cpu()
            .numpy()
            for r in (K_REGION, V_REGION)
        )

    def kv_stack(self):
        return self.gpu_decoder().kv_stack(self.current_context_length)


def igpu_prefill_class(base):
    """`base` (a `LlamaPrefill` class) with its prefill on the iGPU
    (`LlamaGpuDecode.prefill`): `--prefill-engine igpu`."""
    return type(
        f"Igpu{base.__name__}",
        (_IgpuPrefill, base),
        {"__doc__": _IgpuPrefill.__doc__},
    )
