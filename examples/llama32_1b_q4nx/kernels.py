# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Triton kernels for the Llama-3.2-1B prefill on the AMD XDNA NPU.

Every kernel here is written for the NPU backend's constraints: constexpr
shapes, power-of-two extents (`tl.arange`), bf16 in with f32 accumulation, and
a transform script naming the tiling. The wrappers pad to the block sizes the
transform scripts assume and slice back afterwards.
"""

import ctypes
import math
import os
import weakref

import torch
import triton
import triton.language as tl

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.dirname(_HERE)


def script(*candidates):
    """First existing transform script among `examples`-relative candidates.

    Reusing the scripts that ship with the standalone examples rather than
    writing new ones: each is already known to lower and run on this hardware,
    and a transform script is the part of an NPU kernel that is hardest to get
    right from scratch.
    """
    for c in candidates:
        p = c if os.path.isabs(c) else os.path.join(_EXAMPLES, c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no transform script among {candidates}")


# ---------------------------------------------------------------------------
# Dispatch plumbing
# ---------------------------------------------------------------------------
def launch(kernel, grid, *args, transform_script=None, **constexprs):
    """Run a Triton kernel on the NPU.

    Triton's own JIT cache does the caching. This used to go through a
    hand-rolled dict keyed on (kernel, grid, constexprs) that called the
    compiled C extension directly, inherited from
    examples/gpt2/kernels/backend_utils.py on its claim of "0.1ms vs ~27ms per
    dispatch". Measured on this backend the two warm paths are 1.36 ms vs
    1.45 ms for a GEMM and indistinguishable for an elementwise kernel -- about
    1.3% of a prefill. The 27 ms it was really describing was the launcher
    rebuilding the XRT session, which both paths paid and which is now
    process-lifetime state in the driver.

    So all that is left to do here is scope the two globals a compile needs.

    NOTE: neither cache keys on the transform script. Triton hashes the kernel
    source; a second script for the same kernel would silently reuse the first
    binary. Not reachable today -- each kernel here has exactly one script --
    but it is a trap for anyone autotuning by swapping scripts.
    """
    with _npu_driver(), _tiling_script(transform_script):
        kernel[grid](*args, **constexprs)


class _npu_driver:
    """Make the NPU driver active for the duration of a launch.

    Restores whatever was active on entry rather than resetting: on a host with
    no iGPU there is no driver to auto-detect back to. Reading `.active` is
    itself what resolves the default, and on an iGPU-free host that raises
    "0 active drivers" -- so only read it if one has already been chosen.
    """

    def __enter__(self):
        from triton.backends.amd_triton_npu.driver import NPUDriver

        self._prev = getattr(triton.runtime.driver, "_active", None)
        triton.runtime.driver.set_active(NPUDriver())

    def __exit__(self, *exc):
        if self._prev is not None:
            triton.runtime.driver.set_active(self._prev)


class _tiling_script:
    """Point AIR_TRANSFORM_TILING_SCRIPT at `path` for the duration."""

    _VAR = "AIR_TRANSFORM_TILING_SCRIPT"

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self._old = os.environ.get(self._VAR)
        if self.path:
            os.environ[self._VAR] = self.path

    def __exit__(self, *exc):
        if self._old is not None:
            os.environ[self._VAR] = self._old
        elif self.path:
            os.environ.pop(self._VAR, None)


def _pow2(n):
    return 1 << (n - 1).bit_length()


def _pad2d(x, m, n):
    return torch.nn.functional.pad(x, (0, n - x.shape[1], 0, m - x.shape[0]))


# ---------------------------------------------------------------------------
# Matmul  --  C[M,N] = A[M,K] @ B[K,N], bf16 in, f32 accumulate
# ---------------------------------------------------------------------------
@triton.jit
def _matmul_kernel(
    A,
    B,
    C,
    stride_am: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # BLOCK_K spans all of K: the transform script tiles the reduction across
    # L3/L2/L1 on-device, so the host does not chunk it.
    a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :])
    b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :])
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :], tl.dot(a, b))


BLOCK_M = 128  # the matmul transform script's herd tiling assumes >= 128
BLOCK_N = 256

# Every wrapper below tiles the ROW (sequence) dimension on the host at this
# granularity, so no kernel's grid or constexprs depend on the prompt length.
#
# They used to. A kernel is compiled per (grid, constexprs) and cached on that,
# so the first prompt long enough to need a bigger pad recompiled the whole
# model -- measured at 231 s for the step from 128 to 256 rows, during which a
# chat REPL simply sits there. Context grows every turn, so that is not an edge
# case, it is turn three.
#
# Fixed row tiles cost extra dispatches on a long prompt, which is cheap now
# that a warm dispatch is ~0.2-3 ms, and buy one compile for any length.
ROW_TILE = 128

# Padded weights, keyed by storage address. A projection is reused every
# layer-step for the life of the process, and padding + contiguous on a
# [2048, 8192] bf16 weight is a 32 MB host copy -- doing it per call cost more
# than the NPU dispatch it was preparing for.
#
# Not a WeakKeyDictionary: `weakref.ref.__eq__` compares its referents with
# `==`, and on a tensor that yields a tensor, so any bucket collision raises
# "Boolean value of Tensor with more than one value is ambiguous". Key on
# `data_ptr()` instead and confirm identity on the way out -- an address can be
# reused once its tensor is freed, which the finalizer below also guards.
_wcache = {}


def _resident_copy(b):
    """`b`, moved into memory the NPU can read where it lies -- or `b` itself.

    A staged argument is copied into a device buffer on every launch and, on
    HSA, its cache is flushed over the same bytes. For a weight that is written
    once and read for the life of the process, both are pure waste: a 16-layer
    prefill copies ~2.2 GB per call, nearly all of it weights that never
    change.

    A shared buffer is dispatched on in place, so neither happens. Marking it
    resident drops the flush too, which is sound here for the reason the flush
    exists: nothing writes these pages again after this function returns.

    Returns `b` unchanged if the runtime is not HSA (XRT stages through a BO of
    its own) or if shared memory is unavailable -- this is an optimisation, and
    a kernel that runs slower is better than one that does not run.
    """
    if os.environ.get("AMD_TRITON_NPU_RUNTIME") != "hsa":
        return b
    try:
        from triton.backends.amd_triton_npu import shared
        from triton.backends.amd_triton_npu.driver import load_hsa_runtime

        buf = shared.empty(tuple(b.shape), dtype=b.dtype, device="hsa:0")
        t = buf.torch()
        t.copy_(b)
        lib = load_hsa_runtime()
        err = ctypes.create_string_buffer(512)
        # The copy above still sits in the host's cache, and a resident buffer
        # is declared as a token span, so nothing would flush it: the first
        # dispatch would read whatever was in memory, and later ones would be
        # correct only because the lines had since been evicted. Marking the
        # region resident also marks it dirty, so the next dispatch declares it
        # in full and ROCR flushes it -- the runtime keeps that bargain rather
        # than leaving it to every caller to remember.
        if lib.triton_npu_hsa_shared_set_resident(
            ctypes.c_void_p(t.data_ptr()), ctypes.c_int(1), err, ctypes.c_size_t(512)
        ):
            raise RuntimeError(err.value.decode())
        # The buffer owns the pages; the tensor is a view of them, so it has to
        # keep the buffer alive for as long as anything holds the view.
        t._shared_buffer = buf
        return t
    except Exception as e:  # noqa: BLE001 -- see the docstring
        if os.environ.get("AMD_TRITON_NPU_DEBUG"):
            print(f"[kernels] weights stay staged: {e}", flush=True)
        return b


def shared_empty(shape, dtype):
    """An uninitialised tensor the NPU can reach where it lies, or a plain one.

    Same idea as `_resident_copy`, for a buffer whose contents *do* change: the
    dispatch still flushes it, but there is no staging copy in either direction
    -- the device reads and writes the caller's own pages. An output allocated
    this way is never copied back at all.

    The buffer is pooled by `shared.py`, so this is not an allocation per call;
    it is attached to the returned tensor so the pages are reclaimed only once
    nothing is looking at them.
    """
    if os.environ.get("AMD_TRITON_NPU_RUNTIME") != "hsa":
        return torch.empty(shape, dtype=dtype)
    try:
        from triton.backends.amd_triton_npu import shared

        buf = shared.empty(tuple(shape), dtype=dtype, device="hsa:0")
        t = buf.torch()
        t._shared_buffer = buf
        return t
    except Exception as e:  # noqa: BLE001 -- an optimisation, never a blocker
        if os.environ.get("AMD_TRITON_NPU_DEBUG"):
            print(f"[kernels] activation stays staged: {e}", flush=True)
        return torch.empty(shape, dtype=dtype)


def _padded_weight(w, Kp, Np):
    addr = w.data_ptr()
    entry = _wcache.get(addr)
    if entry is not None:
        ref, dims, padded = entry
        if ref() is w and dims == (Kp, Np):
            return padded
    b = _resident_copy(_pad2d(w.to(torch.bfloat16), Kp, Np).contiguous())
    _wcache[addr] = (
        weakref.ref(w, lambda _r, a=addr: _wcache.pop(a, None)),
        (Kp, Np),
        b,
    )
    return b


def triton_matmul(x, w, block_m=BLOCK_M, block_n=BLOCK_N):
    """x: [M, K] float32/bf16, w: [K, N] bf16 -> [M, N] float32, on the NPU."""
    M, K = x.shape
    Kw, N = w.shape
    assert K == Kw, (x.shape, w.shape)
    Mp = math.ceil(M / block_m) * block_m
    Np = math.ceil(N / block_n) * block_n
    Kp = _pow2(K)  # tl.arange needs a power of two
    b = _padded_weight(w, Kp, Np)
    # Pad input and output ONCE, then hand each dispatch a contiguous row slice
    # of them. Allocating per tile and copying the result back instead cost
    # more than the dispatch: an [128, 16384] f32 copy per GEMM, 64 GEMMs deep.
    a = shared_empty((Mp, Kp), torch.bfloat16)
    a.zero_()
    a[:M, :K] = x.to(torch.bfloat16)
    c = shared_empty((Mp, Np), torch.float32)
    # One block_m-row tile per dispatch: the grid is (1, N/block_n), which
    # depends on the weight alone and never on the prompt length.
    for m0 in range(0, Mp, block_m):
        launch(
            _matmul_kernel,
            (1, Np // block_n),
            a[m0 : m0 + block_m],
            b,
            c[m0 : m0 + block_m],
            Kp,
            Np,
            Np,
            transform_script=script("gpt2/transform_matmul_aie2p.mlir"),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=Kp,
        )
    return c[:M, :N]


# ---------------------------------------------------------------------------
# SwiGLU  --  silu(gate) * up, elementwise over a flat vector
# ---------------------------------------------------------------------------
@triton.jit
def _swiglu_kernel(G, U, Y, BLOCK: tl.constexpr):
    """SiLU(gate) * up, as examples/swiglu computes it.

    tl.sigmoid needs f32, so the product is formed in f32 and rounded back to
    bf16 before the multiply by `up` -- the transform script's vector type
    casts assume exactly this shape.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(G + offs[:])
    up = tl.load(U + offs[:])
    gate_f32 = gate.to(tl.float32)
    silu_gate = (gate_f32 * tl.sigmoid(gate_f32)).to(gate.dtype)
    tl.store(Y + offs[:], silu_gate * up)


SWIGLU_BLOCK = 1024


def triton_swiglu(gate, up, block=SWIGLU_BLOCK):
    """silu(gate) * up over matching [M, N] tensors -> [M, N] float32."""
    shape = gate.shape
    # Fixed-size chunks, so the grid never tracks the prompt length (ROW_TILE).
    chunk = ROW_TILE * (shape[-1] if gate.ndim > 1 else block)
    chunk = math.ceil(chunk / block) * block
    n = gate.numel()
    npad = math.ceil(n / chunk) * chunk
    # Pad once; each dispatch gets a contiguous slice, no per-chunk copy back.
    g = torch.nn.functional.pad(gate.reshape(-1).to(torch.bfloat16), (0, npad - n))
    u = torch.nn.functional.pad(up.reshape(-1).to(torch.bfloat16), (0, npad - n))
    g, u = g.contiguous(), u.contiguous()
    y = torch.empty(npad, dtype=torch.bfloat16)
    for o in range(0, npad, chunk):
        launch(
            _swiglu_kernel,
            (chunk // block,),
            g[o : o + chunk],
            u[o : o + chunk],
            y[o : o + chunk],
            transform_script=script("swiglu/transform_aie2p.mlir"),
            BLOCK=block,
        )
    return y[:n].to(torch.float32).reshape(shape)


# ---------------------------------------------------------------------------
@triton.jit
def _rms_norm_kernel(
    X,
    W,
    Y,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """y = x * rsqrt(mean(x^2) + eps) * w, per row.

    examples/weighted_rms_norm's kernel, BLOCK_M rows at a time so the row
    reduction is a vector op rather than a scalar chain -- but reducing in f32.
    That example rounds the squares to bf16 before summing and is only tested
    at N=256; at our N=2048 a bf16 sum of 2048 terms carries ~9% relative
    error, which is visible in the model output.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]
    x = tl.load(X + offsets)
    w = tl.load(W + cols)

    x_f32 = x.to(tl.float32)
    x_sq = x_f32 * x_f32
    sum_sq = tl.sum(x_sq, axis=1)
    rstd = tl.math.rsqrt(sum_sq / N + eps)

    y = x_f32 * rstd[:, None] * w.to(tl.float32)[None, :]
    tl.store(Y + offsets, y.to(x.dtype))


RMS_BLOCK_M = 2


def triton_rms_norm(x, weight, eps, block_m=RMS_BLOCK_M):
    """x: [M, D] -> RMS-normalized and scaled by `weight` [D]."""
    M, dim = x.shape
    Mp = math.ceil(M / ROW_TILE) * ROW_TILE
    wb = weight.to(torch.bfloat16).contiguous()
    # Pad once; each dispatch gets a contiguous row slice, no copy back.
    xb = torch.nn.functional.pad(x.to(torch.bfloat16), (0, 0, 0, Mp - M)).contiguous()
    y = torch.empty_like(xb)
    # Fixed ROW_TILE-row chunks, so the grid is constant (ROW_TILE).
    for m0 in range(0, Mp, ROW_TILE):
        launch(
            _rms_norm_kernel,
            (ROW_TILE // block_m,),
            xb[m0 : m0 + ROW_TILE],
            wb,
            y[m0 : m0 + ROW_TILE],
            dim,
            float(eps),
            transform_script=script("llama32_1b_q4nx/transform_rms_norm_aie2p.mlir"),
            BLOCK_M=block_m,
            BLOCK_N=dim,
        )
    return y[:M].to(torch.float32)
