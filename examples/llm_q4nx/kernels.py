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
import re
import os
import weakref

import numpy as np
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
    source and its constexprs; a second script for the *same* constexprs would
    silently reuse the first binary, and the second shape would run a schedule
    chosen for the first. That stopped being hypothetical when Qwen2.5-7B began
    choosing a schedule per GEMM, so `_check_script` below turns it into an
    error instead of leaving it to this comment.
    """
    _check_script(kernel, constexprs, transform_script)
    with _npu_driver(), _tiling_script(transform_script):
        kernel[grid](*args, **constexprs)


#: (kernel, constexprs) -> the transform script it was first launched with.
_script_for_key = {}


def _check_script(kernel, constexprs, transform_script):
    """Refuse a second transform script for constexprs already compiled.

    Triton's cache would hand back the first binary, so the second script would
    be silently ignored -- the caller gets a schedule it did not ask for, at a
    shape the first one may not lower correctly for. Raising names both scripts;
    the fix is to give the two shapes different constexprs, or to settle on one
    schedule for both.
    """
    key = (kernel, tuple(sorted(constexprs.items())))
    first = _script_for_key.setdefault(key, transform_script)
    if first != transform_script:
        raise RuntimeError(
            f"{getattr(kernel, '__name__', kernel)} was already compiled for "
            f"{dict(key[1])} with transform script {first!r}; Triton's cache "
            f"does not key on the script, so {transform_script!r} would be "
            f"ignored and the first schedule used instead."
        )


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


#: The smallest output block one AIE core is given under these schedules --
#: their `l1_m`/`l1_n` floor. A `block_m x block_n` GEMM tile asks for a herd
#: of `(block_m/l1_m, block_n/l1_n)` cores, which is how a tile becomes
#: occupancy.
L1 = 64

#: npu2's compute array. `air-to-aie` lays herd dimension 0 across COLUMNS and
#: dimension 1 across ROWS, so a block that covers the array is 8 cores wide in
#: M and 4 in N. Taken the other way round it does not run slower, it fails to
#: place: a herd needing five rows raises "row index (6) must be less than
#: ... (6)".
AIE_COLS, AIE_ROWS = 8, 4

BLOCK_M = 128  # the matmul transform script's herd tiling assumes >= 128

#: The largest core tile these schedules can drain. One L1->L2 output copy is
#: `l1_m * l1_n` 32-bit words and a DMA buffer descriptor carries at most
#: 16383, so 128x128 misses by a single word and 8192 is the ceiling.
MAX_L1_NUMEL = 8192

#: ...which, over the whole array, bounds the block itself. This is what stops
#: both tile dimensions from growing at once: 1024x256 and 512x512 are both
#: exactly at it, and 1024x512 is twice over.
MAX_BLOCK_NUMEL = AIE_COLS * AIE_ROWS * MAX_L1_NUMEL

#: Row tiles a GEMM may be compiled for, widest first. A tile is computed in
#: full whether or not the rows fill it, so the widest is not the fastest at
#: every prompt length -- `row_tier` chooses. The top of the range is past the
#: array's width: `L1 * AIE_COLS` is the narrowest block that reaches all eight
#: columns, and doubling it keeps the same herd while halving how often the
#: weight is re-read. Whether it is reachable depends on `block_n`, through
#: `MAX_BLOCK_NUMEL`.
WIDE_M = L1 * AIE_COLS
ROW_TIERS = (2 * WIDE_M, WIDE_M, 256, BLOCK_M)

#: Column tiles, the same way. A GEMM re-reads the whole of A once per column
#: tile, so the widest tile the weight can fill moves the fewest bytes -- at
#: the same core count, since `MAX_BLOCK_NUMEL` then takes it back out of the
#: rows. Which of the two axes should give way is the caller's cost model, not
#: a property of the array: see `tile_n` for the dispatch-per-row-block path
#: and `fused_mlp` for the chained one. `col_tier` chooses the tile itself; a
#: narrow projection takes the small one rather than padding to a tile it
#: cannot fill.
COL_TIERS = (2 * L1 * AIE_ROWS, L1 * AIE_ROWS)

#: The matmul schedules, by the core tile `(l1_m, l1_n)` they were generated
#: for. The tile does not simply track the block, and that is measured rather
#: than assumed: holding the herd at eight columns by shrinking `l1_m` is
#: slower than letting the herd shrink, at both 256 and 128 rows. Below 512
#: rows there are not enough of them to give every column a tile worth
#: having.
_MATMUL_SCRIPTS = {
    (64, 64): "gpt2/transform_matmul_aie2p.mlir",
    (128, 64): "llm_q4nx/transform_matmul_m128_aie2p.mlir",
    (64, 128): "llm_q4nx/transform_matmul_n128_aie2p.mlir",
}

#: The K depth every checked-in schedule was generated for.
L2_K = 64

#: The same core tiles staged 256 deep in L2 instead, still reducing 64 at a
#: time in L1 (`l1_k`). Every L3->L2 round costs a fixed setup whatever it
#: carries, and a long reduction pays it K / l2_k times: Gemma4's `down`
#: contracts 12288, so 192 rounds at 64 and 48 at 256 -- mlir-air's own
#: `tile_k_l2` for that GEMM. 7.49 -> 5.43 ms on the wide `down` at M=1024,
#: bit-identical, since each core still reduces in the same order.
#:
#: Not the default, because the weight walk's DMA stride is `l2_k * N` and
#: npu2 caps a BD stride at 2^20 words: at the FFN width of `gate`/`up`
#: (12288) 256 is three times over and aiecc rejects it. A caller with a long
#: K and a narrow N asks for it by name.
DEEP_L2_K = 256
_DEEP_K_SCRIPTS = {
    (64, 128): "llm_q4nx/transform_matmul_n128_k256_aie2p.mlir",
}


#: `triton_matmul`'s default: pick the schedule from the block shape. Distinct
#: from `None`, which means "let the driver generate one" and is what
#: `qwen25_prefill` passes for the shape these schedules reject.
BY_BLOCK = "<chosen from the block shape>"


def matmul_script(block_m, block_n, l2_k=L2_K):
    """The schedule to lower a `block_m x block_n` GEMM tile with.

    The core tile is the largest that still spreads the block over the whole
    array, floored at `L1` -- see `_MATMUL_SCRIPTS` for why the floor is there
    and not lower. `l2_k` is `L2_K` or `DEEP_L2_K`; see the latter for when.
    """
    l1 = (max(L1, block_m // AIE_COLS), max(L1, block_n // AIE_ROWS))
    if l2_k == DEEP_L2_K and l1 in _DEEP_K_SCRIPTS:
        return script(_DEEP_K_SCRIPTS[l1])
    if l2_k != L2_K:
        raise ValueError(f"no {l1[0]}x{l1[1]} schedule with l2_k={l2_k}")
    if l1 not in _MATMUL_SCRIPTS:
        raise ValueError(
            f"a {block_m}x{block_n} block wants an l1 tile of {l1[0]}x{l1[1]}, "
            f"which has no schedule; blocks are bounded by MAX_BLOCK_NUMEL "
            f"({MAX_BLOCK_NUMEL}) and this one is {block_m * block_n}"
        )
    return script(_MATMUL_SCRIPTS[l1])


def row_tier(n_rows, tiers=ROW_TIERS, cap=None):
    """The row tile to run `n_rows` through: fewest dispatches, then smallest.

    Fewest dispatches because a wider tile spreads over proportionally more
    columns, so what it really buys is one dispatch instead of several;
    smallest on a tie because two tiers needing the same number of dispatches
    differ only in how much of the last one is padding.

    `cap` bounds the tile where the caller has a ceiling of its own -- Triton's
    tensor limit, which binds here as `block_m * Kp`, and `MAX_BLOCK_NUMEL`.
    """
    ok = [t for t in tiers if cap is None or t <= cap] or [min(tiers)]
    return min(ok, key=lambda t: (-(-n_rows // t), t))


def col_tier(n_cols):
    """The column tile for an `n_cols`-wide GEMM. `row_tier`'s rule, one axis over.

    The same trade and so the same rule: fewest tiles, because A is re-read
    once per tile, and the narrower one on a tie, because then the tiers differ
    only in padding.
    """
    return row_tier(n_cols, tiers=COL_TIERS)


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
    try:
        from triton.backends.amd_triton_npu import shared

        dev = "hsa:0" if os.environ.get("AMD_TRITON_NPU_RUNTIME") == "hsa" else "xrt:0"
        buf = shared.empty(tuple(shape), dtype=dtype, device=dev)
        t = buf.torch()
        # The caller binds it (`shared_bo`); without that the pages are still
        # staged and copied back like any other array, and the only thing this
        # buys is the pooling.
        t._shared_buffer = buf
        return t
    except Exception as e:  # noqa: BLE001 -- an optimisation, never a blocker
        if os.environ.get("AMD_TRITON_NPU_DEBUG"):
            print(f"[kernels] activation stays staged: {e}", flush=True)
        return torch.empty(shape, dtype=dtype)


#: `shared_empty` pages kept alive by shape, so a given `bo_key` always binds
#: the SAME buffer. `shared.py` pools its allocations, so without this a second
#: call under one key gets a different BO and the runner refuses it -- "already
#: bound to a different buffer for arg 0". Keyed by shape and dtype only: the
#: contents are overwritten before every dispatch, so two call sites at one
#: shape can share the page.
_IO_PAGES = {}


def io_page(shape, dtype):
    """A `shared_empty` tensor for this shape, allocated once and reused."""
    key = (tuple(shape), dtype)
    t = _IO_PAGES.get(key)
    if t is None:
        t = shared_empty(shape, dtype)
        _IO_PAGES[key] = t
    return t


def shared_bo(t):
    """The XRT buffer object behind a `shared_empty` tensor, or None.

    `bound_buffers` takes these: the chain then dispatches on the caller's own
    pages instead of staging a copy in and copying the result back out. At
    qkv's shape that is 6 MiB in and 42 MiB out, on every one of 35 layers.
    """
    buf = getattr(t, "_shared_buffer", None)
    return None if buf is None else buf.bo


class ResidentWeight:
    """A weight already padded for the NPU, carrying the dims it came from.

    `_padded_weight` below caches its copy against the *original* tensor: keyed
    on its address, weakref'd so the entry dies with it. That is right for a
    cache, and it means the original can never be freed -- the copy would go
    with it. So an NPU-only run holds both, which on Qwen3-4B is 6.8 GiB of
    unpadded weights kept alive solely to anchor 10.6 GiB of padded ones.

    This is the other arrangement: pad once, keep the logical `(K, N)` beside
    the buffer, and hold no reference to the original at all, so the caller can
    drop it. `shape` is exposed so `triton_matmul` reads the logical dims from
    either kind of weight without asking which it has.

    The trade is that there is no way back: a `ResidentWeight` cannot be used by
    the torch reference path, and `_matmul` says so rather than failing
    obscurely.
    """

    __slots__ = ("padded", "K", "N")

    def __init__(self, padded, K, N):
        self.padded, self.K, self.N = padded, K, N

    @property
    def shape(self):
        return (self.K, self.N)


def resident_weight(w, block_n=None, exact=False):
    """Pad `w` for the NPU now, returning a handle that does not reference it.

    `exact` contracts K itself rather than the power of two above it, which
    only the chain path can dispatch (`triton_matmul`'s `stage_key`): the
    per-row-block loop compiles a `@triton.jit` kernel and `tl.arange` needs a
    power of two. The padding is fixed here at load time, so the caller has to
    know which way its consumer will dispatch -- see `CHAIN_WEIGHTS`.
    """
    K, N = w.shape
    Kp = exact_k(K) if exact else _pow2(K)
    block_n = tile_n(N, Kp) if block_n is None else block_n
    Np = math.ceil(N / block_n) * block_n
    padded = _resident_copy(_pad2d(w.to(torch.bfloat16), Kp, Np).contiguous())
    return ResidentWeight(padded, K, N)


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


def unaliased_stride(n_elems):
    """A row stride of at least `n_elems` that is not a power of two.

    DDR interleaves its banks on a power-of-two byte boundary, so a row stride
    that is itself a power of two puts every row of a tile in the same bank.
    It is not a small effect: on mlir-air's GEMM with everything else held
    fixed, a K=2048 contraction costs 48% more per K step than K=1920 or 2112,
    and `tl.arange` makes every contraction here a power of two.

    The *extent* has to stay one; the stride does not. One L1 tile of slack
    buys the rows back and costs one column block that nothing reads. Strides
    that are already off the boundary are returned unchanged.
    """
    return n_elems + L1 if n_elems and not n_elems & (n_elems - 1) else n_elems


#: Triton's own cap on the element count of one tile (`tl.load` of
#: `[BLOCK_K, BLOCK_N]`). Not a device limit and not tunable -- the frontend
#: refuses to build the tensor. It binds here because `BLOCK_K` is the *whole*
#: padded reduction: at Kp=32768 a 256-wide tile is 8388608 elements, twice the
#: maximum, and every model whose MLP intermediate exceeds 16384 hits it in the
#: `down` projection. Applied before dispatch, so the fix is a narrower tile
#: rather than a CompilationError from inside the frontend pointing at a
#: `tl.load`.
MAX_TILE_NUMEL = 4194304

#: The matmul schedule. Hand-written for gpt2 and reused by every model here.
#: It does not lower at every shape -- see `triton_matmul`'s `transform_script`.
MATMUL_SCRIPT = "gpt2/transform_matmul_aie2p.mlir"


def narrow_k(src, k, kp, block_m, block_n):
    """Narrow a captured GEMM module's contraction from `kp` down to `k`.

    `tl.arange` requires a power of two, so a `@triton.jit` GEMM can only ask
    for `kp` where the real reduction is `k` -- for every projection in a model
    whose D is not a power of two, that is 33% of the weight traffic spent on
    zeros. `linalg.matmul` carries no such rule and the schedules tile K by
    `l2_k`, so a module stating `k` needs nothing else changed.

    Only the K EXTENT is edited. Both row strides are kernel constexprs, so a
    capture taken at the real stride already has every offset right, and these
    four are all that is left: A is `[block_m, K]` and B is `[K, block_n]`.

    Measured on Gemma4's wide gate, M=1024 N=12288: 8.76 -> 6.54 ms at an
    unchanged 5.9 TFLOP/s -- the same efficiency over 25% fewer bytes.
    """
    if k == kp:
        return src
    if isinstance(src, bytes):
        src = src.decode()
    # The two `sizes:` patterns are pinned to a bf16 operand's cast. C's cast
    # reads `sizes: [block_m, block_n]`, which is textually B's pattern
    # whenever block_m == kp (and A's whenever block_n == kp) -- and C is f32.
    bf16_cast = r"(?=, strides: \[[^\]]*\] : memref<\*xbf16>)"
    for pat, rep in (
        (rf"sizes: \[{block_m}, {kp}\]{bf16_cast}", f"sizes: [{block_m}, {k}]"),
        (rf"sizes: \[{kp}, {block_n}\]{bf16_cast}", f"sizes: [{k}, {block_n}]"),
        (rf"\b{block_m}x{kp}xbf16\b", f"{block_m}x{k}xbf16"),
        (rf"\b{kp}x{block_n}xbf16\b", f"{k}x{block_n}xbf16"),
    ):
        src, hits = re.subn(pat, rep, src)
        if not hits:
            raise RuntimeError(f"narrow_k matched nothing: {pat}")
    return src


#: A reduction is contracted exactly when it is a multiple of this. The
#: schedules tile K by `l2_k`, at most 256 across the checked-in scripts, and a
#: K that is not a multiple of its tile would leave a partial one the grid
#: cannot express. Gemma4's 1536 clears it; a prime D would fall back to
#: padding, which is still correct.
K_EXACT_GRAIN = 256


def exact_k(K):
    """The reduction to contract for a weight with `K` rows: `K` or its pad."""
    kp = _pow2(K)
    return K if K != kp and K % K_EXACT_GRAIN == 0 else kp


def tile_n(N, Kp):
    """The column tile for an `N`-wide weight contracted over `Kp`.

    One rule for `triton_matmul` and `resident_weight` both, because a weight
    padded under a different one is a buffer shaped for someone else's grid --
    which is also why it cannot depend on the prompt: the padding is fixed at
    load time and the rows are not known until the call.

    So the rows get first claim on `MAX_BLOCK_NUMEL` and the columns take what
    is left. That is the right way round *here* and only here: this path issues
    one dispatch per row block, each costing its own fixed overhead whatever it
    contains, so it always wants the widest row tier. `fused_mlp` chooses the
    other way round because a chain carries its row blocks in the grid and pays
    nothing for them.

    The remaining cap is Triton's, not the device's: the tile is
    `Kp x block_n` elements and the frontend refuses to build a larger tensor.
    Qwen2.5-7B's `down` (Kp=32768) is the only shape here that it narrows.
    """
    if Kp > MAX_TILE_NUMEL:
        raise ValueError(
            f"K pads to {Kp}, which exceeds Triton's {MAX_TILE_NUMEL} maximum "
            f"tensor size on its own; no block_n can fit."
        )
    return min(col_tier(N), MAX_TILE_NUMEL // Kp, MAX_BLOCK_NUMEL // ROW_TIERS[0])


def _np(t):
    """`t` as a numpy array, bf16 included.

    numpy has no bfloat16, so a bf16 tensor is reinterpreted through uint16
    into `ml_dtypes`'. Nothing converts: the bytes are the same either way, and
    a chain wants numpy because that is what its staging copies from.
    """
    if t.dtype is torch.bfloat16:
        from ml_dtypes import bfloat16

        return t.view(torch.uint16).numpy().view(bfloat16)
    return t.numpy()


#: Per-core L1 an elementwise tile's operands may occupy. The core has 64 KiB;
#: 48 places and 96 does not -- measured, the failure is "'aie.tile' op
#: allocated buffers exceeded available memory".
ELEM_L1_BYTES = 48 * 1024


def elem_block(n_elems, bytes_per_elem):
    """The elementwise tile to cover `n_elems` with, or the cap if None.

    Bounded by BYTES per core and not by a fixed element count, because that is
    what the hardware bounds. `@flatten_tile_forall_aie2p` splits a block over
    the 8 columns with `num_threads [8]`, so a core holds `block / 8` elements
    of every operand at once and `bytes_per_elem` is their widths summed.

    The tile is most of what an elementwise pass costs. Fitting
    `t = a*elements + b*bytes` over a bf16 add at 12.58M elements gives 74 GB/s
    of marginal bandwidth against a 5.7 Gelem/s fixed term -- two thirds of the
    time is per-element, which is really per-TILE work charged against a fixed
    tile length. Lengthening the tile amortises it, monotonically to the L1
    bound: three bf16 streams run 3.25 ms at 16384 and 1.85 at 65536; three
    f32 streams run 4.15 at 16384 and 3.46 at 32768, and 65536 does not place.

    `n_elems` is the range to cover. The grid is `n_elems // block`, so a block
    that does not divide the range truncates it and drops the tail with no
    error -- the same trap `D_pad` carries against `BLOCK_N`. Taking the
    largest power of two that divides makes it impossible instead of merely
    unlikely, and `tl.arange` wants a power of two anyway. Pass None where the
    caller sizes its own range around the block.
    """
    cap = 1 << ((AIE_COLS * ELEM_L1_BYTES) // bytes_per_elem).bit_length() - 1
    return cap if n_elems is None else min(cap, n_elems & -n_elems)


#: Chains that hold a projection's weight on the device between calls, keyed by
#: the compiled shape. One per shape and NOT one per weight: a chain owns an
#: `hw_context` and the NPU runs out of those at around 35 (see `fused_mlp`),
#: so a layer's identity is the `bo_key` instead, which is the same arrangement
#: the FFN chain uses.
_PROJ_CHAINS = {}


def _proj_chain(block_m, block_n, Mp, Np, Kp, a_stride, transform_script):
    """The chain that runs a `Mp x Kp @ Kp x Np` projection, built once.

    The whole grid in one dispatch rather than the host's row-block loop, so
    the weight is one operand of one op and `static_indices` can hold it.
    """
    key = (block_m, block_n, Mp, Np, Kp, a_stride, transform_script)
    chain = _PROJ_CHAINS.get(key)
    if chain is not None:
        return chain
    from triton.backends.amd_triton_npu.multilaunch import NPUChain

    chain = NPUChain(f"proj_{Mp}x{Kp}x{Np}_{block_m}x{block_n}")
    # Captured at the power of two the frontend insists on, then narrowed to
    # the reduction actually wanted -- see `narrow_k`. At a `Kp` that is
    # already exact the capture is used unchanged.
    kp = _pow2(Kp)
    tscript = script(transform_script) if isinstance(transform_script, str) else None
    src = chain._capture_ttshared(
        _matmul_kernel,
        (Mp // block_m, Np // block_n),
        (
            torch.zeros((Mp, a_stride), dtype=torch.bfloat16),
            torch.zeros((kp, Np), dtype=torch.bfloat16),
            torch.zeros((Mp, Np), dtype=torch.float32),
            a_stride,
            Np,
            Np,
        ),
        {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": kp},
    )
    chain.add(
        narrow_k(src, Kp, kp, block_m, block_n),
        grid=(Mp // block_m, Np // block_n),
        arg_map={0: 0, 1: 1, 2: 2},
        args=(),
        transform_script=tscript,
    )
    _PROJ_CHAINS[key] = chain
    return chain


def triton_matmul(
    x, w, block_m=None, block_n=None, transform_script=BY_BLOCK, stage_key=None
):
    """x: [M, K] float32/bf16, w: [K, N] bf16 -> [M, N] float32, on the NPU.

    Both block dimensions default to whatever `row_tier` and `tile_n` think
    this shape is worth: a tile is computed in full, so the widest one is the
    fastest only once the rows and columns fill it. Each distinct value
    compiles its own kernel, so the tiers are few and fixed rather than
    tracking the prompt.

    `transform_script` is the schedule to lower with, as an `examples`-relative
    path; `BY_BLOCK` picks one to match the block, and None lets the driver
    generate its own. The shared hand-written schedules are not universal, and
    the one shape here that they reject is recorded at the call site that
    passes None (`qwen25_prefill._layer`), because a shape they reject fails
    loudly in aiecc rather than silently.

    `stage_key` names a weight that does not change between calls -- a layer's
    projection -- and dispatches through a chain that stages it once instead of
    per launch. `launch` copies every pointer argument into a device BO on
    every launch, so a 20 MiB weight is otherwise re-staged twice a layer for
    35 layers. Measured at qkv's shape: 19.49 ms against 11.35. Name the weight
    (`f"qkv_L{i}"`); the BO set is keyed on that AND the padded buffer's
    address, so two models of one shape in a process cannot share a weight.
    The address is only stable if `w` is: pass the same tensor every call, not
    one rebuilt per call, or each call stages a fresh copy.
    """
    M, K = x.shape
    Kw, N = w.shape
    assert K == Kw, (x.shape, w.shape)
    # The chain path hands the op to AIR as MLIR, so it can contract K itself;
    # the dispatch loop below still compiles a `@triton.jit` kernel, whose
    # `tl.arange` needs a power of two. A resident weight was padded at load
    # time and is authoritative -- its row count IS the reduction, and a
    # caller that cannot run it says so here rather than dying inside the
    # frontend on an arange.
    if isinstance(w, ResidentWeight):
        Kp = w.padded.shape[0]
    else:
        Kp = exact_k(K) if stage_key is not None else _pow2(K)
    # An exact K can only be dispatched through the chain, which hands AIR the
    # module as MLIR; the per-row-block loop below compiles a `@triton.jit`
    # kernel and `tl.arange` needs a power of two. A resident weight fixed its
    # padding at load time and may reach a call site that passes no
    # `stage_key`, so the reduction decides the path, not the caller.
    via_chain = stage_key is not None or Kp != _pow2(Kp)
    # `None` on either block dimension means "as wide as this shape is worth",
    # which is what a caller almost always wants; an explicit one is honoured
    # as given. Both caps are derived rather than declared per model: Triton's
    # tensor limit on each tile, and `MAX_BLOCK_NUMEL` between them, so the two
    # cannot both widen into a block that has no schedule.
    if block_n is None:
        block_n = tile_n(N, Kp)
    if block_m is None:
        block_m = row_tier(M, cap=min(MAX_TILE_NUMEL // Kp, MAX_BLOCK_NUMEL // block_n))
    if transform_script is BY_BLOCK:
        transform_script = matmul_script(block_m, block_n)
    Mp = math.ceil(M / block_m) * block_m
    Np = math.ceil(N / block_n) * block_n
    if isinstance(w, ResidentWeight):
        # Padded ahead of time, by someone who then freed the original. Its
        # padding was computed from the same rule, but a caller passing a
        # different block_n would silently get a buffer shaped for another
        # grid, so the shape is checked rather than assumed.
        b = w.padded
        if tuple(b.shape) != (Kp, Np):
            raise ValueError(
                f"resident weight padded to {tuple(b.shape)}, but this call "
                f"needs {(Kp, Np)} (block_n={block_n})"
            )
    else:
        b = _padded_weight(w, Kp, Np)
    # Pad input and output ONCE, then hand each dispatch a contiguous row slice
    # of them. Allocating per tile and copying the result back instead cost
    # more than the dispatch: an [128, 16384] f32 copy per GEMM, 64 GEMMs deep.
    # `Kp` is the extent the kernel contracts; `a_stride` is how far apart the
    # rows sit, and they are deliberately not the same number.
    a_stride = unaliased_stride(Kp)
    a = io_page((Mp, a_stride), torch.bfloat16)
    a.zero_()
    a[:M, :K] = x.to(torch.bfloat16)
    # The chain binds `c` by `bo_key`, so it must be the same page every call
    # and is copied out below. The dispatch loop returns `c` itself, so it gets
    # a page of its own: a pooled one would be overwritten by the next call at
    # this shape, under a caller still holding the result.
    c = (
        io_page((Mp, Np), torch.float32)
        if via_chain
        else shared_empty((Mp, Np), torch.float32)
    )
    if via_chain:
        # `static_indices` holds one weight per key, so a key that does not
        # identify the WEIGHT would hand the next call the last one's buffer.
        # The address does identify it. A `stage_key` alone does not: chains
        # are process-global and shared by shape, so a second model of the
        # same shape would reuse the first's `qkv_L0` -- the name only labels.
        key = f"{stage_key or 'mm'}_{b.data_ptr():x}_{Mp}x{Kp}x{Np}"
        # The activation and the result are the caller's own pages where the
        # interop allows it, so neither is staged in nor copied back.
        io = {i: bo for i, bo in ((0, shared_bo(a)), (2, shared_bo(c))) if bo}
        # The build too, not just the dispatch: it warmup-compiles the kernel,
        # which needs a driver, and with no iGPU visible there is no default
        # one to fall back on ("0 active drivers") -- as `FusedMLP.run` scopes.
        with _npu_driver():
            chain = _proj_chain(
                block_m, block_n, Mp, Np, Kp, a_stride, transform_script
            )
            got = chain.run(
                [_np(a), _np(b), _np(c)],
                bo_key=key,
                static_indices={1},
                # The kernel writes all of `c` and the host only reads it, so
                # it needs no host->device sync before the dispatch.
                intermediate_indices={2},
                output_indices={2},
                bound_buffers=io or None,
            )
        if 2 in io:
            # Read where the kernel wrote it, then copied ONCE into a tensor
            # of the caller's own: the page is reused by the next call at this
            # shape, so a view into it would be overwritten under them. That
            # trades a device->host copy of the whole padded buffer for a
            # host->host copy of just the part asked for.
            return c.reshape(Mp, Np)[:M, :N].clone()
        # Unbound, `got[2]` is a view of the runner's own BO for this key, which
        # the next dispatch under it overwrites -- the same lifetime as above.
        return torch.from_numpy(np.asarray(got[2])).reshape(Mp, Np)[:M, :N].clone()
    # One block_m-row tile per dispatch: the grid is (1, N/block_n), which
    # depends on the weight alone and never on the prompt length.
    for m0 in range(0, Mp, block_m):
        launch(
            _matmul_kernel,
            (1, Np // block_n),
            a[m0 : m0 + block_m],
            b,
            c[m0 : m0 + block_m],
            a_stride,
            Np,
            Np,
            transform_script=(
                script(transform_script) if isinstance(transform_script, str) else None
            ),
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


#: Both pad to bf16 and write bf16, so three 2-byte streams. Worth about 2x
#: per layer on the PLE branch's [2040, 256] against the 1024 it replaced,
#: at an unchanged error.
SWIGLU_BLOCK = elem_block(None, 2 + 2 + 2)


def glu_chunk(n, width, block):
    """How much of a GLU's range one dispatch covers. A multiple of `block`.

    The chunk is sized from the row WIDTH rather than the row count, so the
    grid (`chunk // block`) never tracks the prompt length -- the AIR lowering
    is cached on the grid, and one that followed P would recompile per prompt.

    Width alone is not enough when the rows are narrow. Gemma4's PLE branch is
    [P, 256]: `ROW_TILE * 256` is 32768, which rounds up to exactly ONE 65536
    block, so a P=2040 range of 8 blocks went out as eight dispatches of a
    single program each -- 280 for the prefill, every one staging its own
    operands. One dispatch of eight programs is the same arithmetic and the
    same tile, measured 3.409 -> 0.929 ms per layer and bit-identical.

    So take the larger of the width chunk and the whole range, with the range
    rounded DOWN to a power of two and capped. Rounding down keeps the set of
    distinct grids small (1, 2, 4, 8 plus whatever the widths ask for) instead
    of one per prompt length, and the cap stops a wide model's chunk from
    shrinking: at width 12288 the width term is already 24 blocks, and 8 would
    make it three times as many dispatches.
    """
    blocks = max(1, math.ceil(n / block))
    span = block * (1 << min(blocks.bit_length() - 1, _GLU_MAX_POW2))
    return max(math.ceil(ROW_TILE * width / block) * block, span)


#: Largest power-of-two grid `glu_chunk` will reach for on range alone, as an
#: exponent. Eight programs covers the PLE branch at every length this runs at
#: and bounds the distinct grids; wider rows still get their width's chunk.
_GLU_MAX_POW2 = 3


def triton_swiglu(gate, up, block=SWIGLU_BLOCK):
    """silu(gate) * up over matching [M, N] tensors -> [M, N] float32."""
    shape = gate.shape
    n = gate.numel()
    chunk = glu_chunk(n, shape[-1] if gate.ndim > 1 else block, block)
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
#: sqrt(2/pi), doubled. See `_geglu_kernel` for why it is doubled.
_GELU_2C = 1.5957691216057308
_GELU_K = 0.044715


@triton.jit
def _geglu_kernel(G, U, Y, C2: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    """gelu_tanh(gate) * up -- Gemma3's GLU, where Llama and Qwen3 use SiLU.

    The activation is `gelu_pytorch_tanh`, exactly:

        0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3))),  c = sqrt(2/pi)

    and with a real tanh, because AIE2P has one and does not have a reciprocal.
    Its two vector transcendentals are `exp2` and `tanh`
    (`aie2p_nlf_vector.h`); `inv`, `invsqrt` and `sqrtf` are in the scalar
    header. The `x * sigmoid(2z)` form this used to take -- exact, since
    tanh(z) = 2*sigmoid(2z) - 1, which is why `C2` is *2*c -- therefore paid a
    scalar call per lane for its divide.

    Substituting the usual `x * sigmoid(1.702x)` fast GELU (which is what
    examples/gelu uses) is a different function and still ruled out: mlir-air's
    decode runs gelu_tanh in `glu.cc`, so an approximation here would make the
    prefill and the decode disagree about the model.

    `tl.extra.cuda.libdevice.tanh` is not CUDA here. It emits
    `tt.extern_elementwise` naming `__nv_tanhf`, triton-shared maps that symbol
    to `math.tanh`, and `@cast_bf16_only_ops` then puts it on the hardware op.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(G + offs[:])
    up = tl.load(U + offs[:])
    g = gate.to(tl.float32)
    y = (C2 * 0.5) * (g + K * g * g * g)
    gelu_gate = (0.5 * g * (1.0 + tl.extra.cuda.libdevice.tanh(y))).to(gate.dtype)
    tl.store(Y + offs[:], gelu_gate * up)


def triton_geglu(gate, up, block=SWIGLU_BLOCK):
    """gelu_tanh(gate) * up over matching [M, N] tensors -> [M, N] float32.

    The SwiGLU wrapper's chunking and padding, unchanged -- only the activation
    differs -- so see `glu_chunk` for how the dispatch is sized.
    """
    shape = gate.shape
    n = gate.numel()
    chunk = glu_chunk(n, shape[-1] if gate.ndim > 1 else block, block)
    npad = math.ceil(n / chunk) * chunk
    g = torch.nn.functional.pad(gate.reshape(-1).to(torch.bfloat16), (0, npad - n))
    u = torch.nn.functional.pad(up.reshape(-1).to(torch.bfloat16), (0, npad - n))
    g, u = g.contiguous(), u.contiguous()
    y = torch.empty(npad, dtype=torch.bfloat16)
    for o in range(0, npad, chunk):
        launch(
            _geglu_kernel,
            (chunk // block,),
            g[o : o + chunk],
            u[o : o + chunk],
            y[o : o + chunk],
            transform_script=script("swiglu/transform_aie2p.mlir"),
            C2=_GELU_2C,
            K=_GELU_K,
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
    DIM: tl.constexpr,
):
    """y = x * rsqrt(mean(x^2) + eps) * w, per row.

    examples/weighted_rms_norm's kernel, BLOCK_M rows at a time so the row
    reduction is a vector op rather than a scalar chain -- but reducing in f32.
    That example rounds the squares to bf16 before summing and is only tested
    at N=256; at our N=2048 a bf16 sum of 2048 terms carries ~9% relative
    error, which is visible in the model output.

    `N` is the row stride and `DIM` the model dimension. They are equal for a
    power-of-two model dim and differ when the caller has zero-padded the rows
    up to one -- Llama-3.2-3B's 3072 is not a power of two, and `tl.arange`
    requires that the reduction width is. The padding contributes nothing to
    the sum of squares, so only the divisor has to know the difference.
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
    rstd = tl.math.rsqrt(sum_sq / DIM + eps)

    y = x_f32 * rstd[:, None] * w.to(tl.float32)[None, :]
    tl.store(Y + offsets, y.to(x.dtype))


RMS_BLOCK_M = 2


def triton_rms_norm(x, weight, eps, block_m=RMS_BLOCK_M):
    """x: [M, D] -> RMS-normalized and scaled by `weight` [D]."""
    M, dim = x.shape
    Mp = math.ceil(M / ROW_TILE) * ROW_TILE
    # The reduction width must be a power of two (tl.arange). Llama-3.2-1B's
    # 2048 already is, so it pads by nothing and its generated code is
    # unchanged; Llama-3.2-3B's 3072 is not. Zero columns add zero to the sum
    # of squares, and a zero weight leaves their output zero, so the padding is
    # invisible to the result -- only the mean's divisor has to ignore it.
    dim_p = 1 << (dim - 1).bit_length()
    wb = torch.nn.functional.pad(weight.to(torch.bfloat16), (0, dim_p - dim))
    wb = wb.contiguous()
    # Pad once; each dispatch gets a contiguous row slice, no copy back.
    xb = torch.nn.functional.pad(
        x.to(torch.bfloat16), (0, dim_p - dim, 0, Mp - M)
    ).contiguous()
    y = torch.empty_like(xb)
    # Fixed ROW_TILE-row chunks, so the grid is constant (ROW_TILE).
    for m0 in range(0, Mp, ROW_TILE):
        launch(
            _rms_norm_kernel,
            (ROW_TILE // block_m,),
            xb[m0 : m0 + ROW_TILE],
            wb,
            y[m0 : m0 + ROW_TILE],
            dim_p,
            float(eps),
            transform_script=script("llm_q4nx/transform_rms_norm_aie2p.mlir"),
            BLOCK_M=block_m,
            BLOCK_N=dim_p,
            DIM=dim,
        )
    return y[:M, :dim].to(torch.float32)
