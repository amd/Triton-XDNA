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
import subprocess
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
#: SLOWER than letting the herd shrink. On one wide-layer `gate`, a 256-row
#: block runs 79.6 ms at `l1_m=32` (eight columns) against 73.9 at `l1_m=64`
#: (four), and a 128-row block 128.1 at 16 against 124.1 at 64. Below 512 rows
#: there are not enough of them to give every column a tile worth having.
_MATMUL_SCRIPTS = {
    (64, 64): "gpt2/transform_matmul_aie2p.mlir",
    (128, 64): "llm_q4nx/transform_matmul_m128_aie2p.mlir",
    (64, 128): "llm_q4nx/transform_matmul_n128_aie2p.mlir",
}

#: The K depth every checked-in schedule was generated for, and the tile
#: `mm_object` has to bake into the microkernel to match one.
L2_K = 64

#: mlir-air's hand-tuned GEMM microkernel. Compiled per core tile and linked
#: into the compute herd instead of generating its inner loop, which at the
#: shipped 512x512 block is 1.57x on a wide `gate` -- the generated loop issues
#: about a third of the MACs per instruction bundle that this does. The two
#: agree bit for bit.
#:
#: Vendored by `utils/fetch_mlir_air_src.py`; absent in a source tree that has
#: never fetched it, which is why every caller tolerates None.
_MM_SRC = os.path.join(
    os.path.dirname(_EXAMPLES),
    "third_party/mlir-air-src/programming_examples",
    "matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc",
)

#: Peano flags, as mlir-air's own `compile_gemm_mm` passes them. BFP16
#: emulation stays on for the reason it is on there: it is the validated aie2p
#: path, and the native bf16 branch of that file lays `C_block` out differently
#: and gives wrong results at these tiles.
_MM_FLAGS = [
    "-O2",
    "-std=c++20",
    "--target=aie2p-none-unknown-elf",
    "-DNDEBUG",
    "-D__AIE_API_AIE_ADF_HPP__",
    "-DBIT_WIDTH=8",
    "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
    "-Wno-parentheses",
    "-Wno-attributes",
    "-Wno-macro-redefined",
    "-Wno-empty-body",
]


def _cache_dir():
    """Where generated schedules and microkernel objects live.

    Beside the AIR project, so everything a compile produced is in one place,
    and absolute: aircc resolves a relative `link_with` against its own working
    directory, which is not this process's.
    """
    from triton.backends.amd_triton_npu.config import npu_config

    d = os.path.abspath(os.path.join(npu_config.air_project_path, "microkernels"))
    os.makedirs(d, exist_ok=True)
    return d


#: What `mm_aie2p.cc` calls its entry point before `SYM_SUFFIX` is pasted on.
_MM_SYM = "op_has_no_registered_library_name"


def mm_symbol(l1_m, l1_n, l2_k=L2_K):
    """The entry point `mm_object` builds for this tile.

    One name per tile, because a chain can hold GEMMs at more than one of them
    and they are stitched into a single module: two tiles both exporting the
    bare name collide there ("redefinition of symbol named ...") and, if they
    did not, the link would silently take one object for both. mlir-air's
    source anticipates this -- `SYM_SUFFIX` exists in `mm_aie2p.cc` for it.
    """
    return f"{_MM_SYM}_{l1_m}x{l1_n}x{l2_k}"


def mm_object(l1_m, l1_n, l2_k=L2_K):
    """The microkernel compiled for this core tile, or None if it cannot be.

    `DIM_M`/`DIM_N`/`DIM_K` are compile-time constants in `mm_aie2p.cc`, so the
    object is only valid for the tile it was built for and each one gets its
    own file. Built once and cached on disk.

    None rather than an exception on every way this can be unavailable -- no
    vendored source, no Peano, not npu2, a compiler error -- because the
    generated inner loop is a correct fallback for all of them, and a prefill
    that runs slower is a better outcome than one that does not run.
    """
    from triton.backends.amd_triton_npu.driver import (
        detect_npu_version,
        find_peano_root,
    )

    # `--target=aie2p` and `mm_aie2p.cc` are both npu2-only; npu1 wants the
    # other source and the other triple. `npu_config.target` is None until
    # something resolves it, so ask the resolver rather than the setting.
    if not os.path.exists(_MM_SRC) or detect_npu_version() != "npu2":
        return None
    out = os.path.join(_cache_dir(), f"mmk_{l1_m}x{l1_n}x{l2_k}.o")
    if os.path.exists(out):
        return out
    peano = find_peano_root()
    aie = os.environ.get("MLIR_AIE_INSTALL_DIR", "")
    if not peano or not aie:
        return None
    cmd = (
        [os.path.join(peano, "bin", "clang++")]
        + _MM_FLAGS
        + [
            f"-I{os.path.join(aie, 'include')}",
            f"-DDIM_M={l1_m}",
            f"-DDIM_N={l1_n}",
            f"-DDIM_K={l2_k}",
            f"-DDIM_M_DIV_4={l1_m // 4}",
            f"-DDIM_N_DIV_4={l1_n // 4}",
            f"-DDIM_M_DIV_8={l1_m // 8}",
            f"-DDIM_N_DIV_8={l1_n // 8}",
            f"-DSYM_SUFFIX={mm_symbol(l1_m, l1_n, l2_k)[len(_MM_SYM):]}",
            "-c",
            _MM_SRC,
            "-o",
            out,
        ]
    )
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        if os.environ.get("AMD_TRITON_NPU_DEBUG"):
            print(
                f"[kernels] no microkernel for {l1_m}x{l1_n}x{l2_k}: "
                f"{r.stderr.strip().splitlines()[-1:]}",
                flush=True,
            )
        return None
    return out


#: `triton_matmul`'s default: pick the schedule from the block shape. Distinct
#: from `None`, which means "let the driver generate one" and is what
#: `qwen25_prefill` passes for the shape these schedules reject.
BY_BLOCK = "<chosen from the block shape>"


def matmul_script(block_m, block_n):
    """The schedule to lower a `block_m x block_n` GEMM tile with.

    The core tile is the largest that still spreads the block over the whole
    array, floored at `L1` -- see `_MATMUL_SCRIPTS` for why the floor is there
    and not lower.

    Where `mm_object` can build the microkernel for that tile, the schedule is
    generated to call it instead of generating the inner loop. The checked-in
    scripts are the same generator's output without it, so the two differ in
    the three phases that exist to feed aievec and nowhere else.
    """
    l1 = (max(L1, block_m // AIE_COLS), max(L1, block_n // AIE_ROWS))
    if l1 not in _MATMUL_SCRIPTS:
        raise ValueError(
            f"a {block_m}x{block_n} block wants an l1 tile of {l1[0]}x{l1[1]}, "
            f"which has no schedule; blocks are bounded by MAX_BLOCK_NUMEL "
            f"({MAX_BLOCK_NUMEL}) and this one is {block_m * block_n}"
        )
    obj = mm_object(*l1)
    return _library_call_script(*l1, obj) if obj else script(_MATMUL_SCRIPTS[l1])


#: The core tile the drain-fused epilogue runs at. The drain holds the f32
#: accumulator, A, B and a bf16 output tile at once --
#: `l1_m*l1_n*4 + l1_m*L2_K*2 + L2_K*l1_n*2 + l1_m*l1_n*2` -- so the plain
#: 64x128 tile's 72 KiB does not fit in 64. Which axis gives way is a choice,
#: and it is not free either way: measured on the gate's own shape against the
#: 512x512 block, halving the columns costs 1.33x and halving the rows 1.14x.
#: mlir-air halves the rows too -- its drain method forces tile_m=32 where its
#: fused-cast uses 64 -- and this is 44 KiB.
FUSED_L1 = (32, 128)

#: ...and a narrower K tile than the plain schedules use. A and B are both
#: ping-ponged, so at 64 they cost 40 KiB of the 64 between them and the tile
#: lands on exactly 64.00 with the stack still to place. Halving K halves both:
#: 16 accumulator + 8 drain + 4 A + 16 B = 44.
#:
#: This is one knob where mlir-air has two. Its gemma4 spends 32 on the L1 tile
#: and 64-256 on the L3->L2 one, so it pays the small tile's L1 price without
#: the transfer count; `generate_matmul_transform` cannot separate them today.
FUSED_L2_K = 32

#: ...and the block that tile covers the array as. A GEMM whose block is not
#: exactly this cannot take the fused path.
FUSED_BLOCK = (FUSED_L1[0] * AIE_COLS, FUSED_L1[1] * AIE_ROWS)

#: Opt-in, because the fused schedule needs two fixes that are not upstream
#: yet: AIE2P's aievec rejects the n-D native-lane-count vectors a packed
#: accumulator produces, and `air-shrink-memref-sizes-by-access` mis-sizes an
#: accumulator whose only users are vector accesses. Against a released
#: MLIR-AIE/AIR the fused gate GEMM does not compile at all, so this defaults
#: off and every gate keeps running the unfused chain.
FUSED_EPILOGUE = os.environ.get("LLM_Q4NX_FUSED_EPILOGUE") == "1"


def fused_epilogue_script(block_m, block_n):
    """The drain-fused matmul schedule for this block, or None if there is none.

    None rather than an exception: a caller falls back to running the
    activation as its own pass, which is correct, just slower.
    """
    if not FUSED_EPILOGUE or (block_m, block_n) != FUSED_BLOCK:
        return None
    from triton.backends.amd_triton_npu.matmul_transform import (
        generate_matmul_transform,
    )

    l1_m, l1_n = FUSED_L1
    # The two are independent -- the microkernel is the compute herd's inner
    # loop and the activation is the epilogue herd's -- and taking the
    # activation without it would be a bad trade: mlir-air's mm_aie2p.cc is
    # worth 1.57x on the GEMM, which is more than moving the activation saves.
    obj = mm_object(l1_m, l1_n, FUSED_L2_K)
    tag = "mmo" if obj else "vec"
    path = os.path.join(
        _cache_dir(),
        f"transform_mm_gelu_{tag}_{l1_m}x{l1_n}x{FUSED_L2_K}.mlir",
    )
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(
                generate_matmul_transform(
                    l1_m=l1_m,
                    l1_n=l1_n,
                    l2_k=FUSED_L2_K,
                    library_call=obj,
                    library_call_symbol=mm_symbol(l1_m, l1_n, FUSED_L2_K),
                    contract_input_type=None if obj else "bf16",
                    fused_epilogue=True,
                )
            )
    return path


def _library_call_script(l1_m, l1_n, obj):
    """Generate, once, the schedule that links this tile against `obj`.

    Written beside the object rather than into `examples/`: it names an
    absolute path that only means anything on the machine that built it.
    """
    from triton.backends.amd_triton_npu.matmul_transform import (
        generate_matmul_transform,
    )

    path = os.path.join(_cache_dir(), f"transform_mm_{l1_m}x{l1_n}x{L2_K}.mlir")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(
                generate_matmul_transform(
                    l1_m=l1_m,
                    l1_n=l1_n,
                    l2_k=L2_K,
                    library_call=obj,
                    library_call_symbol=mm_symbol(l1_m, l1_n),
                )
            )
    return path


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


def resident_weight(w, block_n=None):
    """Pad `w` for the NPU now, returning a handle that does not reference it."""
    K, N = w.shape
    Kp = _pow2(K)
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


def triton_matmul(x, w, block_m=None, block_n=None, transform_script=BY_BLOCK):
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
    """
    M, K = x.shape
    Kw, N = w.shape
    assert K == Kw, (x.shape, w.shape)
    Kp = _pow2(K)  # tl.arange needs a power of two
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
    a = shared_empty((Mp, a_stride), torch.bfloat16)
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
#: sqrt(2/pi), doubled. See `_geglu_kernel` for why it is doubled.
_GELU_2C = 1.5957691216057308
_GELU_K = 0.044715


@triton.jit
def _geglu_kernel(G, U, Y, C2: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    """gelu_tanh(gate) * up -- Gemma3's GLU, where Llama and Qwen3 use SiLU.

    The activation is `gelu_pytorch_tanh`, exactly:

        0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3))),  c = sqrt(2/pi)

    written here with sigmoid instead of tanh. `tl.math.tanh` does not exist in
    this Triton version, and substituting the usual `x * sigmoid(1.702x)` fast
    GELU (which is what examples/gelu uses) would be a different function --
    mlir-air's decode runs gelu_tanh in `glu.cc`, so an approximation here would
    make the prefill and the decode disagree about the model.

    No approximation is needed, because tanh(z) = 2*sigmoid(2z) - 1 turns the
    expression above into an exact identity:

        0.5 * x * (1 + 2*sigmoid(2z) - 1)  =  x * sigmoid(2z)

    with z = c * (x + 0.044715 x^3). That is why `C2` is *2*c and not c.

    The op set is the SwiGLU kernel's -- mul, add, sigmoid -- so it lowers under
    the same transform script, which is what makes this a new kernel rather than
    a new schedule. f32 for the polynomial and the sigmoid, as tl.sigmoid
    requires, rounded back to bf16 before the multiply by `up`.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(G + offs[:])
    up = tl.load(U + offs[:])
    g = gate.to(tl.float32)
    z = C2 * (g + K * g * g * g)
    gelu_gate = (g * tl.sigmoid(z)).to(gate.dtype)
    tl.store(Y + offs[:], gelu_gate * up)


def triton_geglu(gate, up, block=SWIGLU_BLOCK):
    """gelu_tanh(gate) * up over matching [M, N] tensors -> [M, N] float32.

    The SwiGLU wrapper's chunking and padding, unchanged -- only the activation
    differs -- so see `triton_swiglu` for why the chunk size is fixed.
    """
    shape = gate.shape
    chunk = ROW_TILE * (shape[-1] if gate.ndim > 1 else block)
    chunk = math.ceil(chunk / block) * block
    n = gate.numel()
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
