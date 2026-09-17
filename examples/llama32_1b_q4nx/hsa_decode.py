# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The fused Q4NX decode on the HSA runtime.

mlir-air's decoder dispatches through pyxrt. This runs the same design on
HsaRuntime instead, which is what makes a fully-HSA chatbot possible.

The whole problem is the context length. It changes every token, and an AIE
dispatch has two ways to hear about it:

* the instruction stream, which encodes L in a handful of words. mlir-air's
  xclbin path rewrites exactly those words per token and passes the stream as a
  kernel argument. This file used to do the same, calibrating the L-dependent
  words from two builds at adjacent L and patching 248 of them per token.
* mlir-aie scratchpad parameters: two scalars in device memory that the design
  reads in its dispatch preamble. This is what it does now.

The second is better in every way that matters. There is no template pair to
build, no slope to calibrate, and nothing per-L to get wrong -- one ELF serves
every context length. It also removes the only reason HsaRuntime ever had an
entry point that writes arbitrary bytes into executable device memory.

It needs a ROCR that can resolve the device address of an application's buffer
-- one whose `hsa_amd_pointer_info` reports an AIE allocation's device address
in `agentBaseAddress`: a full-ELF design reaches its scratchpad through an
address patched into its control code, and that address is the one the NPU
sees, not the host address the allocation is known by. Without it the
dispatch does not fault -- the design waits on data that never arrives -- so
HsaRuntime refuses to prepare such a design and says so.

Weights and the KV cache live in shared regions, so they are dispatched on in
place. Copying 0.7 GB of weights per token would cap throughput near 23 tok/s
before any compute.
"""

import ctypes
import os
import re
import sys
import weakref

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

# aiecc names a full-ELF kernel "<device>:<sequence>". The decode's builder
# names both, so this is the same for every model sharing the fused engine.
ELF_KERNEL_NAME = "main:q4nx_decode"


class HsaDecodeError(RuntimeError):
    pass


def _errbuf(n=1024):
    return ctypes.create_string_buffer(n), ctypes.c_size_t(n)


class _SharedArray(np.ndarray):
    """An ndarray that can carry the finalizer freeing its shared region."""


def shared_array(lib, shape, dtype):
    """A numpy array over a shared region, so dispatch runs on it in place.

    `triton_npu_hsa_dispatch` stages every tensor host->device and back unless
    its pointer falls inside a region the runtime knows about. For the decode
    that distinction is the whole ballgame: the weights are ~0.7 GB and would
    otherwise be copied on every token.

    The region must outlive every dispatch naming it; the returned array keeps
    it alive and frees it when collected.
    """
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    buf, n = _errbuf()
    lib.triton_npu_hsa_shared_alloc.restype = ctypes.c_void_p
    va = lib.triton_npu_hsa_shared_alloc(ctypes.c_uint64(nbytes), buf, n)
    if not va:
        raise HsaDecodeError(buf.value.decode())
    arr = (
        np.ctypeslib.as_array(
            ctypes.cast(va, ctypes.POINTER(ctypes.c_uint8)), shape=(nbytes,)
        )
        .view(dtype)[: int(np.prod(shape))]
        .reshape(shape)
        .view(_SharedArray)
    )

    # Free on collection, and only then: a dispatch naming a freed region is a
    # use-after-free the runtime cannot see. A plain ndarray takes no
    # attributes, hence the subclass -- the finalizer has to ride with the
    # object whose lifetime it tracks.
    def _free(_va=va, _lib=lib):
        b, m = _errbuf()
        _lib.triton_npu_hsa_shared_free(ctypes.c_void_p(_va), b, m)

    arr._finalizer = weakref.finalize(arr, _free)
    return arr


def mark_dirty(lib, arr):
    """Say that the host has written `arr`, so the device gets to see it.

    Needed only for a resident region (below): the next dispatch declares the
    region in full again, and ROCR flushes it. An ordinary operand is declared
    in full every time and so needs nothing.
    """
    buf, n = _errbuf()
    rc = lib.triton_npu_hsa_shared_mark_dirty(ctypes.c_void_p(arr.ctypes.data), buf, n)
    if rc != 0:
        raise HsaDecodeError(buf.value.decode())


def mark_resident(lib, arr):
    """Stop re-flushing `arr` on every dispatch; sync it once, here.

    A dispatch declares each operand's size, and ROCR walks exactly those bytes
    with CLFLUSH before and after the submit -- ~18 us per declared MB. For the
    weights that is 0.7 GB re-flushed per token (~12 ms) to protect against host
    writes that never happen, and it is essentially the whole of the HSA-vs-XRT
    gap: XRT uploads a BO once and passes a handle.

    Only for a region written once and thereafter the device's. Anything the
    host writes between dispatches must call `mark_dirty` afterwards.
    """
    buf, n = _errbuf()
    rc = lib.triton_npu_hsa_shared_set_resident(
        ctypes.c_void_p(arr.ctypes.data), ctypes.c_int(1), buf, n
    )
    if rc != 0:
        raise HsaDecodeError(buf.value.decode())


def use_scratchpad():
    """Whether to take the scratchpad ELF or the patched instruction stream.

    The ELF is the better shape -- one artifact for every context length, and
    nothing writing into the instruction stream per token -- but it needs a
    ROCR that can resolve the device address of one of our buffers -- one whose
    `hsa_amd_pointer_info` answers for an AIE allocation rather than calling it
    HSA_EXT_POINTER_TYPE_UNKNOWN -- and no released one does. So this is a
    capability question, asked of the ROCR that is actually loaded, not a
    preference.

    AMD_TRITON_NPU_HSA_DECODE=elf|insts forces one, which is how the two are
    compared on a machine that could run either.
    """
    from triton.backends.amd_triton_npu.driver import load_hsa_runtime

    want = os.environ.get("AMD_TRITON_NPU_HSA_DECODE", "").lower()
    if want in ("elf", "insts"):
        return want == "elf"
    if want:
        raise HsaDecodeError(
            f"AMD_TRITON_NPU_HSA_DECODE={want!r}; expected 'elf' or 'insts'"
        )
    return bool(load_hsa_runtime().triton_npu_hsa_full_elf_supported())


class HsaProgram:
    """A prepared PDI + instruction stream, with the stream patchable per token."""

    def __init__(self, pdi_path, insts_path):
        from triton.backends.amd_triton_npu.driver import load_hsa_runtime

        self.lib = load_hsa_runtime()
        buf, n = _errbuf()
        self.lib.triton_npu_hsa_prepare.restype = ctypes.c_void_p
        self.handle = self.lib.triton_npu_hsa_prepare(
            pdi_path.encode(), insts_path.encode(), buf, n
        )
        if not self.handle:
            raise HsaDecodeError(buf.value.decode())
        self.insts_words = os.path.getsize(insts_path) // 4

    def patch_insts(self, word_offset, words):
        """Overwrite `words` (uint32) at `word_offset` in the live stream."""
        a = np.ascontiguousarray(words, dtype=np.uint32)
        buf, n = _errbuf()
        rc = self.lib.triton_npu_hsa_patch_insts(
            ctypes.c_void_p(self.handle),
            ctypes.c_uint64(word_offset * 4),
            a.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint64(a.nbytes),
            buf,
            n,
        )
        if rc != 0:
            raise HsaDecodeError(buf.value.decode())

    def dispatch(self, arrays):
        """One enqueue over `arrays` in kernel-argument order."""
        n = len(arrays)
        ptrs = (ctypes.c_void_p * n)()
        sizes = (ctypes.c_uint64 * n)()
        for i, a in enumerate(arrays):
            ptrs[i] = ctypes.c_void_p(a.ctypes.data)
            sizes[i] = ctypes.c_uint64(a.nbytes)
        buf, nb = _errbuf()
        rc = self.lib.triton_npu_hsa_dispatch(
            ctypes.c_void_p(self.handle), ctypes.c_uint32(n), ptrs, sizes, buf, nb
        )
        if rc != 0:
            raise HsaDecodeError(buf.value.decode())


class InstsForL:
    """The L-dependent words of the decode's instruction stream.

    Calibrated exactly as mlir-air's DecodeInstsGen does: two builds at the
    same ATTN_MAXL and adjacent L differ only in the words that encode L, and
    those move linearly, so a diff of the two gives base and slope.
    """

    def __init__(self, insts_lo, insts_hi, l_lo, l_hi):
        i1 = np.fromfile(insts_lo, dtype=np.uint32)
        i2 = np.fromfile(insts_hi, dtype=np.uint32)
        if i1.size != i2.size:
            raise HsaDecodeError("calibration streams differ in length")
        self.ld = np.where(i1 != i2)[0]
        if self.ld.size == 0:
            raise HsaDecodeError("the two calibration builds are identical")
        self.lo, self.hi = int(self.ld.min()), int(self.ld.max()) + 1
        self.base = i1[self.ld].astype(np.int64)
        self.slope = (i2[self.ld].astype(np.int64) - self.base) // (l_hi - l_lo)
        self.base_L = l_lo
        self.full = i1.copy()

    def slice_for(self, L):
        """The [lo:hi] window of the stream at context length L."""
        out = self.full[self.lo : self.hi].copy()
        vals = self.base + (L - self.base_L) * self.slope
        out[self.ld - self.lo] = vals.astype(np.uint32)
        return out


class HsaElfProgram:
    """A full-ELF kernel, with a scratchpad the host writes per dispatch.

    The other shape, above, puts the context length in the instruction stream
    and rewrites it every token. This one does not touch the stream at all: L
    is two scalars in device memory that the design reads in its dispatch
    preamble, so one artifact serves every context length.
    """

    def __init__(self, elf_path, kernel_name):
        from triton.backends.amd_triton_npu.driver import load_hsa_runtime

        self.lib = load_hsa_runtime()
        buf, n = _errbuf()
        self.lib.triton_npu_hsa_prepare_elf.restype = ctypes.c_void_p
        self.handle = self.lib.triton_npu_hsa_prepare_elf(
            str(elf_path).encode(), kernel_name.encode(), buf, n
        )
        if not self.handle:
            raise HsaDecodeError(buf.value.decode())

        addr = ctypes.c_void_p()
        size = ctypes.c_uint64()
        buf, n = _errbuf()
        rc = self.lib.triton_npu_hsa_scratchpad(
            ctypes.c_void_p(self.handle),
            ctypes.byref(addr),
            ctypes.byref(size),
            buf,
            n,
        )
        if rc != 0:
            raise HsaDecodeError(buf.value.decode())
        if not addr.value or not size.value:
            raise HsaDecodeError(
                f"'{kernel_name}' declares no scratchpad parameters, so there "
                "is no way to give it a context length"
            )
        # A view, not a copy: writing an element writes device memory. The
        # runtime declares this buffer at its real size on every dispatch, so
        # what is written here is flushed before the device reads it.
        self.params = np.ctypeslib.as_array(
            ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint32)), (size.value // 4,)
        )

    def dispatch(self, arrays):
        """One enqueue over `arrays` in kernel-argument order."""
        n = len(arrays)
        ptrs = (ctypes.c_void_p * n)()
        sizes = (ctypes.c_uint64 * n)()
        for i, a in enumerate(arrays):
            ptrs[i] = ctypes.c_void_p(a.ctypes.data)
            sizes[i] = ctypes.c_uint64(a.nbytes)
        buf, nb = _errbuf()
        rc = self.lib.triton_npu_hsa_dispatch(
            ctypes.c_void_p(self.handle), ctypes.c_uint32(n), ptrs, sizes, buf, nb
        )
        if rc != 0:
            raise HsaDecodeError(buf.value.decode())


def parse_params(path):
    """Read params.txt into (append_name, scale, addend, mask_name).

    The names are not fixed across models. AIR names a BD-offset parameter
    after the sequence argument it is affine in and puts the coefficients in
    the suffix, so llama's REGION_W=256 gives `..._x256_m256` (L*256 - 256)
    while gemma's 512 gives `..._x512_m512`. Hardcoding either writes the wrong
    KV address for the other, silently. Classify by the kind column instead --
    `addr` is the BD offset, `core` the herd RTP -- and take the arithmetic
    from the suffix rather than assuming it.

    Mirrors mlir-air's own reader in fused_decode/decode_elf.py.
    """
    lines = [ln.split() for ln in open(path).read().split("\n") if ln.strip()]
    entries = [ln for ln in lines[1:] if len(ln) >= 4]
    slot = {ln[0]: int(ln[1]) for ln in entries}
    kind = {ln[0]: ln[3] for ln in entries}
    addr = [n for n in slot if kind[n] == "addr"]
    core = [n for n in slot if kind[n] == "core"]
    if len(addr) != 1 or len(core) != 1:
        raise HsaDecodeError(
            f"{path}: expected exactly one 'addr' and one 'core' parameter, "
            f"got addr={addr} core={core}"
        )
    m = re.search(r"_argoff_(\d+)_x(-?\d+)_([mp])(\d+)$", addr[0])
    if not m:
        raise HsaDecodeError(f"{path}: cannot read affine coefficients from {addr[0]}")
    scale = int(m.group(2))
    addend = int(m.group(4)) * (-1 if m.group(3) == "m" else 1)
    return Params(slot[addr[0]], scale, addend, slot[core[0]])


class Params:
    """Where each parameter lives in the scratchpad, and what to write there."""

    def __init__(self, append_slot, scale, addend, mask_slot):
        self.append_slot = append_slot
        self.scale = scale
        self.addend = addend
        self.mask_slot = mask_slot

    def write(self, params, L):
        """Write context length L into the scratchpad `params`."""
        # The KV append slot: a byte offset, written raw because it is an
        # `addr`-kind parameter.
        params[self.append_slot] = np.uint32(
            (L * self.scale + self.addend) & 0xFFFFFFFF
        )
        # The attention mask threshold. A `core`-kind parameter is shifted left
        # by 2: the firmware's UPDATE_REG masks the low bits and the core
        # shifts back after reading. Values that fit in 30 bits survive, which
        # every context length does.
        params[self.mask_slot] = np.uint32((L << 2) & 0xFFFFFFFF)


def make_hsa_decoder_class(air, artifact_dir):
    """mlir-air's FusedDecoder with the dispatch moved onto HSA.

    Everything host-side stays theirs -- weight load, the region-major KV
    layout, `seed_kv`, sampling. Only `dispatch` changes: instead of handing
    XRT an instruction-stream BO as a kernel argument, it writes the context
    length to a scratchpad and enqueues the five tensors.

    Their `__init__` still runs, so the XRT BOs it builds exist and go unused.
    That costs the weight allocation twice. Worth fixing before this is more
    than a demonstration; not worth forking their setup to avoid today.
    """
    # ...but their __init__ has to be kept off *their* full-ELF path: we are
    # doing the ELF ourselves, and theirs would abort the process on a
    # duplicate LLVM option registration. config.select_decode_artifact()
    # settles that before their module is imported, which is where it has to
    # happen -- they read the selection at import time.

    class HsaFusedDecoder(air.FusedDecoder):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if use_scratchpad():
                self._init_elf()
            else:
                self._init_patched_stream()

            # Host mirrors of the five kernel tensors. The XRT path keeps these
            # in BOs; HSA takes plain pointers and stages them itself.
            lib = self._prog.lib
            # Every tensor in a shared region: the weights because copying
            # 0.7 GB per token dominates everything else, the KV cache because
            # it is 67 MB and the kernel appends to it in place, and the small
            # three because once the big two are in place they are what is
            # left. Staging any of them costs more than the dispatch.
            self._x = shared_array(lib, (self.K,), self.bf16)
            self._w = shared_array(lib, self.Wv16.shape, self.Wv16.dtype)
            self._w[:] = self.Wv16
            _rms = np.concatenate(
                [self.rms_slabs, np.zeros(64, self.bf16), self.final_norm]
            )
            self._r = shared_array(lib, _rms.shape, _rms.dtype)
            self._r[:] = _rms
            self._rms_lut_off = int(self.rms_slabs.size)
            self._y = shared_array(lib, (self.ny,), self.bf16)
            self._kv = shared_array(lib, (16 * self.LREG,), self.bf16)
            # The two big ones are written here (and, for the KV cache, in
            # seed_kv) and belong to the device afterwards, so they carry their
            # own coherency rather than being flushed per token. The small three
            # are exchanged every dispatch and stay ordinary.
            mark_resident(lib, self._w)
            mark_resident(lib, self._kv)
            if self._gen is None:
                how = f"L via {self._prog.params.size} scratchpad parameters"
                what = "ONE full ELF"
            else:
                how = f"{self._gen.ld.size} L-dependent words patched per token"
                what = "ONE PDI + patched insts"
            print(
                f"[hsa-decode] {what}: ATTN_MAXL={self.ATTN_MAXL}, {how}",
                flush=True,
            )

        def _init_elf(self):
            """One ELF for every L, with the context length in a scratchpad."""
            elf = os.path.join(artifact_dir, "decode_scratchpad.elf")
            params_txt = os.path.join(artifact_dir, "decode_scratchpad.params.txt")
            for f in (elf, params_txt):
                if not os.path.exists(f):
                    raise HsaDecodeError(
                        f"{f} is missing; build the decode with\n"
                        "    python decode_build.py --format elf"
                    )
            self._params = parse_params(params_txt)
            # The scale IS the model's region width. If they disagree, the ELF
            # and this driver were built from different geometries and every KV
            # append would land in the wrong place -- which reads as bad
            # numerics, not as a mismatch, so check it here.
            if self._params.scale != self.REGION_W:
                raise HsaDecodeError(
                    f"{os.path.basename(params_txt)} encodes scale "
                    f"{self._params.scale} but the decoder's REGION_W is "
                    f"{self.REGION_W}; the ELF and the driver disagree"
                )
            self._prog = HsaElfProgram(elf, ELF_KERNEL_NAME)
            self._gen = None

        def _init_patched_stream(self):
            """A PDI and a template pair, with L patched into the stream."""
            pdi = os.path.join(artifact_dir, f"decode_L{self.ATTN_MAXL}.pdi")
            lo = os.path.join(artifact_dir, f"decode_L{self.ATTN_MAXL - 1}.insts.bin")
            hi = os.path.join(artifact_dir, f"decode_L{self.ATTN_MAXL}.insts.bin")
            for f in (pdi, lo, hi):
                if not os.path.exists(f):
                    raise HsaDecodeError(
                        f"{f} is missing; build the PDI templates with\n"
                        "    python decode_build.py --format pdi\n"
                        "(this ROCR cannot dispatch the scratchpad ELF -- see "
                        "use_scratchpad)"
                    )
            self._gen = InstsForL(lo, hi, self.ATTN_MAXL - 1, self.ATTN_MAXL)
            self._prog = HsaProgram(pdi, hi)

        def seed_kv(self, fk, fv, P):
            super().seed_kv(fk, fv, P)
            self._kv[:] = np.ascontiguousarray(self.KV).reshape(-1)
            # The KV cache is resident, so the dispatch that follows would
            # declare a token span and flush nothing. Nothing would fail loudly:
            # the device would read whatever part of the prompt's K/V had
            # already been evicted.
            mark_dirty(self._prog.lib, self._kv)

        def dispatch(self, tok, p):
            L = p + 1
            if self._gen is None:
                # The context length, as two scalars in device memory. The
                # design reads them in its dispatch preamble; nothing rewrites
                # the instruction stream.
                self._params.write(self._prog.params, L)
            else:
                # The context length, into the stream the packet already points
                # at.
                self._prog.patch_insts(self._gen.lo, self._gen.slice_for(L))
            self._x[:] = np.asarray(self.embed[tok], self.bf16)
            self._r[self._rms_lut_off : self._rms_lut_off + 32] = self.rope_cos[p][
                :32
            ].astype(self.bf16)
            self._r[self._rms_lut_off + 32 : self._rms_lut_off + 64] = self.rope_sin[p][
                :32
            ].astype(self.bf16)
            self._prog.dispatch([self._x, self._w, self._r, self._y, self._kv])
            voc = self._y[self.decode_y : self.decode_y + self.UNI_LM * self.VP]
            return voc[: self.VOCAB_SIZE].astype(np.float32)

    return HsaFusedDecoder
