# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The fused Q4NX decode on the HSA runtime.

mlir-air's decoder dispatches through pyxrt. This runs the same design on
HsaRuntime instead, which is what makes a fully-HSA chatbot possible.

The context length changes every token, and reaches the design as scratchpad
parameters of a full ELF: two scalars in device memory that the design reads in
its dispatch preamble, so one ELF serves every L.

ROCR loads the ELF from an hsaco and patches only PDI and argument addresses
into its control code, so the scratchpad's address is written in before
packing (`driver._pack_hsaco`). That has to be the NPU's address for it, so
this needs a ROCR whose `hsa_amd_pointer_info` reports an AIE allocation's
device address; `use_scratchpad` asks. The instruction-stream alternative --
patching the L-dependent words per token -- has no counterpart under hsaco
dispatch, where the stream belongs to a loaded executable.

Weights and the KV cache live in shared regions, so they are dispatched on in
place. Copying 0.7 GB of weights per token would cap throughput near 23 tok/s
before any compute.
"""

import ctypes
import os
import re
import tempfile
import threading
import weakref

import numpy as np


def _bfloat16():
    from ml_dtypes import bfloat16

    return bfloat16


#: aiecc names a full-ELF kernel "<device>:<sequence>". The shared engine's
#: builder names both, so this holds for every model it builds.
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


def use_scratchpad(engine="fused_decode"):
    """Whether the HSA decode can run: the scratchpad ELF is its only artifact.

    Decided by the loaded ROCR: the ELF needs `hsa_amd_pointer_info` to report
    the device address of an AIE allocation. Only the shared engine's decoder
    is built as an ELF; the PLE engine has none, so it cannot run on HSA.
    """
    if engine != "fused_decode":
        return False
    from triton.backends.amd_triton_npu.driver import load_hsa_runtime

    return bool(load_hsa_runtime().triton_npu_hsa_scratchpad_supported())


class HsaElfProgram:
    """A full-ELF kernel and the scratchpad the host writes before each dispatch.

    The scratchpad is allocated first, its device address written into the
    ELF, and the result packed into an hsaco and loaded, so each instance is
    its own program. Hold `lock` from writing the scratchpad through the
    dispatch that reads it, and `flush` in between.
    """

    def __init__(self, elf_path, kernel_name):
        from triton.backends.amd_triton_npu.driver import (
            _pack_hsaco,
            full_elf_scratchpad_size,
            load_hsa_runtime,
        )

        self.lib = load_hsa_runtime()
        size = full_elf_scratchpad_size(elf_path)
        if not size:
            raise HsaDecodeError(
                f"'{kernel_name}' declares no scratchpad parameters, so there "
                "is no way to give it a context length"
            )
        dev_addr = ctypes.c_uint64()
        buf, n = _errbuf()
        self.lib.triton_npu_hsa_scratchpad_alloc.restype = ctypes.c_void_p
        va = self.lib.triton_npu_hsa_scratchpad_alloc(
            ctypes.c_uint64(size), ctypes.byref(dev_addr), buf, n
        )
        if not va:
            raise HsaDecodeError(buf.value.decode())
        self._scratchpad = va
        # A view of device memory, not a copy; `flush` makes writes visible.
        self.params = np.ctypeslib.as_array(
            ctypes.cast(va, ctypes.POINTER(ctypes.c_uint32)), (size // 4,)
        )

        # Full ELF is aie2p only. The runtime caches programs by path, and this
        # hsaco is good only for this scratchpad, so the path names it: the
        # runtime never frees a scratchpad, so no other one has its address.
        hsaco = _pack_hsaco(
            "elf", {"elf_path": str(elf_path)}, "aie2p", kernel_name, dev_addr.value
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f"decode_{dev_addr.value:x}.hsaco")
            with open(path, "wb") as f:
                f.write(hsaco)
            buf, n = _errbuf()
            self.lib.triton_npu_hsa_prepare.restype = ctypes.c_void_p
            self.handle = self.lib.triton_npu_hsa_prepare(
                path.encode(), kernel_name.encode(), buf, n
            )
        if not self.handle:
            raise HsaDecodeError(buf.value.decode())
        self.lock = threading.Lock()

    def flush(self):
        """Make the host's writes to `params` visible to the next dispatch."""
        buf, n = _errbuf()
        rc = self.lib.triton_npu_hsa_scratchpad_flush(
            ctypes.c_void_p(self._scratchpad), buf, n
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


class Params:
    """Where the two parameters live in the scratchpad, and what to write there."""

    def __init__(self, append_slot, scale, addend, mask_slot):
        self.append_slot = append_slot
        self.scale = scale
        self.addend = addend
        self.mask_slot = mask_slot

    def write(self, params, L):
        """Write context length L into the scratchpad `params`."""
        # The KV append slot: an `addr`-kind parameter, a byte offset written
        # as is.
        params[self.append_slot] = np.uint32(
            (L * self.scale + self.addend) & 0xFFFFFFFF
        )
        # The attention mask threshold: a `core`-kind parameter, shifted left by
        # 2 because the firmware masks the low bits and the core shifts back.
        params[self.mask_slot] = np.uint32((L << 2) & 0xFFFFFFFF)


def parse_params(path):
    """Read an ELF's params.txt into a `Params`.

    The parameter names depend on the model: AIR names the KV append offset
    after the sequence argument it is affine in and puts the coefficients in
    the suffix, e.g. `..._x256_m256` for L*256 - 256. So parameters are found
    by their kind column (`addr` for the append offset, `core` for the mask),
    and the arithmetic is read from the name.
    """
    with open(path) as f:
        lines = [ln.split() for ln in f.read().split("\n") if ln.strip()]
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


def _elf_file(artifact_dir, name):
    path = os.path.join(artifact_dir, name)
    if not os.path.exists(path):
        raise HsaDecodeError(
            f"{path} is missing; build the decode with\n"
            "    make compile-decode RUNTIME=hsa"
        )
    return path


def elf_window(artifact_dir, max_L=None):
    """The ATTN_MAXL the scratchpad ELF was built for.

    The base decoder picks its window from whichever xclbin pairs are in the
    directory, so the ELF route hands it this one rather than letting it
    choose a pair the ELF was not built with.
    """
    with open(_elf_file(artifact_dir, "decode_scratchpad.maxl")) as f:
        maxl = int(f.read().split()[0])
    if max_L is not None and int(max_L) > maxl:
        raise HsaDecodeError(
            f"max_L={max_L} exceeds the ELF's ATTN_MAXL={maxl}; rebuild with "
            "make compile-decode RUNTIME=hsa DECODE_L=<L>"
        )
    return maxl


def _elf_program(artifact_dir, attn_maxl, region_w, model=None):
    """The scratchpad ELF and its parameter layout, checked against the decoder."""
    elf = _elf_file(artifact_dir, "decode_scratchpad.elf")
    params_txt = _elf_file(artifact_dir, "decode_scratchpad.params.txt")
    # The ELF keeps its own stamp: a later template build for another model
    # rewrites the shared one and leaves this ELF in place.
    if model is not None:
        with open(_elf_file(artifact_dir, "decode_scratchpad.model")) as f:
            built_for = f.read().strip()
        if built_for != model:
            raise HsaDecodeError(
                f"decode_scratchpad.elf was built for {built_for!r}, not "
                f"{model!r}; rebuild with make compile-decode RUNTIME=hsa"
            )
    built_maxl = elf_window(artifact_dir)
    if built_maxl != attn_maxl:
        raise HsaDecodeError(
            f"the ELF was built for ATTN_MAXL={built_maxl} but the decoder uses "
            f"{attn_maxl}; rebuild with make compile-decode RUNTIME=hsa"
        )
    params = parse_params(params_txt)
    # The append offset's scale is the region width. A mismatch means the ELF
    # and the decoder have different geometries, and every KV append would land
    # in the wrong place.
    if params.scale != region_w:
        raise HsaDecodeError(
            f"{os.path.basename(params_txt)} encodes scale {params.scale} but "
            f"the decoder's REGION_W is {region_w}; the ELF and the decoder "
            "disagree"
        )
    return params, HsaElfProgram(elf, ELF_KERNEL_NAME)


def require_xclbin_pair(artifact_dir):
    """Fail early if the xclbin pair mlir-air's decoder loads is missing.

    The HSA decoders subclass mlir-air's, and its `__init__` needs the pair
    even though HSA dispatches the ELF. Its own error does not say so.
    """
    names = os.listdir(artifact_dir) if os.path.isdir(artifact_dir) else ()
    built = {
        int(m.group(1))
        for m in (re.fullmatch(r"decode_L(\d+)\.xclbin", n) for n in names)
        if m and f"decode_L{m.group(1)}.insts.bin" in names
    }
    if not any(L - 1 in built for L in built):
        raise HsaDecodeError(
            f"{artifact_dir} has no decode_L<N>.xclbin + decode_L<N-1>.xclbin "
            "pair, which mlir-air's decoder loads even under HSA. Build it "
            "with\n"
            "    make compile-decode RUNTIME=hsa"
        )


def make_hsa_decoder_class(
    air, artifact_dir, decoder_class="FusedDecoder", engine="fused_decode", model=None
):
    """mlir-air's fused decoder with the dispatch moved onto HSA.

    Everything host-side stays theirs -- weight load, the region-major KV
    layout, `seed_kv`, sampling. Only `dispatch` changes: instead of handing
    XRT an instruction-stream BO as a kernel argument, it gives the program the
    context length in its scratchpad and enqueues the kernel's tensors.

    Their `__init__` still runs, so the XRT BOs it builds exist and go unused.
    That costs the weight allocation twice. Worth fixing before this is more
    than a demonstration; not worth forking their setup to avoid today.

    `decoder_class` names the base to wrap (3B/8B are not `FusedDecoder`).
    `engine` is the decode engine the model was built with. Only the shared
    engine is built as a scratchpad ELF; the per-layer embedding engine
    ("ple", Gemma4-E2B) is not, and so has no HSA decode.
    """
    if engine != "fused_decode":
        raise HsaDecodeError(
            f"the {engine!r} decode engine has no scratchpad ELF, which is the "
            "only decode HSA can dispatch; use the XRT runtime"
        )
    base = getattr(air, decoder_class)

    class HsaFusedDecoder(base):
        def _g(self, name):
            # Geometry is on the decoder (1B/4B) or its `fd` module (3B/8B).
            v = getattr(self, name, None)
            if v is not None:
                return v
            fd = getattr(self, "fd", None)
            if fd is not None:
                return getattr(fd, name)
            raise AttributeError(name)

        def __init__(self, *a, **kw):
            require_xclbin_pair(artifact_dir)
            if not use_scratchpad(engine):
                raise HsaDecodeError(
                    "this ROCR does not report a device address for an AIE "
                    "allocation, which the decode's scratchpad needs; use a "
                    "ROCR from utils/rocm-requirements.txt"
                )
            kw["max_L"] = elf_window(artifact_dir, kw.get("max_L"))
            super().__init__(*a, **kw)
            self._params, self._prog = _elf_program(
                artifact_dir, self.ATTN_MAXL, self._g("REGION_W"), model
            )
            # Host mirrors of the five kernel tensors. The XRT path keeps these
            # in BOs; HSA takes plain pointers and stages them itself.
            lib = self._prog.lib
            # Every tensor in a shared region: the weights because copying
            # 0.7 GB per token dominates everything else, the KV cache because
            # it is 67 MB and the kernel appends to it in place, and the small
            # three because once the big two are in place they are what is
            # left. Staging any of them costs more than the dispatch.
            self._bf16 = getattr(self, "bf16", None) or _bfloat16()
            self._x = shared_array(lib, (self.K,), self._bf16)
            self._build_w_buffer(lib)
            self._build_r_buffer(lib)
            self._y = shared_array(lib, (self.ny,), self._bf16)
            # From geometry, not self.KV: 3B/8B keep no host KV mirror.
            self._kv = shared_array(lib, (self._n_layers() * self.LREG,), self._bf16)
            # The two big ones are written here (and, for the KV cache, in
            # seed_kv) and belong to the device afterwards, so they carry their
            # own coherency rather than being flushed per token. The small three
            # are exchanged every dispatch and stay ordinary.
            mark_resident(lib, self._w)
            for b in self._w_tail:
                mark_resident(lib, b)
            mark_resident(lib, self._kv)
            print(
                f"[hsa-decode] ONE full ELF: ATTN_MAXL={self.ATTN_MAXL}, "
                f"L in {self._prog.params.size} scratchpad words",
                flush=True,
            )

        def seed_kv(self, *a, **kw):
            # Staged fallback (variadic: base arities differ); zero-copy uses seed_direct.
            super().seed_kv(*a, **kw)
            kv = getattr(self, "KV", None)
            if kv is not None:
                self._kv[:] = np.ascontiguousarray(kv).reshape(-1)
            else:
                # 3B/8B have no host mirror; read it back off the device BO.
                self._kv[:] = self._bo_words(self.kvc).view(self._bf16)
            mark_dirty(self._prog.lib, self._kv)  # resident: nothing else flushes it

        def _n_layers(self):
            for attr in ("UNI_DEC", "N_LAYERS"):
                n = getattr(self, attr, None)
                if n:
                    return int(n)
            fd = getattr(self, "fd", None)
            if fd is not None and getattr(fd, "UNI_DEC", None):
                return int(fd.UNI_DEC)
            return 16  # 1B hardcodes 16, defines no UNI_DEC

        def _rope_w(self):
            # The rope LUT covers the rotated head dims: ROPE_W where the
            # decoder rotates only part of the head, otherwise all of DH.
            return int(getattr(self, "ROPE_W", None) or self.DH)

        #: Base carries per-layer rope slabs (Qwen3/Gemma3), not the 1B's LUT.
        def _per_layer_rope(self):
            return hasattr(self, "_rope_base") and hasattr(self, "_rope_slab")

        def _bo_words(self, bo):
            # 3B/8B keep no host mirror; read the host-only BO back exactly.
            return np.frombuffer(bo.map(), dtype=np.int16).copy()

        def _build_w_buffer(self, lib):
            """Fill `_w` (+ `_w_tail` for the 8B's split weights) from the base."""
            self._w_tail = []
            if getattr(self, "_wsplit", False) and getattr(self, "w_bos", None):
                bufs = []
                for bo in self.w_bos:
                    w = self._bo_words(bo)
                    b = shared_array(lib, w.shape, w.dtype)
                    b[:] = w
                    bufs.append(b)
                self._w, self._w_tail = bufs[0], bufs[1:]
                return
            w16 = getattr(self, "Wv16", None)
            if w16 is not None:
                self._w = shared_array(lib, w16.shape, w16.dtype)
                self._w[:] = w16
                return
            # No host mirror: read it back from the single weights BO.
            w = self._bo_words(self.w_bo)
            self._w = shared_array(lib, w.shape, w.dtype)
            self._w[:] = w

        def _build_r_buffer(self, lib):
            """Fill `_r`: [rms_slabs | per-token rope gap | final_norm]."""
            if getattr(self, "rms_slabs", None) is None:
                r = self._bo_words(self.r_bo).view(self._bf16)
                self._r = shared_array(lib, r.shape, self._bf16)
                self._r[:] = r
                self._rms_lut_off = int(self.rms_lut_off)
                return
            if self._per_layer_rope():
                gap = self._n_layers() * self.ROPE_W_LEN
            else:
                gap = self._rope_w()
                self._rms_lut_off = int(self.rms_slabs.size)
            _rms = np.concatenate(
                [self.rms_slabs, np.zeros(gap, self._bf16), self.final_norm]
            )
            self._r = shared_array(lib, _rms.shape, _rms.dtype)
            self._r[:] = _rms

        def _write_rms_rope(self, p):
            """Write position p's rope into `_r` (the slabs are already there)."""
            if self._per_layer_rope():
                rope = self._rope_slab(p)
                off = self._rope_base
                self._r[off : off + rope.size] = rope
            else:
                w = self._rope_w()
                half = w // 2
                off = self._rms_lut_off
                self._r[off : off + half] = self.rope_cos[p][:half].astype(self._bf16)
                self._r[off + half : off + w] = self.rope_sin[p][:half].astype(
                    self._bf16
                )

        def kv_sink(self):
            """The resident KV buffer + geometry, for a producer to fill in place.

            K group g, pos p : L*lreg + g*region_stride + p*region_w
            V group g, pos p : L*lreg + (ngrp+g)*region_stride + p*region_w
            """
            region_w = self._g("REGION_W")
            return dict(
                buf=self._kv,
                n_layers=self._n_layers(),
                lreg=self.LREG,
                region_stride=self.cur_maxl * region_w,
                region_w=region_w,
                ngrp=self._g("NGRP"),
                dtype=self._bf16,
            )

        def seed_direct(self, P):
            """Mark KV a producer wrote via `kv_sink`; the scatter is already done."""
            self.current_seed_P = P
            mark_dirty(self._prog.lib, self._kv)

        def dispatch(self, tok, p):
            L = p + 1
            self._x[:] = np.asarray(self.embed[tok], self._bf16)
            self._write_rms_rope(p)
            with self._prog.lock:
                self._params.write(self._prog.params, L)
                self._prog.flush()
                # 8B split-weight tail follows the fixed five (base's w_bos[1:]
                # order).
                self._prog.dispatch(
                    [self._x, self._w, self._r, self._y, self._kv, *self._w_tail]
                )
            voc = self._y[self.decode_y : self.decode_y + self._g("UNI_LM") * self.VP]
            return voc[: self.VOCAB_SIZE].astype(np.float32)

    return HsaFusedDecoder
