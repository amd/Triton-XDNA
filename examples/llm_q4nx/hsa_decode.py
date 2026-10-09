# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The fused Q4NX decode on the HSA runtime.

mlir-air's decoder dispatches through pyxrt. This runs the same design on
HsaRuntime instead, which is what makes a fully-HSA chatbot possible.

The whole problem is the context length. It changes every token, and an AIE
dispatch has only two ways to hear about it:

* the instruction stream, which encodes L in a handful of words. mlir-air's
  xclbin path rewrites exactly those words per token and passes the stream as a
  kernel argument.
* mlir-aie scratchpad parameters, which is what the full-ELF path uses --
  `xrt::run::get_ctrl_scratchpad_bo`, an XRT API with no HSA counterpart
  (`grep -ic scratch hsa_ext_amd_aie.h` is 0).

So on HSA it has to be the first, and the AIE dispatch packet does carry
`insts_addr_low/high` and `insts_size` per enqueue -- the stream is an input to
each dispatch, not a property of the program. What blocked it was only that
`triton_npu_hsa_prepare` read the stream from a file once. `patch_insts` (added
to HsaRuntime for this) writes into the very buffer the packet already points
at, so the 248 L-dependent words are rewritten per token exactly as the xclbin
path rewrites them.

The design itself is L-independent: the L=2048 and L=2047 builds produce a
BYTE-IDENTICAL PDI and differ only in those insts words. One PDI serves every
context length.

Weights and the KV cache live in shared regions, so they are dispatched on in
place. Copying 0.7 GB of weights per token would cap throughput near 23 tok/s
before any compute.
"""

import ctypes
import os
import weakref

import numpy as np


def _bfloat16():
    from ml_dtypes import bfloat16

    return bfloat16


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


def _programs(artifact_dir, attn_maxl):
    """The PDI and the L-patching generator for a template built at `attn_maxl`."""
    pdi = os.path.join(artifact_dir, f"decode_L{attn_maxl}.pdi")
    lo = os.path.join(artifact_dir, f"decode_L{attn_maxl - 1}.insts.bin")
    hi = os.path.join(artifact_dir, f"decode_L{attn_maxl}.insts.bin")
    for f in (pdi, lo, hi):
        if not os.path.exists(f):
            raise HsaDecodeError(
                f"{f} is missing; build the PDI templates with\n"
                "    make compile-decode RUNTIME=hsa"
            )
    return InstsForL(lo, hi, attn_maxl - 1, attn_maxl), HsaProgram(pdi, hi)


def make_hsa_decoder_class(
    air, artifact_dir, decoder_class="FusedDecoder", engine="fused_decode"
):
    """mlir-air's fused decoder with the dispatch moved onto HSA.

    Everything host-side stays theirs -- weight load, the region-major KV
    layout, `seed_kv`, sampling. Only `dispatch` changes: instead of handing
    XRT an instruction-stream BO as a kernel argument, it patches the stream
    the HSA program already owns and enqueues the kernel's tensors.

    Their `__init__` still runs, so the XRT BOs it builds exist and go unused.
    That costs the weight allocation twice. Worth fixing before this is more
    than a demonstration; not worth forking their setup to avoid today.

    `decoder_class` names the base to wrap (3B/8B are not `FusedDecoder`).
    `engine` is the decode engine the model was built with; the per-layer
    embedding engine ("ple", Gemma4-E2B) has its own operand set.
    """
    base = getattr(air, decoder_class)
    if engine == "ple":
        return _make_hsa_ple_decoder_class(base, artifact_dir)

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
            super().__init__(*a, **kw)
            self._gen, self._prog = _programs(artifact_dir, self.ATTN_MAXL)
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
                f"[hsa-decode] ONE PDI + patched insts: ATTN_MAXL={self.ATTN_MAXL}, "
                f"{self._gen.ld.size} L-dependent words",
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
                gap = self.DH
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
                half = self.DH // 2
                off = self._rms_lut_off
                self._r[off : off + half] = self.rope_cos[p][:half].astype(self._bf16)
                self._r[off + half : off + self.DH] = self.rope_sin[p][:half].astype(
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
            # The context length, into the stream the packet already points at.
            self._prog.patch_insts(self._gen.lo, self._gen.slice_for(L))
            self._x[:] = np.asarray(self.embed[tok], self._bf16)
            self._write_rms_rope(p)
            # 8B split-weight tail follows the fixed five (base's w_bos[1:] order).
            self._prog.dispatch(
                [self._x, self._w, self._r, self._y, self._kv, *self._w_tail]
            )
            voc = self._y[self.decode_y : self.decode_y + self._g("UNI_LM") * self.VP]
            return voc[: self.VOCAB_SIZE].astype(np.float32)

    return HsaFusedDecoder


def _make_hsa_ple_decoder_class(base, artifact_dir):
    """The per-layer-embedding decoder (Gemma4-E2B) on HSA.

    Its kernel takes seven tensors: x, the weights, the norm/rope buffer, y,
    the KV cache, the PLE slab and px (the token embedding again, which the PLE
    branch reads). Per token the host writes x and px, this token's embedding
    rows into the PLE slab, and the position's rope into the norm buffer. The
    logits come back softcapped, as the base's do.
    """

    class HsaPleDecoder(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._gen, self._prog = _programs(artifact_dir, self.ATTN_MAXL)
            lib = self._prog.lib
            bf16 = self.bf16

            def copy_of(arr):
                out = shared_array(lib, arr.shape, arr.dtype)
                out[:] = arr
                return out

            self._x = shared_array(lib, (self.K,), bf16)
            self._px = shared_array(lib, (self.K,), bf16)
            self._w = copy_of(np.frombuffer(self.w_bo.map(), dtype=np.int16))
            norms = np.concatenate(
                [
                    self.rms_slabs,
                    np.zeros(self._RMS_SIZE - self._rope_base - self.K, bf16),
                    self.final_norm,
                ]
            )
            self._r = copy_of(norms)
            self._y = shared_array(lib, (self.ny,), bf16)
            self._kv = shared_array(lib, (self.UNI * self.KV_LAYER,), bf16)
            self._kv[:] = 0
            self._pw = copy_of(self.ple.reshape(-1))
            # The weights and the KV cache are the device's once written. The
            # PLE slab is too, apart from the embedding rows patched per token,
            # which `dispatch` marks dirty.
            for buf in (self._w, self._kv, self._pw):
                mark_resident(lib, buf)
            # The XRT copies of the two big buffers are not read again.
            self.w_bo = None
            self.pw_bo = None
            print(
                f"[hsa-decode] gemma4 PLE: ONE PDI + patched insts, "
                f"ATTN_MAXL={self.ATTN_MAXL}, {self._gen.ld.size} L-dependent words",
                flush=True,
            )

        def seed_kv(self, ks, vs, P):
            super().seed_kv(ks, vs, P)
            self._kv[:] = np.ascontiguousarray(self.KV).reshape(-1)
            mark_dirty(self._prog.lib, self._kv)

        def kv_sink(self):
            """The resident KV buffer and its geometry, for a producer to fill."""
            return dict(
                buf=self._kv,
                n_layers=self.UNI,
                lreg=self.KV_LAYER,
                region_stride=self.ATTN_MAXL * self.REGION_W,
                region_w=self.REGION_W,
                ngrp=self.NGRP,
                dtype=self.bf16,
            )

        def seed_direct(self, P):
            """Mark the KV a producer wrote through `kv_sink`."""
            self.current_seed_P = P
            mark_dirty(self._prog.lib, self._kv)

        def dispatch(self, tok, p):
            L = p + 1
            if not (1 <= L <= self.maxL):
                raise ValueError(f"position {p} is outside [0,{self.maxL - 1}]")
            self._prog.patch_insts(self._gen.lo, self._gen.slice_for(L))
            x0 = np.asarray(
                self.qm.embed_rows("model.embed_tokens.weight", [tok])[0], self.bf16
            )
            self._x[:] = x0
            self._px[:] = x0
            emb = self._ple_embed(tok)
            for i in range(self.UNI):
                off = i * self.PLE_LAYER + self.PLE_EMB_OFF
                self._pw[off : off + emb[i].size] = emb[i]
            mark_dirty(self._prog.lib, self._pw)
            rope = self._rope_slab(p)
            self._r[self._rope_base : self._rope_base + rope.size] = rope
            self._prog.dispatch(
                [self._x, self._w, self._r, self._y, self._kv, self._pw, self._px]
            )
            n = self.UNI_LM * self.VP
            yv = self._y[self.decode_y : self.decode_y + n].astype(np.float32)
            yv = yv[: self.VOCAB_SIZE]
            cap = self.gw.FINAL_LOGIT_SOFTCAP
            if cap:
                yv /= cap
                np.tanh(yv, out=yv)
                yv *= cap
            return yv

    return HsaPleDecoder
