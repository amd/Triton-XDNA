#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Superkernels: several Triton kernels in one dispatch that loops over waves.

Each case traces a wave of one-core elementwise kernels, checks the AIR
module it lowers to, then compiles it and runs it on the NPU against numpy:

  chain   a call writes an on-chip tensor that the next call reads, and a
          DDR buffer is updated in place every wave
  phases  two calls of one kernel on one herd become a phase loop, each
          phase writing a different on-chip tensor
  table   a per-wave `Table` argument and a host-set `Scalar`

Also checks the errors tracing and lowering report. Needs an npu2 device for the runs;
with no NPU only the IR checks run.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile

D, DP = 1536, 2048  # vector length, and the power of two it is padded to
WAVES = 4

# The vector-add schedule with the whole vector on one core.
ELEMENTWISE = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %flat = transform.structured.flatten_elementwise %op
        : (!transform.any_op) -> !transform.any_op
    %res, %new = transform.structured.bufferize_to_allocation %flat
        <{memory_space = 1, bufferize_destination_only, emit_dealloc}>
        : !transform.any_op
    %op1 = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %tiled, %forall = transform.structured.tile_using_forall %op1
        num_threads [1] (mapping = [#gpu.thread<x>])
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @pad_and_promote_binary_bf16 failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @one_shot_bufferize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @post_bufferize_cleanup failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @vectorize_generics_at_32 failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    %vh = transform.include @air_herd_mapping_and_vectorize
        failures(propagate) (%arg1) : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
"""

AIRCC_ARGS = ("--air-channel-multiplexing=L1", "--omit-ping-pong-transform=L1")

failures = 0


def check(ok, what):
    global failures
    print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
    failures += not ok


def skip(msg):
    print(f"SKIP: {msg}")
    sys.exit(77)


def kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def scale(X, W, Y, row, D: tl.constexpr, DP: tl.constexpr):
        i = tl.arange(0, DP)
        m = i < D
        x = tl.load(X + i, mask=m)
        tl.store(Y + i, x * tl.load(W + row * D + i, mask=m), mask=m)

    @triton.jit
    def add(X, Y, Z, D: tl.constexpr, DP: tl.constexpr):
        i = tl.arange(0, DP)
        m = i < D
        tl.store(Z + i, tl.load(X + i, mask=m) + tl.load(Y + i, mask=m), mask=m)

    @triton.jit
    def sub(X, Y, Z, D: tl.constexpr, DP: tl.constexpr, SWAP: tl.constexpr):
        i = tl.arange(0, DP)
        m = i < D
        a, b = tl.load(X + i, mask=m), tl.load(Y + i, mask=m)
        tl.store(Z + i, b - a if SWAP else a - b, mask=m)

    return scale, add, sub


def bf(a):
    """Round float32 values to bfloat16, as float32."""
    import numpy as np

    u = np.asarray(a, np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def chain(sk, scale, add, script):
    import torch
    from triton.backends.amd_triton_npu.superkernel import WAVE

    x = sk.ddr("x", D, torch.bfloat16)
    w = sk.ddr("w", WAVES * D, torch.bfloat16)
    h = sk.onchip("h", D, torch.bfloat16)
    sk.schedule(scale, script)
    sk.schedule(add, script)
    with sk.wave():
        with sk.cores((2, 2), "scale"):
            scale[(1,)](x, w, h, WAVE, D, DP)
        with sk.cores((3, 2), "add"):
            add[(1,)](x, h, x, D, DP)

    def ref(x, b, rt):
        for l in range(WAVES):
            x = bf(x + bf(x * b["w"][l]))
        return x

    return ref


def phases(sk, scale, add, script):
    import torch
    from triton.backends.amd_triton_npu.superkernel import WAVE

    x = sk.ddr("x", D, torch.bfloat16)
    w = sk.ddr("w", 2 * WAVES * D, torch.bfloat16)
    h0 = sk.onchip("h0", D, torch.bfloat16)
    h1 = sk.onchip("h1", D, torch.bfloat16)
    sk.schedule(scale, script)
    sk.schedule(add, script)
    with sk.wave():
        with sk.cores((2, 2), "scale"):
            scale[(1,)](x, w, h0, WAVE * 2, D, DP)
            scale[(1,)](x, w, h1, WAVE * 2 + 1, D, DP)
        with sk.cores((3, 2), "add"):
            add[(1,)](h0, h1, x, D, DP)

    def ref(x, b, rt):
        w = b["w"]
        for l in range(WAVES):
            x = bf(bf(x * w[2 * l]) + bf(x * w[2 * l + 1]))
        return x

    return ref


ROWS = [3, 0, 2, 1]


def table(sk, scale, add, script):
    import torch
    from triton.backends.amd_triton_npu.superkernel import Table

    x = sk.ddr("x", D, torch.bfloat16)
    w = sk.ddr("w", WAVES * D, torch.bfloat16)
    h = sk.onchip("h", D, torch.bfloat16)
    k = sk.scalar("k")
    sk.schedule(scale, script)
    with sk.wave():
        with sk.cores((2, 2), "pick"):
            scale[(1,)](x, w, h, Table(ROWS), D, DP)
        with sk.cores((3, 2), "fixed"):
            scale[(1,)](h, w, x, k, D, DP)

    def ref(x, b, rt):
        w = b["w"]
        for l in range(WAVES):
            x = bf(bf(x * w[ROWS[l]]) * w[rt[0]])
        return x

    return ref


def ir_checks(name, text):
    """What each case's module must show."""
    func = text[text.index("func.func") :].split("\n", 1)[0]
    check(text.count("air.segment @") == 1, f"{name}: one segment")
    loop = re.search(r"scf\.for %\S+ = %\S+ to (%\S+) step", text)
    trips = loop and re.search(
        rf"{re.escape(loop[1])} = arith.constant (\d+) : index", text
    )
    check(bool(trips) and int(trips[1]) == WAVES, f"{name}: a loop over the waves")
    check("air.channel @" in text, f"{name}: on-chip tensors moved by channels")
    if name == "chain":
        check(func.count("memref<*xbf16>") == 2, "chain: only x and w are buffers")
    if name == "phases":
        check("air.disable_ping_pong" in text, "phases: a phase loop")
        check("npu_dma_packet" in text, "phases: outputs routed by packet")
    if name == "table":
        check("i32" in func, "table: the scalar is an argument of the func")
        check("arith.select" in text, "table: per-wave values selected")


def error_checks(Superkernel, scale, add, sub, script, proj):
    import torch
    from triton.backends.amd_triton_npu.superkernel import WAVE

    sk = Superkernel("bad", proj, WAVES)
    x = sk.ddr("x", D, torch.bfloat16)
    h = sk.onchip("h", D, torch.bfloat16)
    sk.schedule(add, script)
    try:
        with sk.wave():
            with sk.cores((2, 2), "a"):
                add[(1,)](h, x, x, D, DP)
        check(False, "a read of an unwritten on-chip tensor is an error")
    except ValueError as e:
        check("before any call writes it" in str(e), f"read before write: {e}")
    try:
        with sk.wave():
            with sk.cores((2, 2), "a"):
                add[(1,)](x, x, h, D, DP)
            with sk.cores((3, 2), "a"):
                add[(1,)](h, x, x, D, DP)
        check(False, "one herd at two corners is an error")
    except ValueError as e:
        check("placed at" in str(e), f"herd placed twice: {e}")
    try:
        with sk.wave():
            add[(1,)](x, x, x, D, DP)
        check(False, "a call outside sk.cores is an error")
    except ValueError as e:
        check("sk.cores" in str(e), f"call outside cores: {e}")

    sk = Superkernel("bad_phases", proj, WAVES)
    x = sk.ddr("x", D, torch.bfloat16)
    h0, h1, o = (sk.onchip(n, D, torch.bfloat16) for n in ("h0", "h1", "o"))
    sk.schedule(add, script)
    with sk.wave():
        with sk.cores((2, 2), "w"):
            add[(1,)](x, x, h0, D, DP)
            add[(1,)](x, x, h1, D, DP)
        with sk.cores((3, 2), "r"):
            add[(1,)](h0, x, o, D, DP)
            add[(1,)](h1, x, o, D, DP)
    try:
        sk.module_text()
        check(False, "phases reading different on-chip tensors are rejected")
    except ValueError as e:
        check("not supported" in str(e), f"phases with different inputs: {e}")

    sk = Superkernel("bad_in_place", proj, WAVES)
    x = sk.ddr("x", D, torch.bfloat16)
    h = sk.onchip("h", D, torch.bfloat16)
    sk.schedule(add, script)
    try:
        with sk.wave():
            with sk.cores((2, 2), "a"):
                add[(1,)](h, x, h, D, DP)
        check(False, "a call reading the on-chip tensor it writes first is an error")
    except ValueError as e:
        check("before any call writes it" in str(e), f"in-place first write: {e}")

    sk = Superkernel("bad_order", proj, WAVES)
    x = sk.ddr("x", D, torch.bfloat16)
    w = sk.ddr("w", 2 * WAVES * D, torch.bfloat16)
    h0, h1 = (sk.onchip(n, D, torch.bfloat16) for n in ("h0", "h1"))
    sk.schedule(scale, script)
    sk.schedule(add, script)
    with sk.wave():
        with sk.cores((2, 2), "s"):
            scale[(1,)](x, w, h0, WAVE * 2, D, DP)
        with sk.cores((3, 2), "a"):
            add[(1,)](x, h0, x, D, DP)
        with sk.cores((2, 2), "s"):
            scale[(1,)](x, w, h1, WAVE * 2 + 1, D, DP)
        with sk.cores((4, 2), "b"):
            add[(1,)](x, h1, x, D, DP)
    try:
        sk.module_text()
        check(False, "phases that would run before a call they depend on")
    except ValueError as e:
        check("traced between the phases" in str(e), f"phase order: {e}")

    sk = Superkernel("bad_wiring", proj, WAVES)
    x = sk.ddr("x", D, torch.bfloat16)
    y = sk.ddr("y", D, torch.bfloat16)
    o0, o1 = (sk.onchip(n, D, torch.bfloat16) for n in ("o0", "o1"))
    sk.schedule(sub, script)
    sk.schedule(add, script)
    with sk.wave():
        with sk.cores((2, 2), "d"):
            sub[(1,)](x, y, o0, D, DP, False)
            sub[(1,)](x, y, o1, D, DP, True)
        with sk.cores((3, 2), "a"):
            add[(1,)](o0, o1, x, D, DP)
    try:
        sk.module_text()
        check(False, "phases computing different things share one core body")
    except ValueError as e:
        check("differ" in str(e), f"phases with other operand order: {e}")


def run(name, sk, xclbin, header, ref, rt):
    import numpy as np
    from triton.backends.amd_triton_npu.npu_runner import Device, sync_from, sync_to

    rng = np.random.default_rng(0)
    x0 = bf(rng.standard_normal(D))
    data = {"x": x0}
    for t in sk.buffers:
        if t.name != "x":
            data[t.name] = bf(1 + 0.1 * rng.standard_normal(t.n))
    dev = Device(xclbin, len(sk.buffers), header)
    bos = {}
    for i, t in enumerate(sk.buffers):
        v = data[t.name].view(np.uint32) >> 16  # bfloat16 bits
        v = v.astype(np.uint16)
        bo = dev.bo(v.nbytes, i)
        np.frombuffer(bo.map(), np.uint16, v.size)[:] = v
        sync_to(bo)
        bos[i] = bo
    la = dev.launch(sk.name, bos, rt=rt)
    la.start()
    la.wait()
    xi = [t.name for t in sk.buffers].index("x")
    sync_from(bos[xi])
    got = np.frombuffer(bos[xi].map(), np.uint16, D).astype(np.uint32) << 16
    got = got.view(np.float32)
    want = ref(x0, {k: v.reshape(-1, D) for k, v in data.items()}, rt)
    err = float(np.max(np.abs(got - want) / (np.abs(want) + 1e-2)))
    check(err < 2e-2, f"{name}: NPU result matches numpy (max rel err {err:.2e})")


def main():
    try:
        from triton.backends.amd_triton_npu.config import config_context
        from triton.backends.amd_triton_npu.superkernel import Superkernel
    except ImportError as e:
        skip(f"cannot import the backend: {e}")
    try:
        import pyxrt

        pyxrt.device(0)
        npu = True
    except Exception as e:
        print(f"no NPU ({e}): IR checks only")
        npu = False
    os.environ.setdefault("AMD_TRITON_NPU_TARGET", "npu2")

    scale, add, sub = kernels()
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "elementwise.mlir")
        with open(script, "w") as f:
            f.write(ELEMENTWISE)
        with config_context(bf16_emulation=False):
            error_checks(Superkernel, scale, add, sub, script, os.path.join(tmp, "bad"))
            for name, build, rt in (
                ("chain", chain, ()),
                ("phases", phases, ()),
                ("table", table, (2,)),
            ):
                print(f"{name}:", flush=True)
                sk = Superkernel(name, os.path.join(tmp, name), WAVES)
                ref = build(sk, scale, add, script)
                ir_checks(name, sk.module_text())
                if not npu:
                    continue
                xclbin, header = sk.compile(
                    runtime_loop_tiling_sizes=(),
                    bf16_emulation=False,
                    extra_args=AIRCC_ARGS,
                )
                run(name, sk, xclbin, header, ref, rt)
    if failures:
        print(f"{failures} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
