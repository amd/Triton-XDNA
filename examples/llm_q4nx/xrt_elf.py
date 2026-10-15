# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""mlir-air's XRT full-ELF decode, with its scratchpad written through pyxrt.

mlir-air's `ElfDecode` writes the decode's scratchpad parameters through
mlir-aie's `ParameterScratchpad`, a C++ extension that links its own LLVM.
Imported after Triton, it aborts the process ("Option 'print-inst-addrs'
registered more than once!"). The parameters are a few 32-bit words in a buffer
pyxrt already exposes, so `install` swaps in a writer that needs nothing else.
"""

import numpy as np


class Scratchpad:
    """Named parameters in an XRT run's control scratchpad.

    The same interface as mlir-aie's `ParameterScratchpad` -- `write(name,
    value)`, then `sync()` -- and the same encoding: one uint32 per parameter
    at its index in params.txt, a `core` parameter shifted left by 2 (the
    firmware masks the low bits and the core shifts back), an `addr` parameter
    written as is and checked against the range params.txt gives it.
    """

    def __init__(self, run, params_path):
        import pyxrt

        self._to_device = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self._bo = run.get_ctrl_scratchpad_bo()
        self._words = np.frombuffer(self._bo.map(), dtype=np.uint32)
        self._params = parse(params_path)
        for name, (slot, _, _) in self._params.items():
            if slot >= self._words.size:
                raise ValueError(
                    f"{params_path}: {name} is slot {slot}, past the "
                    f"{self._words.size}-word scratchpad"
                )

    def write(self, name, value):
        slot, kind, bounds = self._params[name]
        v = int(value)
        if bounds is not None and not bounds[0] <= v <= bounds[1]:
            raise ValueError(f"{name}={v} is outside [{bounds[0]}, {bounds[1]}]")
        bits = v & 0xFFFFFFFF
        if kind == "core":
            bits = (bits << 2) & 0xFFFFFFFF
        self._words[slot] = bits

    def sync(self):
        self._bo.sync(self._to_device)


def parse(path):
    """params.txt as {name: (slot, kind, (min, max) or None)}.

    A count line, then `<name> <slot> <type> <kind> <min> <max>` per parameter;
    anything after those (the DMA length bounds) is not needed to write them.
    """
    with open(path) as f:
        lines = [ln.split() for ln in f if ln.strip()]
    n = int(lines[0][0])
    params = {}
    for name, slot, _type, kind, lo, hi in (ln[:6] for ln in lines[1 : 1 + n]):
        if kind not in ("core", "addr"):
            raise ValueError(f"{path}: {name} has kind {kind!r}")
        bounds = None if lo == "-" else (int(lo), int(hi))
        params[name] = (int(slot), kind, bounds)
    return params


def install(air):
    """Make `air`'s full-ELF decoder write its scratchpad with `Scratchpad`.

    A no-op for a driver without one. Idempotent: the decode_elf module is
    shared by every model's driver.
    """
    load = getattr(air, "_load_elf_mod", None)
    if load is None:
        return
    elf = load()
    if getattr(elf.ElfDecode, "_pyxrt_scratchpad", False):
        return

    class ElfDecode(elf.ElfDecode):
        _pyxrt_scratchpad = True

        def bind(self, bos):
            self.run = self.xrt.run(self.kern)
            for i, b in enumerate(bos):
                self.run.set_arg(i, b)
            self.params = Scratchpad(self.run, self.params_path)
            # Older builds declare the context length as a trailing scalar
            # argument, which XRT requires set; newer ones read it only from
            # the scratchpad and declare no such argument.
            try:
                self.run.set_arg(self.scalar_arg, 0)
                self._l_arg = True
            except IndexError:
                self._l_arg = False
            print(
                f"[xrt-elf] ONE full ELF: {self.elf_path.name}, L in "
                f"{len(self.params._params)} scratchpad words",
                flush=True,
            )

        def dispatch(self, L, np):
            self.params.write(
                self.append_param, np.int32(L * self.append_scale + self.append_addend)
            )
            self.params.write(self.mask_param, np.int32(L))
            self.params.sync()
            if self._l_arg:
                self.run.set_arg(self.scalar_arg, L)
            self.run.start()
            self.run.wait(elf.DISPATCH_TIMEOUT_MS)
            return self.run.state()

    elf.ElfDecode = ElfDecode
