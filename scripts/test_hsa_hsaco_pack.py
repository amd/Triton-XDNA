#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pack a compiled kernel into an hsaco and verify its section, without hardware.

Compiles through the ordinary XRT path (link_profile="xrt"), which needs
only the aiecc toolchain -- no ROCR, no HSA runtime build, no live NPU
device. ``npu_config.target`` bypasses even the device-presence check that
`detect_npu_version` would otherwise do. The packed hsaco is produced by
calling `_pack_hsaco` directly on the real compiled artifacts, so this never
touches `compile_module`'s HSA link_profile at all -- which (unconditionally,
regardless of compile_only) builds and links the C++ HSA runtime against a
real ROCR install, the one thing this test is specifically trying to avoid
needing.

Exercises both packed-kernel kinds `_pack_hsaco` supports: PDI+insts (forcing
xclbin output format) and full ELF (npu2's default format).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "examples", "zero_copy"))


def skip(msg):
    print(f"SKIP: {msg}")
    sys.exit(77)


def _compile_and_pack(npu_config, driver, add_chain, torch, output_format):
    """Compile add_kernel via XRT with the given output_format, then pack it.

    Returns the parsed hsaco section info (one entry per packed kernel).
    """
    npu_config.output_format = output_format
    npu_config.compile_only = True
    import triton

    triton.runtime.driver.set_active(driver.NPUDriver("xrt"))
    zeros = torch.zeros(4096, dtype=torch.float32)
    grid = (4096 // add_chain.BLOCK_SIZE,)
    compiled = add_chain.add_kernel[grid](
        zeros, zeros, zeros, 4096, BLOCK_SIZE=add_chain.BLOCK_SIZE
    )
    cache_dir = driver.get_npu_cache_dir(compiled)
    if cache_dir is None:
        skip("no NPU cache dir -- compile_only path did not run")

    npu_version = driver.detect_npu_version("xrt")
    arch = driver._NPU_VERSION_TO_ARCH[npu_version]
    kernel_name = getattr(compiled, "name", None) or add_chain.add_kernel.fn.__name__

    if output_format == "elf":
        elf_path = os.path.join(cache_dir, "aie.elf")
        kname_path = os.path.join(cache_dir, "elf_kernel_name.txt")
        if not (os.path.isfile(elf_path) and os.path.isfile(kname_path)):
            print(f"FAIL: expected {elf_path} and {kname_path}")
            return None
        with open(kname_path) as f:
            elf_kernel_name = f.read().strip()
        artifacts = {"elf_path": elf_path, "elf_kernel_name": elf_kernel_name}
        pack_kernel_name = elf_kernel_name
    else:
        bin_path = os.path.join(cache_dir, "aie.xclbin")
        insts_path = os.path.join(cache_dir, "insts.bin")
        if not (os.path.isfile(bin_path) and os.path.isfile(insts_path)):
            print(f"FAIL: expected {bin_path} and {insts_path}")
            return None
        artifacts = {"bin_path": bin_path, "insts_path": insts_path}
        pack_kernel_name = kernel_name

    hsaco_bytes = driver._pack_hsaco(output_format, artifacts, arch, pack_kernel_name)

    from aie.compiler.hsaco import dump

    tmp_path = os.path.join(cache_dir, "test_pack.hsaco")
    with open(tmp_path, "wb") as f:
        f.write(hsaco_bytes)
    sections = dump.read_sections_from_hsaco(tmp_path)
    return [dump.parse_section(data) for _, data in sections]


def main():
    try:
        import torch
        import triton.backends.amd_triton_npu.driver as driver
        from aie.compiler import hsaco  # noqa: F401  (import-availability check)
        from common import add_chain
    except ImportError as e:
        skip(f"cannot import the backend or aie.compiler.hsaco: {e}")

    # Bypass hardware/device detection entirely: this test only needs aiecc
    # to produce artifacts, never a live NPU or ROCR install.
    driver.npu_config.target = os.environ.get("AMD_TRITON_NPU_TARGET", "npu2")

    ok = True
    for output_format in ("elf", "xclbin"):
        infos = _compile_and_pack(
            driver.npu_config, driver, add_chain, torch, output_format
        )
        if infos is None:
            ok = False
            continue
        for info in infos:
            print(
                f"output_format={output_format} "
                f"kernels={[k['name'] for k in info['kernels']]}"
            )
            for k in info["kernels"]:
                if k["num_cols"] <= 0:
                    print(f"FAIL: kernel {k['name']!r} has num_cols={k['num_cols']}")
                    ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
