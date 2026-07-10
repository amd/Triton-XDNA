# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Multi-launch ELF batching for chained NPU ops (the load_pdi path).

Combines several per-op AIR modules into ONE MLIR module with multiple
``air.launch`` ops, compiles it to a single full ELF, and dispatches the whole
chain with one persistent ``xrt::hw_context`` + one ``xrt::run``. Adjacent
launches reconfigure the AIE array via ``load_pdi`` (firmware-sequenced), so the
~147 ms-per-launch hw_context rebuild is paid once for the whole chain instead of
once per op. Intermediates flow op->op through DDR with no host round-trip.

This is an ADDITIVE path: nothing here touches the per-kernel ``compile_module``
flow used by the example suite.
"""

import hashlib
import os
import re

from triton.runtime.cache import get_cache_manager

from .config import npu_config, config_context
from . import stitching
from .driver import (
    _ttshared_to_air,
    _aircc_compile,
    _get_output_format,
    _get_cached_aircc_artifacts,
    _put_aircc_artifacts,
    detect_npu_version,
)


def _extract_air_arg_types(air_text):
    """Return the func arg type strings of the public func in an AIR module.

    e.g. for ``func.func @relu_kernel(%arg0: memref<*xbf16> {tt.divisibility =
    16 : i32}, %arg1: memref<*xbf16> {...}, %arg2: i32, ...)`` returns
    ``["memref<*xbf16>", "memref<*xbf16>", "i32", ...]`` (attributes stripped).
    """
    for line in air_text.split("\n"):
        if "func.func @" in line and "private" not in line:
            sig = re.search(r"func\.func @\w+\(([^)]*)\)", line)
            if not sig:
                raise ValueError("could not parse func signature from: " + line)
            types = []
            for arg in sig.group(1).split(","):
                arg = arg.strip()
                if not arg:
                    continue
                # "%argN: <type> {attrs}" -> keep <type> (drop name + attr dict)
                after_colon = arg.split(":", 1)[1].strip()
                # strip a trailing "{...}" attribute dict if present
                brace = after_colon.find("{")
                if brace != -1:
                    after_colon = after_colon[:brace].strip()
                types.append(after_colon)
            return types
    raise ValueError("no public func.func found in AIR module")


class _Op:
    __slots__ = ("air_text", "arg_map", "arg_types", "prefix")

    def __init__(self, air_text, arg_map, arg_types, prefix):
        self.air_text = air_text
        self.arg_map = arg_map  # {op_arg_idx: combined_idx}
        self.arg_types = arg_types  # per-op func arg type strings
        self.prefix = prefix


class MultiLaunchBuilder:
    """Accumulate per-op AIR modules and stitch them into one multi-launch ELF.

    Usage::

        b = MultiLaunchBuilder("relu_gelu")
        b.add_op(relu_ttshared, grid=(1,1,1), arg_map={0: 0, 1: 1},
                 transform_script="relu/transform_aie2p.mlir")
        b.add_op(gelu_ttshared, grid=(1,1,1), arg_map={0: 1, 1: 2},
                 transform_script="gelu/transform_aie2p.mlir")
        combined_types = b.combined_arg_types()   # ["memref<*xbf16>", ...]
        elf_path, kernel_name = b.compile()

    ``arg_map`` maps each op's *memref* func-arg index to a combined-func arg
    index. Sharing a combined index across two ops wires op_i's output buffer to
    op_{i+1}'s input buffer (the DDR hand-off). Non-memref (i32 grid metadata)
    args are dropped: they are unused in the launch body.
    """

    def __init__(self, name, air_project_path=None):
        self.name = name
        self.ops = []
        self.air_project_path = air_project_path or npu_config.air_project_path
        # combined_idx -> type string (filled as ops are added; validated for
        # consistency when an index is reused for a hand-off).
        self._combined_types = {}

    def add_op(self, asm_src, grid, arg_map, transform_script=None, actual_sizes=None):
        """Lower one Triton-Shared MLIR op to AIR and register it in the chain.

        Args:
            asm_src: the op's Triton-Shared MLIR (the ``ttsharedir`` bytes/str,
                exactly what ``compile_module``'s launch path receives).
            grid: (gridX, gridY, gridZ) for this op.
            arg_map: {op_memref_arg_idx: combined_idx}. Only memref args matter.
            transform_script: optional path to this op's tiling transform script
                (sets ``transform_tiling_script`` for this op's lowering only).
            actual_sizes: optional actual-sizes string forwarded to lowering.
        """
        gridX, gridY, gridZ = grid
        if transform_script is not None:
            with config_context(transform_tiling_script=transform_script):
                air_module = _ttshared_to_air(
                    asm_src, gridX, gridY, gridZ, actual_sizes=actual_sizes
                )
        else:
            air_module = _ttshared_to_air(
                asm_src, gridX, gridY, gridZ, actual_sizes=actual_sizes
            )
        air_text = str(air_module)
        arg_types = _extract_air_arg_types(air_text)

        # Record/validate combined-arg types for the memref args this op wires.
        for op_idx, comb_idx in arg_map.items():
            ty = arg_types[op_idx]
            prev = self._combined_types.get(comb_idx)
            if prev is not None and prev != ty:
                raise ValueError(
                    f"combined arg {comb_idx} type conflict: {prev} vs {ty} "
                    f"(op {len(self.ops)} '{self.name}')"
                )
            self._combined_types[comb_idx] = ty

        prefix = f"op{len(self.ops)}"
        self.ops.append(_Op(air_text, dict(arg_map), arg_types, prefix))

    def combined_arg_types(self):
        """Type strings for the combined func args, ordered by combined index."""
        n = max(self._combined_types) + 1
        types = []
        for i in range(n):
            if i not in self._combined_types:
                raise ValueError(
                    f"combined arg index {i} never assigned by any op's arg_map"
                )
            types.append(self._combined_types[i])
        return types

    def build_module_text(self):
        """Stitch the registered ops into one combined multi-launch module text."""
        bodies, maps_all, privates = [], [], set()
        for op in self.ops:
            ir = stitching._wrap_ir_in_launch(op.air_text)
            body = stitching._extract_between_func_and_return(ir)
            maps = stitching._extract_affine_maps(ir)
            body = stitching._rename_all(body, op.prefix)
            maps = [stitching._rename_all(m, op.prefix) for m in maps]
            body = stitching._fix_launch_func_args(body, op.prefix, op.arg_map)
            bodies.append(body)
            maps_all.extend(maps)
            for p in stitching._extract_private_funcs(ir):
                privates.add(p.strip())

        types = self.combined_arg_types()
        sig = ",\n    ".join(f"%arg{i}: {ty}" for i, ty in enumerate(types))
        privates_str = "\n  ".join(sorted(privates))
        bodies_str = "\n".join(bodies)
        combined = "\n".join(maps_all) + f"""
module {{
  {privates_str}
  func.func @{self.name}(
    {sig}
  ) {{
{bodies_str}
    return
  }}
}}
"""
        return combined

    def cache_key(self, output_format, npu_version, text=None):
        text = text if text is not None else self.build_module_text()
        key_data = (
            text
            + f"_format_{output_format}"
            + f"_npu_{npu_version}"
            + f"_bf16emu_{npu_config.bf16_emulation}"
        )
        return hashlib.md5(key_data.encode("utf-8")).hexdigest()

    def compile(self, output_format=None):
        """Stitch + compile the chain to one ELF (or xclbin).

        The compiled artifacts are persisted in Triton's on-disk cache keyed by
        the stitched module text (+ format/npu/bf16). On a hit, aircc is skipped
        entirely and the cached paths are returned -- mirrors the per-kernel
        ``compile_module`` cache in driver.py.

        Returns:
            elf:    (elf_path, kernel_name)
            xclbin: (xclbin_path, insts_path)
        """
        output_format = output_format or _get_output_format()
        npu_version = detect_npu_version()
        combined = self.build_module_text()

        key = self.cache_key(output_format, npu_version, combined)
        cache = get_cache_manager(key)
        cached = _get_cached_aircc_artifacts(cache, output_format)
        if cached is not None:
            if output_format == "elf":
                with open(cached["elf_kernel_name_path"]) as f:
                    return cached["elf_path"], f.read()
            return cached["xclbin_path"], cached["insts_path"]

        air_proj = self.air_project_path
        os.makedirs(air_proj, exist_ok=True)
        air_mlir_path = os.path.join(air_proj, f"{self.name}_multilaunch.mlir")
        with open(air_mlir_path, "w") as f:
            f.write(combined)

        artifacts = _aircc_compile(air_mlir_path, output_format, npu_version, air_proj)
        # Cache the format-specific artifacts and return the cached paths.
        cached = _put_aircc_artifacts(cache, artifacts, output_format)
        if output_format == "elf":
            return cached["elf_path"], artifacts["elf_kernel_name"]
        return cached["xclbin_path"], cached["insts_path"]


class MultiLaunchRunner:
    """Dispatch a compiled multi-launch ELF with one persistent context.

    Mirrors the upstream llama32_1b KernelCache.load_and_run BO model:
      - persist {device, elf, hw_context, kernel} for this ELF (one ctx);
      - cache the xrt.bo set per ``bo_key`` and reuse across calls;
      - ``static_indices``  : args written to device on first call only (weights);
      - ``intermediate_indices`` : args the kernel overwrites (DDR hand-off
        scratch) -> never written from host;
      - ``output_indices``  : args synced device->host after the run (default:
        last arg).

    One ``xrt.run`` of the firmware-sequenced ``main`` program runs the whole
    chain. ELF-only (the load_pdi path is npu2/ELF).
    """

    def __init__(self, elf_path, kernel_name):
        import pyxrt as xrt

        self._xrt = xrt
        self.device = xrt.device(0)
        self.elf = xrt.elf(elf_path)
        self.context = xrt.hw_context(self.device, self.elf)
        self.kernel = xrt.ext.kernel(self.context, kernel_name)
        self._bos = {}  # bo_key -> list[xrt.ext.bo]

    def run(
        self,
        inputs,
        *,
        bo_key,
        static_indices=(),
        intermediate_indices=(),
        output_indices=None,
    ):
        """Execute the chain.

        Args:
            inputs: list of numpy arrays, one per combined func arg, in order.
                For ELF args the data_ptr/size come from these arrays. Static
                inputs (weights) and the final outputs are real arrays;
                intermediates can be zero-filled placeholders of the right size.
            bo_key: cache key for the BO set (e.g. f"{name}_L{layer}").
            static_indices: indices written host->device on first call only.
            intermediate_indices: indices never written from host.
            output_indices: indices synced device->host and returned (default:
                {len(inputs)-1}).

        Returns:
            dict {idx: numpy array view} for each output index.
        """
        xrt = self._xrt
        import numpy as np
        from ml_dtypes import bfloat16

        static_set = set(static_indices)
        inter_set = set(intermediate_indices)
        if output_indices is None:
            readback = {len(inputs) - 1}
        else:
            readback = set(output_indices)

        sizes = [a.size * a.itemsize for a in inputs]
        first_call = bo_key not in self._bos
        if first_call:
            self._bos[bo_key] = [xrt.ext.bo(self.device, s) for s in sizes]
        bos = self._bos[bo_key]

        # Write inputs (skip static-after-first and intermediates).
        for i, a in enumerate(inputs):
            if i in static_set and not first_call:
                continue
            if i in inter_set and not first_call:
                continue
            if i in inter_set and first_call:
                # still allocate-clean on first call but no meaningful host data;
                # write zeros so device memory is defined.
                a.fill(0)
            buf = a.view(np.int16) if a.dtype == bfloat16 else a
            mv = bos[i].map()
            src = np.frombuffer(buf, dtype=np.uint8)
            dst = np.frombuffer(mv, dtype=np.uint8, count=len(src))
            np.copyto(dst, src, casting="no")
            bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        run = xrt.run(self.kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        run.start()
        run.wait2()

        results = {}
        for idx in readback:
            bos[idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            a = inputs[idx]
            results[idx] = np.frombuffer(
                bos[idx].map(), dtype=a.dtype, count=a.size
            ).reshape(a.shape)
        return results

    def unload(self):
        """Release XRT handles in dependency order while the runtime is alive.

        Mirrors mlir-air's ``XRTBackend.unload()``: drop dependents before the
        device (BOs/kernel -> context -> elf -> device) so pyxrt's C++
        destructors run cleanly, instead of leaving them to arbitrary-order GC
        at interpreter shutdown (which can fault in XRT teardown). Idempotent.
        """
        self._bos = {}
        self.kernel = None
        self.context = None
        self.elf = None
        self.device = None

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass


class NPUChain:
    """Model-facing wrapper: declare a chain of NPU ops, dispatch as one ELF.

    Captures each op's Triton-Shared MLIR at ``add()`` time (via Triton
    ``warmup`` -- compiles to ttsharedir without an XRT launch), then on the
    first ``run()`` stitches + compiles the whole chain into a single load_pdi
    ELF (``MultiLaunchBuilder``) and dispatches it through a persistent
    ``MultiLaunchRunner``. Subsequent ``run()`` calls reuse the ELF, hw_context,
    and (per ``bo_key``) the BO set.

    The ops MUST be fusion-ready: each op's output buffer is the next op's input
    buffer, same dtype/shape, no host work in between (no bias-add, dtype cast,
    pad/slice, or K-accumulation between launches). Wire the hand-off by giving
    the producing op's output arg and the consuming op's input arg the same
    combined index in ``arg_map``.

    Usage::

        chain = NPUChain("linear_gelu")
        chain.add(linear_kernel, grid=(M//BM, N//BN), arg_map={0:0, 1:1, 2:2},
                  transform_script=mm_script,
                  args=(a, w, tmp, M, N, K, ...), constexprs={...})
        chain.add(gelu_kernel, grid=(NT,), arg_map={0:2, 1:3},
                  transform_script=elem_script,
                  args=(tmp, y, NT), constexprs={"BLOCK_SIZE": 1024})
        out = chain.run([a_np, w_np, tmp_np, y_np],
                        static_indices={1}, intermediate_indices={2},
                        output_indices={3})

    ``args``/``constexprs`` are only used to drive warmup compilation (shapes +
    grid determine the lowered IR); the actual data is passed to ``run`` as numpy
    arrays in combined-arg order.
    """

    def __init__(self, name, air_project_path=None):
        self.name = name
        self.air_project_path = air_project_path
        self._specs = []  # (kernel, grid, arg_map, transform_script, args, constexprs)
        self._builder = None
        self._runner = None
        self._elf_path = None
        self._kernel_name = None

    def add(
        self,
        kernel,
        grid,
        arg_map,
        *,
        args,
        constexprs=None,
        transform_script=None,
        actual_sizes=None,
    ):
        """Register one Triton kernel as the next launch in the chain.

        Args:
            kernel: a ``@triton.jit`` function.
            grid: (gridX[, gridY[, gridZ]]) tuple for this op.
            arg_map: {op_memref_arg_idx: combined_idx}. Share a combined index
                between a producer's output and a consumer's input to wire the
                DDR hand-off.
            args: positional args for warmup (tensors + scalars in signature
                order). Tensors may be real torch tensors; their shapes/dtypes
                drive lowering.
            constexprs: dict of constexpr kwargs for warmup.
            transform_script: per-op tiling transform script path.
            actual_sizes: optional actual-sizes string forwarded to lowering.
        """
        g = tuple(grid)
        while len(g) < 3:
            g = g + (1,)
        self._specs.append(
            (
                kernel,
                g,
                dict(arg_map),
                transform_script,
                tuple(args),
                dict(constexprs or {}),
                actual_sizes,
            )
        )
        return self

    def _capture_ttshared(self, kernel, grid, args, constexprs):
        """Warmup-compile the kernel to obtain its ttsharedir source.

        Forces the NPU driver active for the warmup so the kernel lowers through
        the NPU backend (whose binary_ext is ``ttsharedir``). Without this, in a
        hetero model the GPU driver may be active and ``asm`` would lack
        ``ttsharedir`` (KeyError). The previously-active driver is restored.
        """
        import triton
        from .driver import NPUDriver

        triton.runtime.driver.set_active(NPUDriver())
        try:
            with config_context(compile_only=True):
                compiled = kernel.warmup(*args, grid=grid, **constexprs)
        finally:
            triton.runtime.driver.reset_active()
        return compiled.asm["ttsharedir"]

    def _build(self):
        b = MultiLaunchBuilder(self.name, air_project_path=self.air_project_path)
        for (
            kernel,
            grid,
            arg_map,
            tscript,
            args,
            constexprs,
            actual_sizes,
        ) in self._specs:
            asm_src = self._capture_ttshared(kernel, grid, args, constexprs)
            b.add_op(
                asm_src,
                grid,
                arg_map,
                transform_script=tscript,
                actual_sizes=actual_sizes,
            )
        self._builder = b
        self._elf_path, self._kernel_name = b.compile()
        self._runner = MultiLaunchRunner(self._elf_path, self._kernel_name)

    def run(
        self,
        inputs,
        *,
        bo_key=None,
        static_indices=(),
        intermediate_indices=(),
        output_indices=None,
    ):
        """Build (first call) + dispatch the chain. Returns {idx: ndarray}."""
        if self._runner is None:
            self._build()
        return self._runner.run(
            inputs,
            bo_key=bo_key or self.name,
            static_indices=static_indices,
            intermediate_indices=intermediate_indices,
            output_indices=output_indices,
        )

    def close(self):
        """Release the chain's persistent XRT context/BOs (see
        ``MultiLaunchRunner.unload``). Safe to call repeatedly."""
        if self._runner is not None:
            self._runner.unload()
            self._runner = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
