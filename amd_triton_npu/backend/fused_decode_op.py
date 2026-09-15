# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""FusedDecode: an AIR-module op that bypasses the Triton compiler.

Every other kernel in this backend reaches AIR through the compiler path --
TTIR, then ``triton-shared-opt --triton-to-linalg-experimental``, then a
transform-dialect schedule, then ``_ttshared_to_air``. A fused decode
superkernel does not fit that path: one dispatch spans every decoder layer,
weights are resident across tokens, and the per-token work is an instruction
patch rather than a launch.

mlir-air already builds exactly that module (``programming_examples/fused_decode``).
So this op supplies AIR IR from *outside* the compiler and rejoins the normal
flow at ``_aircc_compile`` -- from there it is treated like any other AIR design:
same aircc invocation, same artifact cache, same ELF/xclbin handling.

    op = FusedDecodeOp.from_air_mlir("air.mlir", kernel_objects=["mv.o"])
    artifacts = op.lower()      # -> {"elf": ...} or {"xclbin": ..., "insts": ...}

This is deliberately *not* a ``@triton.jit`` kernel and not a ``KernelInterface``.
It has no grid, no signature specialization and no autotuning; presenting it as a
Triton kernel would promise semantics it does not have. It is a lowering hook
plus a compile, nothing more.

Status: the ``from_air_mlir`` path is exercised end to end. ``from_builder`` is
the same hook pointed at mlir-air's Python builder instead of a file; it depends
on an importable ``fused_decode`` module and is untested here (see
``scripts/probe_fused_decode.py``).
"""

import os
import shutil

from .config import npu_config


class FusedDecodeError(RuntimeError):
    """Raised when the AIR module or its kernel objects cannot be prepared."""


class FusedDecodeOp:
    """An AIR module from outside the Triton compiler, compiled like any other.

    Args:
        air_module_source: callable returning the AIR module as MLIR text.
        kernel_objects: object files named by ``link_with`` in the module.
            aircc resolves these relative to its working directory, so they are
            staged next to the module before the compile.
        name: used for the project subdirectory.
        runtime_loop_tiling_sizes: shim-DMA BD tiling factors. Empty by
            default, which leaves aircc's own default in place -- see the note
            below.
    """

    def __init__(
        self,
        air_module_source,
        kernel_objects=(),
        name="fused_decode",
        stack_size=2048,
        aircc_args=(),
        merge_artifacts=(),
        runtime_loop_tiling_sizes=(),
    ):
        self._source = air_module_source
        self._kernel_objects = [os.path.abspath(o) for o in kernel_objects]
        self._merge_artifacts = [os.path.abspath(a) for a in merge_artifacts]
        self.name = name
        self.stack_size = stack_size
        self.aircc_args = tuple(aircc_args)
        # The compiler path pins this to (4, 4) to hold the behaviour its
        # kernels were tuned against, from before aircc's default changed to
        # empty (mlir-air #1470). A module authored outside the compiler was
        # tuned against aircc as it is now, and forcing tiling on it re-tiles
        # shim DMA BDs it had already laid out by hand: on the fused decode
        # that turns one `aiex.configure`/`aiex.run` pair into 23, which the
        # ELF path then fails to materialize. Default to leaving it alone.
        self.runtime_loop_tiling_sizes = tuple(runtime_loop_tiling_sizes)

    # -- construction ------------------------------------------------------

    # aircc options the fused decode requires and the compiler path never emits.
    # These mirror the XRTBackend(...) construction in mlir-air's
    # programming_examples/fused_decode/fused_decode.py::run(); they are not
    # tuning knobs. Without --use-lock-race-condition-fix-v2 the shared-L2
    # fan-in gets a counting lock whose writer/reader counts mismatch, and the
    # design deadlocks on device rather than failing to build.
    FUSED_DECODE_AIRCC_ARGS = (
        "--use-lock-race-condition-fix-v2",
        "--coalesce-shim-dma",
    )
    FUSED_DECODE_STACK_SIZE = 10240

    @classmethod
    def from_air_mlir(cls, path, kernel_objects=(), name="fused_decode", **kw):
        """Take a pre-built AIR module from disk.

        The minimal feasibility path: no builder import, no model config -- just
        proof that externally-authored AIR IR compiles through this backend.
        """
        path = os.path.abspath(path)

        def _read():
            with open(path) as f:
                return f.read()

        return cls(_read, kernel_objects=kernel_objects, name=name, **kw)

    @classmethod
    def from_builder(cls, build_module, kernel_objects=(), name="fused_decode", **kw):
        """Take mlir-air's ``fused_decode.build_module()`` (or any callable
        returning an AIR module or its MLIR text).

        Defaults to the fused decode's own aircc options and stack size, since
        that is what this constructor exists to compile; both are overridable.
        """
        kw.setdefault("aircc_args", cls.FUSED_DECODE_AIRCC_ARGS)
        kw.setdefault("stack_size", cls.FUSED_DECODE_STACK_SIZE)

        def _emit():
            module = build_module()
            return module if isinstance(module, str) else str(module)

        return cls(_emit, kernel_objects=kernel_objects, name=name, **kw)

    # -- lowering ----------------------------------------------------------

    def _project_dir(self):
        path = os.path.join(os.path.abspath(npu_config.air_project_path), self.name)
        os.makedirs(path, exist_ok=True)
        return path

    def _stage(self, project_dir):
        """Write the AIR module and stage the objects ``link_with`` names."""
        air_path = os.path.join(project_dir, "air.mlir")
        with open(air_path, "w") as f:
            f.write(self._source())

        for obj in self._kernel_objects:
            if not os.path.exists(obj):
                raise FusedDecodeError(
                    f"kernel object not found: {obj}. It is named by a "
                    f"link_with attribute in the AIR module and must exist "
                    f"before the compile."
                )
            shutil.copy2(obj, os.path.join(project_dir, os.path.basename(obj)))

        # Merge-mode link artifacts are a second, separate class: LLVM IR (.ll)
        # that aiecc links *into* the core module rather than linking against.
        # The fused decode supplies its attention kernels this way, and aiecc
        # resolves them at "air_project/<name>.ll" relative to its working
        # directory -- not beside the module like a link_with object.
        if self._merge_artifacts:
            merge_dir = os.path.join(project_dir, "air_project")
            os.makedirs(merge_dir, exist_ok=True)
            for art in self._merge_artifacts:
                if not os.path.exists(art):
                    raise FusedDecodeError(
                        f"merge-mode link artifact not found: {art}. aiecc links "
                        f"this into the core and fails without it."
                    )
                shutil.copy2(art, os.path.join(merge_dir, os.path.basename(art)))

        return air_path

    def lower(self, output_format=None, npu_version=None):
        """Emit the AIR module and compile it exactly like any other AIR design.

        Returns the artifact dict from ``_aircc_compile`` -- ``{"elf": ...}`` or
        ``{"xclbin": ..., "insts": ...}``.
        """
        # Imported here: driver imports this module's siblings, and a top-level
        # import would close the cycle.
        from .driver import _aircc_compile, _get_output_format, detect_npu_version

        npu_version = npu_version or detect_npu_version()
        output_format = output_format or _get_output_format()

        project_dir = self._project_dir()
        air_path = self._stage(project_dir)

        # aircc resolves link_with relative to its working directory, which is
        # why the objects were staged beside the module.
        cwd = os.getcwd()
        try:
            os.chdir(project_dir)
            return _aircc_compile(
                air_mlir_path=air_path,
                output_format=output_format,
                npu_version=npu_version,
                air_proj_path=project_dir,
                stack_size=self.stack_size,
                extra_args=self.aircc_args,
                runtime_loop_tiling_sizes=self.runtime_loop_tiling_sizes,
            )
        finally:
            os.chdir(cwd)
