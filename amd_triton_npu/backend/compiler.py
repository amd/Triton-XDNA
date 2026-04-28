# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import sys
import tempfile
import os
import re
import shutil
import subprocess
import functools
import sys
from pathlib import Path

from .config import npu_config

IS_WINDOWS = sys.platform == "win32"


def _get_amd_triton_npu_opt_path() -> str:
    binary_name = "triton-shared-opt.exe" if IS_WINDOWS else "triton-shared-opt"
    path = Path(__file__).resolve().parent.parent.parent / "triton_shared" / binary_name
    if not os.path.isdir(path.parent):
        raise RuntimeError(f"Could not find triton-shared binaries at {path}")
    return path


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    if IS_WINDOWS and not bin_name.endswith(".exe"):
        bin_name += ".exe"
    return os.path.join(path, bin_name)


def _dump_ir_if_needed(files):
    """
    Dump intermediate IR files to the air_project directory.

    Files are always dumped to the air_project path (controlled by
    ``npu_config.air_project_path`` or defaulting to ./air_project/).
    """
    air_proj_path = npu_config.air_project_path
    os.makedirs(air_proj_path, exist_ok=True)
    for f in files:
        shutil.copy(f, os.path.join(air_proj_path, os.path.basename(f)))


def _ttir_to_ttsharedir(mod):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        amd_triton_npu_opt_path = _get_amd_triton_npu_opt_path()
        cmd = [
            amd_triton_npu_opt_path,
            src_path,
            "--triton-to-linalg-experimental",
            "--mlir-print-debuginfo",
            "-o",
            dst_path,
        ]
        if npu_config.debug:
            subprocess.check_call(cmd)
        else:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            if result.returncode != 0:
                if result.stdout:
                    stderr_buf = getattr(sys.stderr, "buffer", None)
                    if stderr_buf is not None:
                        stderr_buf.write(result.stdout)
                    else:
                        sys.stderr.write(
                            result.stdout.decode("utf-8", errors="replace")
                        )
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, output=result.stdout
                )
        _dump_ir_if_needed([src_path])
        return Path(dst_path).read_text()


def _optimize_ttsharedir(ttsharedir: str, metadata):
    pattern = r"func.func @(\w+)\(.+"
    matches = re.findall(pattern, ttsharedir)
    assert len(matches) == 1
    metadata["name"] = matches[0]
    # We don't apply any optimizations now, but we can add passes if needed.
    return ttsharedir


def _llir_to_bin(llir: str, metadata):
    pattern = r"define void @(\w+)\(.+"
    matches = re.findall(pattern, llir)
    assert len(matches) == 1
    metadata["name"] = matches[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        dst_path = os.path.join(tmpdir, "kernel.o")
        Path(src_path).write_text(llir)
        llc_path = _get_llvm_bin_path("llc")
        subprocess.check_call([llc_path, src_path, "-o", dst_path])
        # Actually it's text-format assembly.  Use read_text().
        return Path(dst_path).read_text()


@dataclass(frozen=True)
class NPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    warp_size: int = 32  # TODO: Update to reflect NPU arch.
    backend_name: str = "npu"
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    # Disable FP8 here since this is a sample NPU backend.
    # Target specific backends can eanble it with supported types.
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    sanitize_overflow: bool = True
    instrumentation_mode: str = ""

    def __post_init__(self):
        pass

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class NPUBackend(BaseBackend):
    binary_ext = "ttsharedir"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "npu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def get_target_name(self, options) -> str:
        return f"npu"

    def parse_options(self, opts) -> Any:
        args = {"arch": self.target.arch}
        args.update(
            {k: opts[k] for k in NPUOptions.__dataclass_fields__.keys() if k in opts}
        )
        return NPUOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name,
        )

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `amd_triton_npu.cc`
    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        passes.common.add_cse(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def gluon_to_ttgir(src, metadata, options):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        pm.run(mod)
        return mod

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(
                src, metadata, options
            )
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(
                src, metadata, options
            )
        stages["ttsharedir"] = lambda src, metadata: _optimize_ttsharedir(
            _ttir_to_ttsharedir(src), metadata
        )

    @functools.lru_cache()
    def hash(self):
        return self.target

    # The NPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
