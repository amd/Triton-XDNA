# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import functools
import hashlib
import json
import logging
import tempfile
import sys
import functools

import os, subprocess, platform
import importlib.util
import importlib.metadata
import shutil

from pathlib import Path
from typing import NamedTuple, Optional

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

import aie
import air.compiler.aircc.main as aircc
from air.compiler.util import run_transform
from air.ir import *
import air.passmanager

from .config import npu_config, _VALID_RUNTIMES
from .codegen import (
    extract_actual_sizes,
    extract_signature_and_constants,
    extracted_type,
    format_of,
    ty_to_cpp,
)

IS_WINDOWS = sys.platform == "win32"

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
if npu_config.debug:
    logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.propagate = False


def _lift_tensor_numel_limit():
    """Raise Triton's max-tensor-numel guard for the AIR-tiled NPU backend.

    Triton caps a block tensor at 2**20 elements because on a GPU a block
    becomes a register tile. On this backend a whole-tile load + tl.dot lowers
    to a single linalg.matmul that the transform script tiles across L3/L2/L1
    on-device, so the cap doesn't apply -- it only forced the NPU matmul wrapper
    to chunk K and reduce partials on the host. validate_block_shape reads this
    value from the _utils module global at call time, so bumping it here covers
    every call site (including core.py's imported binding).
    """
    import triton._utils as _tu

    cap = 1 << 22  # 256 (BLOCK_M) x 8192 (largest padded K) = 2**21, with headroom
    cur = getattr(_tu, "TRITON_MAX_TENSOR_NUMEL", None)
    if cur is not None and cur < cap:
        _tu.TRITON_MAX_TENSOR_NUMEL = cap


_lift_tensor_numel_limit()

autotune_time = False


# -------------------- Launcher ----------------------------


@functools.lru_cache(maxsize=8)
def _is_peano_root(root: str) -> bool:
    """True only for an LLVM install that can actually target AIE.

    Checking for ``bin/opt`` is not enough: *every* LLVM install has one, so
    an ``LLVM_BINARY_DIR`` pointing at Triton's own LLVM (which is what
    ``compiler.py`` uses it for) passed the old check and got handed to aiecc
    as ``--peano``. That LLVM has no AIE backend, so it fails partway through
    the pipeline on Peano IR it cannot parse, e.g.

        error: floating point constant invalid for type
          %2 = call <16 x bfloat> @llvm.aie2p.v16accfloat.to.v16bf16(...)

    Ask llc which targets it registers instead; only Peano lists aie2.
    """
    if not root:
        return False
    root_path = Path(root)
    opt_name = "opt.exe" if IS_WINDOWS else "opt"
    llc_name = "llc.exe" if IS_WINDOWS else "llc"
    if not (root_path / "bin" / opt_name).exists():
        return False
    llc = root_path / "bin" / llc_name
    if not llc.exists():
        return False
    try:
        result = subprocess.run(
            [str(llc), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except Exception as e:
        logger.debug("Could not query %s for its targets: %s", llc, e)
        return False
    return "aie2" in result.stdout


def _find_mlir_air_binary(binary_name: str) -> str:
    """Locate a binary inside the mlir-air install prefix.

    Search order:
    1. MLIR_AIR_INSTALL_DIR environment variable
    2. mlir-air .pth file (when built from source on Windows)
    3. pip-wheel layout: navigate up from aircc module location
    4. shutil.which() (PATH lookup)

    Returns:
        str: Absolute path to the binary

    Raises:
        RuntimeError: If not found
    """
    candidates = []

    # 1. Explicit env var
    mlir_air_env = os.environ.get("MLIR_AIR_INSTALL_DIR")
    if mlir_air_env:
        candidates.append(Path(mlir_air_env) / "bin" / binary_name)

    # 2. .pth file: points to <install>/python, so <install>/bin has binaries
    import site

    for sp in site.getsitepackages():
        pth = os.path.join(sp, "mlir-air.pth")
        if os.path.exists(pth):
            with open(pth) as f:
                pth_dir = f.read().strip()
            if pth_dir:
                candidates.append(Path(pth_dir).resolve().parent / "bin" / binary_name)

    # 3. pip-wheel layout: aircc.__file__ -> .../mlir_air/python/air/compiler/aircc/main.py
    aircc_path = Path(aircc.__file__).resolve()
    candidates.append(
        aircc_path.parent.parent.parent.parent.parent / "bin" / binary_name
    )
    # Also try 3 levels up (namespace package: air/compiler/aircc/main.py -> site-packages)
    candidates.append(aircc_path.parent.parent.parent / "bin" / binary_name)

    for c in candidates:
        if c.exists():
            return str(c)

    # 4. Fall back to PATH
    found = shutil.which(binary_name)
    if found:
        return found

    tried = "\n  ".join(str(c) for c in candidates)
    raise RuntimeError(
        f"Could not find {binary_name}. Searched:\n  {tried}\n"
        f"Set MLIR_AIR_INSTALL_DIR to the mlir-air install prefix."
    )


def _get_air_opt_path() -> str:
    """Get the path to the air-opt binary."""
    binary_name = "air-opt.exe" if IS_WINDOWS else "air-opt"
    return _find_mlir_air_binary(binary_name)


def _get_xrt_path() -> str:
    """Get the path to the XRT development directory (headers + import lib).

    Search order:
    1. XILINX_XRT environment variable (standard on both Linux and Windows)
    2. (Windows) C:\\Program Files\\AMD\\xrt  (recommended install location)
    3. (Linux) /opt/xilinx/xrt

    The returned directory must contain the SDK components (include/xrt headers
    and lib/) needed for JIT compilation.  A runtime-only installation (e.g.
    DLLs from the NPU driver) is not sufficient.

    On Windows, download xrt_windows_sdk.zip from the Xilinx/XRT releases page
    and extract the xrt/ directory to C:\\Program Files\\AMD\\xrt.
    """

    def _validate_xrt_sdk(path: str, source: str) -> str:
        """Ensure *path* contains the SDK components needed for compilation."""
        has_headers = os.path.isdir(os.path.join(path, "include", "xrt"))
        has_lib = os.path.isdir(os.path.join(path, "lib"))
        if has_headers and has_lib:
            return path
        if os.path.isdir(path):
            # Directory exists but is missing SDK pieces – give a targeted hint.
            missing = []
            if not has_headers:
                missing.append("include/xrt (headers)")
            if not has_lib:
                missing.append("lib (import libraries)")
            raise RuntimeError(
                f"XRT directory found via {source} at {path}, but it appears to "
                f"be a runtime-only installation — missing: {', '.join(missing)}. "
                "Download xrt_windows_sdk.zip from https://github.com/Xilinx/XRT/releases "
                "and extract the full xrt/ directory (with include/ and lib/) to that location."
            )
        return ""  # path doesn't exist at all

    env_path = os.getenv("XILINX_XRT", "")
    if env_path:
        result = _validate_xrt_sdk(env_path, "XILINX_XRT environment variable")
        if result:
            return result

    if IS_WINDOWS:
        program_files = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        default_path = os.path.join(program_files, "AMD", "xrt")
        result = _validate_xrt_sdk(default_path, "default location")
        if result:
            return result
    else:
        result = _validate_xrt_sdk("/opt/xilinx/xrt", "default location")
        if result:
            return result

    raise RuntimeError(
        "XRT development files not found. "
        "Download xrt_windows_sdk.zip from https://github.com/Xilinx/XRT/releases "
        "and extract the xrt/ directory to C:\\Program Files\\AMD\\xrt "
        "(or set the XILINX_XRT environment variable to its location)."
    )


# Headers the shared HSA runtime (HsaRuntime.cpp) includes, relative to a ROCm
# prefix. hsa_ext_amd_aie.h carries the AIE dispatch extension and is absent
# from stock ROCm releases, so requiring it here turns "your ROCm has no AIE
# support" into an error at discovery rather than a header-not-found deep in
# the compile.
_ROCR_HEADERS = ("hsa/hsa.h", "hsa/hsa_ext_amd.h", "hsa/hsa_ext_amd_aie.h")

# Library directories to probe under a ROCm prefix, in order. lib64 and the
# Debian/Ubuntu multiarch directory matter for system prefixes such as /usr.
_ROCR_LIB_DIRS = ("lib", "lib64", os.path.join("lib", "x86_64-linux-gnu"))

# Library file names, in preference order. The unversioned name is a symlink
# from the development package and is what ``-lhsa-runtime64`` needs; ROCm
# wheels built by TheRock ship no symlinks, so only the SONAME exists there and
# we link it by absolute path instead.
_ROCR_LIB_NAMES = ("libhsa-runtime64.so", "libhsa-runtime64.so.1")

# Python packages that may contain a pip-installed ROCm (TheRock). The core
# runtime package is the one carrying libhsa-runtime64; ``rocm_sdk`` is asked
# for the platform-specific name first, since it varies by target.
_PIP_ROCM_PACKAGE = "_rocm_sdk_core"


class _RocrInstall(NamedTuple):
    """A validated ROCm/ROCR installation usable for AIE dispatch."""

    prefix: str  # install root, for diagnostics
    include_dir: str  # contains hsa/hsa.h
    lib_dir: str  # contains the library, for -L and -rpath
    lib_path: str  # the library file itself
    source: str  # how we found it, for error messages


def _pip_rocm_prefix() -> str:
    """Root of a pip-installed ROCm (TheRock), or ``""`` if there is none.

    Resolves the package directory from its import spec rather than shelling
    out to the ``rocm-sdk`` CLI: ``rocm-sdk path --root`` reports the
    multi-gigabyte devel tree, not the runtime package that actually holds
    libhsa-runtime64.
    """
    names = []
    try:
        import rocm_sdk

        # The core package name is platform-specific and the accessor for it is
        # not a stable public API, so try the known spellings and fall through
        # to the default name if none of them are present.
        for attr in (
            "determine_platform_package_name",
            "_determine_platform_package_name",
            "platform_package_name",
        ):
            fn = getattr(rocm_sdk, attr, None)
            if callable(fn):
                try:
                    name = fn()
                except Exception:
                    continue
                if name:
                    names.append(name)
                    break
    except ImportError:
        pass
    names.append(_PIP_ROCM_PACKAGE)

    for name in names:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is None:
            continue
        # A namespace package has no origin; use its search path instead.
        origin = spec.origin
        if origin and origin != "namespace":
            return os.path.dirname(origin)
        for location in spec.submodule_search_locations or ():
            return location
    return ""


def _probe_rocr(prefix: str, source: str) -> tuple[Optional[_RocrInstall], str]:
    """Check whether ``prefix`` is a ROCm install we can build against.

    Returns ``(install, "")`` on success, or ``(None, reason)`` explaining what
    disqualified it (an empty reason means the prefix simply does not exist, so
    it is not worth reporting).
    """
    if not prefix or not os.path.isdir(prefix):
        return None, ""

    include_dir = os.path.join(prefix, "include")
    missing = [
        h for h in _ROCR_HEADERS if not os.path.isfile(os.path.join(include_dir, h))
    ]

    lib_path = ""
    for rel in _ROCR_LIB_DIRS:
        lib_dir = os.path.join(prefix, rel)
        for name in _ROCR_LIB_NAMES:
            candidate = os.path.join(lib_dir, name)
            if os.path.isfile(candidate):
                lib_path = candidate
                break
        if lib_path:
            break

    if missing or not lib_path:
        reasons = []
        if missing:
            reasons.append(
                "missing header(s) " + ", ".join("include/" + m for m in missing)
            )
        if not lib_path:
            reasons.append(
                f"no {_ROCR_LIB_NAMES[0]} under " + ", ".join(_ROCR_LIB_DIRS)
            )
        return None, f"{prefix} (via {source}): " + "; ".join(reasons)

    return (
        _RocrInstall(
            prefix=prefix,
            include_dir=include_dir,
            lib_dir=os.path.dirname(lib_path),
            lib_path=lib_path,
            source=source,
        ),
        "",
    )


@functools.lru_cache(maxsize=1)
def _get_rocr_install() -> _RocrInstall:
    """Locate a ROCm/ROCR install providing the AIE-capable HSA runtime.

    Search order:

    1. ``AMD_NPU_ROCR_PATH``
    2. ``ROCM_PATH``
    3. a pip-installed ROCm (TheRock), see :func:`_pip_rocm_prefix`
    4. the system location ``/opt/rocm``

    Every candidate must supply the headers HsaRuntime.cpp includes -- notably
    ``hsa/hsa_ext_amd_aie.h`` -- and libhsa-runtime64. Candidates that fail are
    collected and reported together, so a prefix that was set explicitly but is
    unusable says *why* instead of being silently skipped.
    """
    candidates = [
        ("AMD_NPU_ROCR_PATH", os.getenv("AMD_NPU_ROCR_PATH", "")),
        ("ROCM_PATH", os.getenv("ROCM_PATH", "")),
        ("pip-installed ROCm", _pip_rocm_prefix()),
        ("default location", "/opt/rocm"),
    ]

    rejected = []
    for source, prefix in candidates:
        install, reason = _probe_rocr(prefix, source)
        if install is not None:
            return install
        if reason:
            rejected.append(reason)

    detail = ""
    if rejected:
        detail = "\nRejected candidates:\n" + "\n".join("  - " + r for r in rejected)
    raise RuntimeError(
        "No ROCR install with AIE support found. The HSA launch runtime "
        "(AMD_TRITON_NPU_RUNTIME=hsa) needs a ROCm prefix providing "
        + ", ".join("include/" + h for h in _ROCR_HEADERS)
        + " and lib/libhsa-runtime64.so. Stock ROCm releases do not ship "
        "hsa/hsa_ext_amd_aie.h; build an AIE-capable ROCR with "
        "scripts/build-rocr.sh and point AMD_NPU_ROCR_PATH at its install "
        "prefix." + detail
    )


def _run_compile(cmd, env=None):
    """Run a compiler/build subprocess, echoing its output to stderr on failure.

    Honors ``npu_config.debug`` (streams output live). On a non-zero exit,
    writes the captured combined stdout/stderr to our stderr and raises
    ``subprocess.CalledProcessError``. Shared by the aircc, launcher, and
    HSA-runtime-lib builds.
    """
    if npu_config.debug:
        subprocess.check_call(cmd, env=env)
        return
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    if result.returncode != 0:
        if result.stdout:
            stderr_buf = getattr(sys.stderr, "buffer", None)
            if stderr_buf is not None:
                stderr_buf.write(result.stdout)
            else:
                sys.stderr.write(result.stdout.decode("utf-8", errors="replace"))
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout
        )


def _build_hsa_runtime_lib(include_dir: str, rocr: _RocrInstall) -> str:
    """Build (once, cached) the shared HSA runtime library; return its directory.

    Compiles ``include/HsaRuntime/HsaRuntime.cpp`` into ``libtriton_npu_hsa.so``
    linked against ROCR. The build is cached (keyed by source + ROCR path), so it
    happens only once per toolchain. Every generated HSA launcher links this one
    ``.so``; because the dynamic linker loads a shared dependency once per
    process, the ``HsaRuntime`` singleton inside it is process-global (one queue),
    which is what lets multiple kernel signatures run on an AIE agent limited to
    ``QUEUES_MAX == 1``.

    Built at runtime (not as a wheel/install-time artifact) on purpose: ROCR is
    resolved from the environment (see ``_get_rocr_install``), and the wheel may
    be built on a host without ROCR. Do not move this into setup.py/CMake without
    solving ROCR-less builds.

    Returns the directory containing the ``.so`` (for the launcher's -L / -rpath).
    """
    lib_name = "libtriton_npu_hsa.so"
    src_path = os.path.join(include_dir, "HsaRuntime", "HsaRuntime.cpp")
    with open(src_path, "rb") as f:
        src_bytes = f.read()
    key_src = src_bytes + f"_rocr_{rocr.lib_path}_hsartlib".encode()
    key = hashlib.md5(key_src).hexdigest()
    cache = get_cache_manager(key)
    lib_path = cache.get_file(lib_name)
    if lib_path is not None:
        return os.path.dirname(lib_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, lib_name)
        cmd = [
            "g++",
            "-std=c++23",
            "-shared",
            "-fPIC",
            "-O2",
            "-pthread",
            src_path,
            f"-I{include_dir}",
            f"-I{rocr.include_dir}",
            f"-L{rocr.lib_dir}",
            f"-Wl,-rpath,{rocr.lib_dir}",
            # Link the resolved file rather than -lhsa-runtime64: a ROCm wheel
            # ships only libhsa-runtime64.so.1, and without the unversioned
            # symlink the linker's -l lookup fails.
            (
                "-lhsa-runtime64"
                if os.path.basename(rocr.lib_path) == _ROCR_LIB_NAMES[0]
                else rocr.lib_path
            ),
            "-o",
            out_path,
        ]
        _run_compile(cmd)
        with open(out_path, "rb") as f:
            lib_path = cache.put(f.read(), lib_name, binary=True)
    return os.path.dirname(lib_path)


def _find_msvc_cl() -> str:
    """Locate cl.exe for JIT compilation on Windows.

    Search order:
    1. cl.exe already on PATH (e.g. running from a VS Developer Command Prompt)
    2. vswhere.exe to discover Visual Studio installations, then use the
       latest MSVC toolset's Hostx64/x64/cl.exe

    Returns the absolute path to cl.exe.
    Raises Exception with setup instructions if MSVC cannot be found.
    """
    # 1. Already on PATH?
    cl_on_path = shutil.which("cl.exe") or shutil.which("cl")
    if cl_on_path:
        return cl_on_path

    # 2. Discover via vswhere
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    vswhere = os.path.join(
        program_files_x86,
        "Microsoft Visual Studio",
        "Installer",
        "vswhere.exe",
    )
    if os.path.isfile(vswhere):
        try:
            vs_path = (
                subprocess.check_output(
                    [
                        vswhere,
                        "-latest",
                        "-products",
                        "*",
                        "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-property",
                        "installationPath",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                .splitlines()[0]
            )
        except (subprocess.CalledProcessError, IndexError):
            vs_path = ""

        if vs_path:
            # Find the latest MSVC toolset version
            msvc_root = os.path.join(vs_path, "VC", "Tools", "MSVC")
            if os.path.isdir(msvc_root):
                versions = sorted(os.listdir(msvc_root), reverse=True)
                for ver in versions:
                    candidate = os.path.join(
                        msvc_root, ver, "bin", "Hostx64", "x64", "cl.exe"
                    )
                    if os.path.isfile(candidate):
                        return candidate

    raise RuntimeError(
        "MSVC compiler (cl.exe) not found. Triton-XDNA needs MSVC for JIT "
        "compilation of NPU dispatch code on Windows.\n"
        "Options:\n"
        "  1. Run from a 'x64 Native Tools Command Prompt for VS 2022'\n"
        "  2. Install Visual Studio 2022 with the 'Desktop development with C++' workload\n"
        "     (https://visualstudio.microsoft.com/)\n"
        "  3. Install the Build Tools for Visual Studio 2022\n"
        "     (https://visualstudio.microsoft.com/visual-cpp-build-tools/)"
    )


def _get_msvc_env(cl_path: str) -> dict:
    """Build the environment variables needed for cl.exe to find headers and libs.

    If INCLUDE and LIB are already set (e.g. from vcvars), returns the current
    environment unchanged.  Otherwise, derives them from the cl.exe location
    and the Windows SDK.
    """
    env = os.environ.copy()

    # If INCLUDE is already set, assume the environment is already configured
    if env.get("INCLUDE"):
        return env

    # Derive MSVC paths from cl.exe location:
    #   .../VC/Tools/MSVC/<ver>/bin/Hostx64/x64/cl.exe
    cl_dir = os.path.dirname(cl_path)  # .../bin/Hostx64/x64
    msvc_ver_dir = os.path.normpath(
        os.path.join(cl_dir, "..", "..", "..")
    )  # .../VC/Tools/MSVC/<ver>

    msvc_include = os.path.join(msvc_ver_dir, "include")
    msvc_lib = os.path.join(msvc_ver_dir, "lib", "x64")

    if not os.path.isdir(msvc_include):
        raise RuntimeError(
            f"Found cl.exe at {cl_path} but could not locate MSVC include "
            f"directory at {msvc_include}. Run from a VS Developer Command Prompt "
            "or ensure INCLUDE/LIB environment variables are set."
        )

    # Find Windows SDK
    sdk_root = os.environ.get(
        "WindowsSdkDir",
        os.path.join(
            os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
            "Windows Kits",
            "10",
        ),
    )
    sdk_version = os.environ.get("WindowsSDKVersion", "").rstrip("\\")
    if not sdk_version:
        # Auto-detect latest SDK version
        sdk_inc_root = os.path.join(sdk_root, "Include")
        if os.path.isdir(sdk_inc_root):
            versions = sorted(
                [d for d in os.listdir(sdk_inc_root) if d.startswith("10.")],
                reverse=True,
            )
            sdk_version = versions[0] if versions else ""

    include_paths = [msvc_include]
    lib_paths = [msvc_lib]

    if sdk_version:
        sdk_inc = os.path.join(sdk_root, "Include", sdk_version)
        sdk_lib = os.path.join(sdk_root, "Lib", sdk_version)
        for subdir in ["ucrt", "shared", "um"]:
            p = os.path.join(sdk_inc, subdir)
            if os.path.isdir(p):
                include_paths.append(p)
        for subdir in ["ucrt", "um"]:
            p = os.path.join(sdk_lib, subdir, "x64")
            if os.path.isdir(p):
                lib_paths.append(p)

    env["INCLUDE"] = ";".join(include_paths)
    env["LIB"] = ";".join(lib_paths)
    env["PATH"] = os.path.dirname(cl_path) + ";" + env.get("PATH", "")
    return env


def _get_aie_test_utils_path() -> str:
    custom = os.getenv("AIE_TEST_UTILS_DIR")
    if custom:
        return custom
    # aie.__file__ is <mlir_aie>/python/aie/__init__.py; three parents up is
    # the mlir_aie install root that contains runtime_lib/.
    path = (
        Path(aie.__file__).parent.parent.parent / "runtime_lib" / "x86_64" / "test_lib"
    )
    return path


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


@functools.lru_cache(maxsize=1)
def get_npu_device_info():
    # Cached: physical NPU devices do not change within a process, and this
    # is called on the per-dispatch hot path (via detect_npu_version). Without
    # caching, each kernel launch would spawn an `xrt-smi examine` subprocess.
    try:
        import re

        result = subprocess.run(
            ["xrt-smi", "examine"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        # Parse whatever was printed regardless of the exit status, and include
        # stderr: xrt-smi can list the device and still exit non-zero (e.g. a
        # driver/userspace version mismatch, or an unrelated sub-report that
        # failed). mlir-aie's iron_setup and lit helpers likewise ignore the
        # return code when no specific device was demanded.
        output = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            logger.debug(
                "xrt-smi examine exited %d: %s", result.returncode, result.stderr
            )

        # Match either one or two pipes with optional whitespace around them
        device_pattern = re.compile(
            r"\[(?P<bdf>[0-9a-fA-F:.]+)\]\s*\|{1,2}\s*(?P<name>.+?)\s*\|"
        )

        matches = device_pattern.findall(output)

        devices = []
        for bdf, name in matches:
            devices.append({"bdf": bdf, "name": name.strip()})

        if not devices:
            # The `Device(s) Present` table is a presentation layer: its shape
            # changes between XRT releases. Fall back to scanning the whole
            # output for a known model name, as mlir-air's XRTBackend does.
            for keywords in NPU_MODELS.values():
                for kw in keywords:
                    if kw.lower() in output.lower():
                        devices.append({"bdf": "unknown", "name": kw})
                        return devices

        return devices

    except Exception as e:
        logger.exception("Unexpected error during NPU device detection")
        return []


# Device name mappings aligned with mlir-aie (lit_config_helpers.py, iron_setup.py)
NPU_MODELS = {
    "npu1": ["npu1", "Phoenix"],
    "npu2": ["npu4", "Strix", "npu5", "Strix Halo", "npu6", "Krackan"],
}

# Device names reported by the HSA AIE agent (HSA_AGENT_INFO_NAME), which names
# the AIE generation directly rather than the marketing name xrt-smi prints.
NPU_AGENT_NAMES = {"aie2": "npu1", "aie2p": "npu2"}


@functools.lru_cache(maxsize=1)
def _pyxrt_device_name() -> str:
    """Return the device name XRT reports for device 0.

    This asks the XRT API directly instead of scraping ``xrt-smi examine``
    text, which is what mlir-aie's runtime does (``XRTHostRuntime`` reads
    ``pyxrt.xrt_info_device.name`` and only shells out to xrt-smi as debug
    output when opening the device fails). The CLI table is a human-facing
    presentation layer: its columns change between XRT releases, and it can
    come up empty on a host whose sysfs entries it cannot render even though
    the device node itself opens fine.
    """
    import pyxrt

    device = pyxrt.device(0)
    name = device.get_info(pyxrt.xrt_info_device.name)
    # Release the device before returning: this query runs during launcher
    # construction, and holding an open handle would contend with the actual
    # kernel dispatch that follows.
    del device
    return name


def _detect_npu_version_pyxrt() -> str:
    """Identify the NPU generation from the name XRT reports for device 0."""
    name = _pyxrt_device_name()
    for version, keywords in NPU_MODELS.items():
        if any(kw.lower() in name.lower() for kw in keywords):
            return version
    raise UnsupportedNPUDeviceError(
        f"XRT reports device name {name!r}, which this backend does not "
        f"recognize (known: {dict(NPU_MODELS)}). Refusing to guess the NPU "
        "generation: compiling for the wrong one wedges the device until the "
        "driver timeout fires. Set AMD_TRITON_NPU_TARGET (or "
        "npu_config.target) to 'npu1'/'npu2' to override, or add the name to "
        "NPU_MODELS if it is supported."
    )


@functools.lru_cache(maxsize=1)
def _hsa_agent_name() -> str:
    """Device name reported by the HSA AIE agent, e.g. ``"aie2p"``.

    Loads the shared HSA runtime (building it if needed) and calls its
    ``triton_npu_hsa_agent_name``. Deliberately reuses that library rather than
    dlopen-ing ROCR separately from Python: the launcher loads the very same
    ``.so``, so the process keeps a single ``hsa_init`` and a single HsaRuntime
    singleton, which the AIE agent requires (``QUEUES_MAX == 1``).
    """
    import ctypes

    include_dir = os.path.join(Path(__file__).resolve().parent, "include")
    lib_dir = _build_hsa_runtime_lib(include_dir, _get_rocr_install())
    lib = ctypes.CDLL(os.path.join(lib_dir, "libtriton_npu_hsa.so"))

    fn = lib.triton_npu_hsa_agent_name
    fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
    fn.restype = ctypes.c_int

    buf = ctypes.create_string_buffer(64)
    err = ctypes.create_string_buffer(512)
    if fn(buf, len(buf), err, len(err)) != 0:
        raise RuntimeError(err.value.decode("utf-8", errors="replace"))
    return buf.value.decode("utf-8", errors="replace")


class UnsupportedNPUDeviceError(RuntimeError):
    """The AIE agent named a device generation this backend does not know.

    Kept distinct from the errors raised when the agent cannot be *queried*
    (no ROCR, no agent, runtime library unbuildable). A failed query says
    nothing about the hardware, so falling back to xrt-smi is reasonable; an
    answer we do not recognize is a definite negative, and continuing would
    mean compiling for a guessed generation.
    """


def _detect_npu_version_hsa() -> str:
    """Identify the NPU generation from the HSA AIE agent."""
    name = _hsa_agent_name().strip().lower()
    version = NPU_AGENT_NAMES.get(name)
    if version is None:
        raise UnsupportedNPUDeviceError(
            f"The HSA AIE agent reports device name {name!r}, which this "
            f"backend does not recognize (known: {sorted(NPU_AGENT_NAMES)}). "
            "Refusing to guess the NPU generation: compiling for the wrong one "
            "wedges the device until the driver timeout fires. Set "
            "AMD_TRITON_NPU_TARGET (or npu_config.target) to 'npu1'/'npu2' to "
            "override, or add the name to NPU_AGENT_NAMES if it is supported."
        )
    return version


def detect_npu_version(runtime=None):
    """Map known device names to internal NPU version strings.

    If ``npu_config.target`` is set (programmatically or via the
    ``AMD_TRITON_NPU_TARGET`` env var), use that value directly
    (must be 'npu1' or 'npu2'). This enables cross-compilation
    without local NPU hardware.

    ``runtime`` is the caller's launch runtime ("xrt" or "hsa"), defaulting to
    ``npu_config.runtime``. Under HSA the generation is read from the AIE agent
    first, so an HSA-only host does not need XRT installed merely to identify
    the device. If that *query* fails the search continues with xrt-smi, but an
    agent that answers with an unrecognized device name raises
    ``UnsupportedNPUDeviceError`` immediately rather than letting a second
    source supply a generation for hardware we do not know.
    """
    target = npu_config.target
    if target is not None:
        if target not in NPU_MODELS:
            raise RuntimeError(
                f"Invalid target='{target}' from npu_config.target "
                f"(or AMD_TRITON_NPU_TARGET). "
                f"Supported values: {list(NPU_MODELS.keys())}"
            )
        return target

    if runtime is None:
        runtime = npu_config.runtime
    hsa_error = None
    if runtime == "hsa":
        try:
            return _detect_npu_version_hsa()
        except UnsupportedNPUDeviceError:
            # The agent answered and we did not recognize it. Propagate rather
            # than consulting xrt-smi: a second opinion here would only produce
            # a generation for hardware this backend has never been validated
            # against, which is exactly the guess we are refusing to make.
            raise
        except Exception as e:
            # The agent could not be queried at all (no ROCR, no AIE agent, the
            # runtime library would not build). That says nothing about the
            # hardware, so let xrt-smi try.
            hsa_error = e
            logger.debug("HSA device detection failed, trying xrt-smi: %s", e)

    # Ask XRT for the device name before falling back to parsing xrt-smi's
    # table. Same rule as the HSA path above: a device that answers with a
    # name we do not recognize propagates immediately, while a failure to
    # query at all (no pyxrt on the path, device busy) says nothing about the
    # hardware and lets the next source try.
    pyxrt_error = None
    try:
        return _detect_npu_version_pyxrt()
    except UnsupportedNPUDeviceError:
        raise
    except Exception as e:
        pyxrt_error = e
        logger.debug("pyxrt device detection failed, trying xrt-smi: %s", e)

    devices = get_npu_device_info()
    for device in devices:
        name = device["name"]
        for version, keywords in NPU_MODELS.items():
            if any(kw.lower() in name.lower() for kw in keywords):
                return version
    if not devices:
        msg = "No NPU devices found. Ensure XRT is installed and xrt-smi is available."
        if pyxrt_error is not None:
            msg += f" Querying the XRT device directly also failed: {pyxrt_error}"
        if hsa_error is not None:
            msg += f" Detection via the HSA AIE agent also failed: {hsa_error}"
        raise RuntimeError(msg)
    device_names = [d["name"] for d in devices]
    raise RuntimeError(
        f"Unsupported NPU device(s): {device_names}. "
        f"Supported models: {dict(NPU_MODELS)}"
    )


def _get_output_format(runtime=None):
    """Determine the output format for the NPU backend.

    ``runtime`` is the caller's launch runtime ("xrt" or "hsa"); it defaults to
    ``npu_config.runtime`` for callers (e.g. multilaunch) that don't thread a
    driver-bound runtime through. Passing it explicitly is the single source of
    truth for the launcher path, so the format never disagrees with the runtime.

    Checks ``npu_config.output_format`` first (which itself falls back to
    the ``AMD_TRITON_NPU_OUTPUT_FORMAT`` env var).
    If not set, defaults to "elf" on npu2 and "xclbin" on npu1.
    ELF format is only supported on npu2 (AIE2P) devices.

    Under the HSA runtime the artifact must be PDI + insts, since the ROCR AIE
    dispatch path consumes a raw ``aie.pdi`` and its ``insts.bin`` sidecar. In
    that case "pdi" is forced and an explicit ELF/xclbin request is rejected.
    """
    if runtime is None:
        runtime = npu_config.runtime
    npu_version = detect_npu_version(runtime)
    configured_format = npu_config.output_format
    if runtime == "hsa":
        if configured_format is not None and configured_format != "pdi":
            raise RuntimeError(
                f"ROCR requires the 'pdi' output format, but "
                f"output_format={configured_format!r} was requested. Unset "
                "AMD_TRITON_NPU_OUTPUT_FORMAT (or npu_config.output_format), or "
                "set it to 'pdi'."
            )
        return "pdi"
    if configured_format is not None:
        if configured_format == "elf" and npu_version == "npu1":
            raise RuntimeError(
                "ELF output format is not supported on npu1 (AIE2) devices. "
                "Unset or change AMD_TRITON_NPU_OUTPUT_FORMAT, or set "
                "npu_config.output_format to 'xclbin', 'pdi', or None."
            )
        return configured_format
    # Auto-detect: ELF for npu2, xclbin for npu1.
    return "elf" if npu_version == "npu2" else "xclbin"


def _extract_elf_kernel_name(config_json_path):
    """Extract the ELF kernel name from full_elf_config.json.

    The kernel name for XRT is "{kernel_name}:{instance_id}".
    Looks for the "main" kernel entry (the runtime dispatch kernel)
    and uses its first instance ID.
    """
    with open(config_json_path) as f:
        config = json.load(f)
    for kernel in config["xrt-kernels"]:
        if kernel["name"] == "main" and kernel.get("instance"):
            instance_id = kernel["instance"][0]["id"]
            return f"main:{instance_id}"
    # Fallback: use the last kernel entry (which is typically "main")
    last_kernel = config["xrt-kernels"][-1]
    instance_id = last_kernel["instance"][0]["id"]
    return f"{last_kernel['name']}:{instance_id}"


def _inject_transform_library(user_script):
    """
    Process library references in a user transform script.

    Two mechanisms:
    1. transform.include calls are expanded inline (parameter substitution,
       SSA renaming) to avoid segfaults in mlir-air's transform interpreter
       when resolving transform.include across region boundaries.
    2. foreach_match @name symbol references are resolved by injecting the
       referenced named_sequence definitions into the module (these cannot
       be inlined because foreach_match resolves symbols at runtime).

    Args:
        user_script: The user's transform script as a string

    Returns:
        str: The processed script
    """
    has_includes = "transform.include" in user_script
    has_foreach_match = "foreach_match" in user_script
    if not has_includes and not has_foreach_match:
        return user_script

    # Load library content from transform_library/ directory
    lib_dir = os.path.join(os.path.dirname(__file__), "transform_library")
    if not os.path.isdir(lib_dir):
        return user_script
    parts = []
    for fname in sorted(os.listdir(lib_dir)):
        if fname.endswith(".mlir"):
            with open(os.path.join(lib_dir, fname), "r") as f:
                parts.append(f.read())
    lib_content = "\n".join(parts)

    import re

    # Parse all named sequences: full text (for injection) and decomposed (for inlining)
    full_seq_pattern = re.compile(
        r"((?://[^\n]*\n)*"
        r"transform\.named_sequence\s+@(\w+)\s*\([^)]*\)"
        r"(?:\s*->\s*!transform\.any_op)?"
        r"\s*\{.*?\n\})",
        re.DOTALL,
    )
    full_sequences = {}
    for m in full_seq_pattern.finditer(lib_content):
        full_sequences[m.group(2)] = m.group(1)

    # Parse inlinable sequences (readonly or consumed param, for transform.include)
    inline_seq_pattern = re.compile(
        r"transform\.named_sequence\s+@(\w+)\s*\(\s*"
        r"%(\w+)\s*:\s*!transform\.any_op\s*\{transform\.(?:readonly|consumed)\}\s*\)"
        r"(\s*->\s*!transform\.any_op)?"
        r"\s*\{(.*?)\n\}",
        re.DOTALL,
    )
    sequences = {}
    for match in inline_seq_pattern.finditer(lib_content):
        name = match.group(1)
        param = match.group(2)
        has_result = match.group(3) is not None
        body = match.group(4)
        sequences[name] = (param, body, has_result)

    if not sequences and not full_sequences:
        return user_script

    # Inline transform.include calls to avoid mlir-air segfaults
    include_pattern = re.compile(
        r"(?:(%\w+)\s*=\s*)?"
        r"transform\.include\s+@(\w+)\s+"
        r"failures\(\w+\)\s*"
        r"\((%\w+)\)\s*"
        r":\s*\(!transform\.any_op\)\s*->\s*"
        r"(?:!transform\.any_op|\(\s*\))"
    )

    _counter = [0]

    def _expand(text, depth=0):
        if depth > 20 or "transform.include" not in text:
            return text

        def _replace_include(m):
            result_var = m.group(1)
            seq_name = m.group(2)
            actual_arg = m.group(3)

            if seq_name not in sequences:
                return m.group(0)

            param, body, has_result = sequences[seq_name]
            expanded = body.replace(f"%{param}", actual_arg)

            yield_match = re.search(
                r"transform\.yield(?:\s+(%\w+)\s*:\s*!transform\.any_op)?",
                expanded,
            )
            if yield_match:
                yielded_var = yield_match.group(1)
                expanded = expanded[: yield_match.start()].rstrip()
                if result_var and yielded_var:
                    expanded = expanded.replace(yielded_var, result_var)

            suffix = f"_lib{_counter[0]}"
            _counter[0] += 1
            local_vars = set(re.findall(r"%(\w+)", expanded))
            actual_name = actual_arg.lstrip("%")
            result_name = result_var.lstrip("%") if result_var else ""
            skip = {actual_name, result_name, "__", ""}
            for var in local_vars:
                if var not in skip and not var.startswith("_lib"):
                    expanded = re.sub(
                        rf"(?<!\w)%{re.escape(var)}(?!\w)",
                        f"%{var}{suffix}",
                        expanded,
                    )

            return expanded

        text = include_pattern.sub(_replace_include, text)
        return _expand(text, depth + 1)

    result = _expand(user_script) if has_includes else user_script

    # Inject named sequences referenced by foreach_match (symbol references
    # that cannot be inlined — they must exist as definitions in the module).
    if has_foreach_match or "foreach_match" in result:
        all_refs = set(re.findall(r"@(\w+)", result))
        all_refs.discard("__transform_main")
        # Transitively resolve dependencies
        needed = set()
        worklist = [n for n in all_refs if n in full_sequences]
        while worklist:
            name = worklist.pop()
            if name in needed:
                continue
            needed.add(name)
            for dep in re.findall(r"@(\w+)", full_sequences[name]):
                if dep in full_sequences and dep not in needed:
                    worklist.append(dep)
        # Inject definitions for all unresolved @name references
        # (matchers/actions referenced by foreach_match, plus their deps)
        if needed:
            module_marker = "module attributes {transform.with_named_sequence} {"
            idx = result.find(module_marker)
            if idx != -1:
                insert_pos = idx + len(module_marker)
                injection = "\n\n".join(
                    full_sequences[n] for n in full_sequences if n in needed
                )
                result = (
                    result[:insert_pos]
                    + "\n\n"
                    + injection
                    + "\n\n"
                    + result[insert_pos:]
                )

    return result


def _detect_matmul(asm_src_text):
    """Detect a single plain matmul in the TritonShared IR.

    Returns a dict {m, k, n, in_elem, out_elem} when the module contains
    exactly one ``linalg.matmul`` and no linalg compute ops other than
    ``fill`` (fused epilogues are out of scope). Returns None otherwise, in
    which case the caller keeps the built-in default tiling.
    """
    import re

    # Bail if any linalg op other than matmul / fill is present (fusion).
    linalg_ops = set(re.findall(r"linalg\.(\w+)", asm_src_text))
    if not linalg_ops.issubset({"matmul", "fill"}):
        return None

    matmuls = re.findall(
        r"linalg\.matmul\b.*?"
        r"ins\([^:]*:\s*tensor<(\d+)x(\d+)x(\w+)>,\s*tensor<(\d+)x(\d+)x(\w+)>\)\s*"
        r"outs\([^:]*:\s*tensor<(\d+)x(\d+)x(\w+)>\)",
        asm_src_text,
        flags=re.DOTALL,
    )
    if len(matmuls) != 1:
        return None

    m, k, in_elem, k2, n, in_elem2, m2, n2, out_elem = matmuls[0]
    m, k, n, k2, m2, n2 = int(m), int(k), int(n), int(k2), int(m2), int(n2)
    # Defensive shape consistency: A[MxK] * B[KxN] = C[MxN].
    if k != k2 or m != m2 or n != n2 or in_elem != in_elem2:
        return None
    return {"m": m, "k": k, "n": n, "in_elem": in_elem, "out_elem": out_elem}


def _matmul_transform_params(info, npu_version):
    """Derive generate_matmul_transform kwargs from a detected matmul.

    Returns a kwargs dict (l1_m, l1_n, l2_k, pack_sizes, accum_type,
    contract_input_type, bf16_emulation) or None if the shape/dtype cannot
    be mapped to a valid schedule.
    """
    if info is None:
        return None

    # dtype -> accumulator / contract-input / bf16-emulation rules.
    in_elem, out_elem = info["in_elem"], info["out_elem"]
    if in_elem == "f32":
        accum_type, contract_input_type, bf16_emulation = "f32", "bf16", True
    elif in_elem == "bf16":
        accum_type, contract_input_type, bf16_emulation = "f32", None, False
    elif in_elem == "i8":
        # No contract-input cast: the AIEVec lowering recovers signedness from
        # the arith.extsi it peels off the operands, so casting to i16 first
        # only obscures it.
        accum_type, contract_input_type, bf16_emulation = "i32", None, False
    else:
        return None

    # Pack sizes are the hardware MAC shape, so they depend on the element type
    # as well as the target; herd caps depend only on the array. Note
    # pack_sizes is ordered (M, N, K) while a MAC shape is written (M, K, N) --
    # the two coincide only when the shape is cubic, which is why npu2 does not
    # need to distinguish them.
    if npu_version == "npu2":
        # AIE2P has an 8x8x8 MAC for both bf16 and i8.
        pack_m, pack_n, pack_k = 8, 8, 8
        cap_m, cap_n = 4, 4
    else:  # npu1 (Phoenix / AIE2, 4x2 array)
        # IsValidAIE2MatMulShapeAndType lists exactly one shape per input type
        # on AIE2: bf16 is 4x8 x 8x4 -> 4x4, i8 is 4x8 x 8x8 -> 4x8.
        pack_m, pack_k = 4, 8
        pack_n = 8 if in_elem == "i8" else 4
        cap_m, cap_n = 4, 2

    m, k, n = info["m"], info["k"], info["n"]
    # Pack requires exact divisibility of each dim by its pack size.
    if m % pack_m or n % pack_n or k % pack_k:
        return None

    def _largest_divisor_leq(x, cap):
        for d in range(min(x, cap), 0, -1):
            if x % d == 0:
                return d
        return 1

    # Distribute the packed M/N outer dims across the core array (herd).
    herd_m = _largest_divisor_leq(m // pack_m, cap_m)
    herd_n = _largest_divisor_leq(n // pack_n, cap_n)
    l1_m = m // herd_m
    l1_n = n // herd_n

    # L2 K tile: largest multiple of pack_k dividing K, capped at 2*pack_k.
    # A small K tile keeps L2 buffering conservative; larger tiles were
    # observed to miscompute padded boundary tiles under the
    # air-split-launch-for-padding path.
    l2k_cap = 2 * pack_k
    l2_k = pack_k
    for v in range(min(k, l2k_cap) - (min(k, l2k_cap) % pack_k), 0, -pack_k):
        if k % v == 0:
            l2_k = v
            break

    return {
        "l1_m": l1_m,
        "l1_n": l1_n,
        "l2_k": l2_k,
        "pack_sizes": (pack_m, pack_n, pack_k),
        "accum_type": accum_type,
        "contract_input_type": contract_input_type,
        "bf16_emulation": bf16_emulation,
    }


def _get_transform_ir_string(matmul_info=None):
    """
    Get the transform IR string for tiling operations.

    Priority: (1) a user-supplied script via
    ``npu_config.transform_tiling_script`` / ``AIR_TRANSFORM_TILING_SCRIPT``;
    (2) an auto-generated matmul schedule when ``matmul_info`` (kwargs for
    ``generate_matmul_transform``) is provided; (3) the built-in default
    tiling IR string.

    If the script uses `transform.include`, the shared transform library
    (transform_library.mlir) is automatically injected.

    Returns:
        str: The transform IR string to use for tiling
    """
    custom_script_path = npu_config.transform_tiling_script

    if custom_script_path:
        if not os.path.isfile(custom_script_path):
            raise FileNotFoundError(
                f"transform_tiling_script / AIR_TRANSFORM_TILING_SCRIPT is set to "
                f"'{custom_script_path}' but the file was not found "
                f"(cwd: {os.getcwd()}). Use an absolute path or run from the "
                f"directory containing the script."
            )
        with open(custom_script_path, "r") as f:
            logger.debug("Using custom tiling script from: %s", custom_script_path)
            user_script = f.read()
        return _inject_transform_library(user_script)

    # Auto-generated matmul schedule (parameters derived from the IR).
    if matmul_info is not None:
        from .matmul_transform import generate_matmul_transform

        logger.debug("Auto-generating matmul transform with params: %s", matmul_info)
        script = generate_matmul_transform(**matmul_info)
        try:
            air_proj_path = npu_config.air_project_path
            os.makedirs(air_proj_path, exist_ok=True)
            with open(os.path.join(air_proj_path, "auto_transform.mlir"), "w") as f:
                f.write(script)
        except OSError:
            pass
        return script

    # Default hardcoded transform IR string
    matmul_tiling_size_l1_m = 32
    matmul_tiling_size_l1_n = 32
    matmul_tiling_size_l1_k = 32
    elemwise_tiling_size_l1_m = 32
    elemwise_tiling_size_l1_n = 32

    return f"""
        module attributes {{transform.with_named_sequence}} {{
          transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
                %mul = transform.structured.match ops{{["linalg.mul"]}} in %arg1  : (!transform.any_op) -> !transform.any_op
                %mul_1, %loop = transform.air.linalg_tile %mul [{elemwise_tiling_size_l1_m}, {elemwise_tiling_size_l1_n}]
                transform.air.linalg_promote %mul_1 {{"operands_to_promote"=[2], "memory_space"="L1"}} : (!transform.any_op) -> !transform.any_op
                transform.air.linalg_promote %mul_1 {{"operands_to_promote"=[0,1], "memory_space"="L1"}} : (!transform.any_op) -> !transform.any_op

                %add = transform.structured.match ops{{["linalg.add"]}} in %arg1  : (!transform.any_op) -> !transform.any_op
                %add_1, %add_loop = transform.air.linalg_tile %add [{elemwise_tiling_size_l1_m}, {elemwise_tiling_size_l1_n}]
                transform.air.linalg_promote %add_1 {{"operands_to_promote"=[2], "memory_space"="L1"}} : (!transform.any_op) -> !transform.any_op
                transform.air.linalg_promote %add_1 {{"operands_to_promote"=[0,1], "memory_space"="L1"}} : (!transform.any_op) -> !transform.any_op

                %matmul = transform.structured.match ops{{["linalg.matmul"]}} in %arg1  : (!transform.any_op) -> !transform.any_op
                %fill = transform.structured.match ops{{["linalg.fill"]}} in %arg1  : (!transform.any_op) -> !transform.any_op
                %matmul_1, %matmul_loop = transform.air.linalg_tile %matmul [{matmul_tiling_size_l1_m}, {matmul_tiling_size_l1_n}]
                %fill_1 = transform.air.fuse_into_containing_op %fill into %matmul_loop : (!transform.any_op, !transform.any_op) -> !transform.any_op
                transform.air.linalg_promote %fill_1 {{"operands_to_promote"=[1], "memory_space"="L1"}} : (!transform.any_op) -> !transform.any_op
                transform.air.linalg_promote %matmul_1 {{"operands_to_promote"=[2], "memory_space"="L1"}} : (!transform.any_op) -> !transform.any_op
                %matmul_2, %reduction_loop = transform.air.linalg_tile %matmul_1 [0, 0, {matmul_tiling_size_l1_k}]
                transform.air.linalg_promote %matmul_2 {{"operands_to_promote"=[0,1], "memory_space"="L1"}} : (!transform.any_op) -> !transform.any_op
            transform.yield
          }}
        }}
        """


def _ttshared_to_air(mod, gridX, gridY, gridZ, actual_sizes=None, matmul_info=None):
    # Get Triton-Shared-MLIR as string
    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = os.path.join(tmpdir, "airinput.mlir")
        air_opt_path = _get_air_opt_path()
        # MLIR-AIR compilation step 1: mapping grid to air.launch
        pipeline = (
            "builtin.module("
            + ",".join(
                [
                    "air-resolve-tensor-opoperand-conflicts",
                    "air-override-memref-memory-space{scope=func memory-space=1}",
                ]
            )
            + ")"
        )
        air_context = air.ir.Context()
        air_module = Module.parse(mod, context=air_context)
        pm = air.passmanager.PassManager.parse(pipeline, context=air_context)
        pm.run(air_module.operation)
        # MLIR-AIR compilation step 2: tiling the launch body
        transform_ir_string = _get_transform_ir_string(matmul_info=matmul_info)
        transform_ir = Module.parse(transform_ir_string, context=air_context)
        run_transform(transform_ir, air_module)
        # MLIR-AIR compilation step 3: converting to AIR
        wrap_params = f"loop-bounds={gridX},{gridY},{gridZ}"
        if actual_sizes:
            wrap_params += f" actual-sizes={actual_sizes}"
        pipeline = (
            "builtin.module("
            + ",".join(
                [
                    f"func.func(air-wrap-func-with-parallel{{{wrap_params}}})",
                    "air-par-to-launch{depth=0 has-air-segment=true}",
                    "canonicalize",
                    "cse",
                    "air-copy-to-dma",
                ]
            )
            + ")"
        )
        pm = air.passmanager.PassManager.parse(pipeline, context=air_context)
        pm.run(air_module.operation)
        with open(dst_path, "w") as f:
            f.write(str(air_module))
        _dump_ir_if_needed([dst_path])
        return air_module


def _generate_launcher(constants, signature, kernel_name):
    arg_decls = ", ".join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    args_format = "".join([format_of(extracted_type(ty)) for ty in signature.values()])
    format = "iiiOOOO" + args_format
    args_list = (
        ", " + ", ".join(f"&_arg{i}" for i, ty in signature.items())
        if len(signature) > 0
        else ""
    )

    kernel_arg_decls = ", ".join(
        ty_to_cpp(ty) if ty[0] != "*" else f"int64_t, void*"
        for i, ty in signature.items()
        if ty != "constexpr"
    )
    kernel_arg_decls += ", " if kernel_arg_decls else ""

    kernel_parameters = ", ".join(
        f"static_cast<{ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"0, &ptr_arg{i}"
        for i, ty in signature.items()
        if ty != "constexpr"
    )
    kernel_parameters += ", " if kernel_parameters else ""

    global autotune_time

    return f"""
#include <assert.h>
#include <fstream>
#include <iostream>
#include <stdbool.h>
#include <Python.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <sstream>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

static char aie_path[1024] = {{0}};
static char insts_path[1024] = {{0}};

static PyObject* py_set_paths(PyObject* self, PyObject* args) {{
    const char* aie;
    const char* insts;

    if (!PyArg_ParseTuple(args, "ss", &aie, &insts)) {{
        return NULL;
    }}

    strncpy(aie_path, aie, sizeof(aie_path) - 1);
    strncpy(insts_path, insts, sizeof(insts_path) - 1);
    aie_path[sizeof(aie_path) - 1] = '\\0';
    insts_path[sizeof(insts_path) - 1] = '\\0';

    Py_RETURN_NONE;
}}

// Call to XRT goes here:
static void _launch(int gridX, int gridY, int gridZ, {', '.join(f"long size{i}" for i, ty in signature.items() if i not in constants and ty[0]=="*")}, {arg_decls}) {{
  if (gridX*gridY*gridZ > 0) {{
    try {{

    // PDI artifacts target an alternative (non-XRT) runtime and cannot be
    // loaded through the XRT xclbin API below. Fail early with a clear
    // message pointing the user to their target runtime.
    {{
        std::string _aie_path(aie_path);
        if (_aie_path.size() >= 4 &&
            _aie_path.compare(_aie_path.size() - 4, 4, ".pdi") == 0)
            throw std::runtime_error(
                std::string("PDI artifact '") + aie_path +
                "' targets an alternative (non-XRT) runtime and cannot be "
                "launched via XRT. Pass the .pdi and its .insts.bin sidecar to "
                "your target runtime directly.");
    }}

    // Load instruction binary (inlined, replaces test_utils dependency)
    std::vector<uint32_t> instr_v;
    {{
        std::ifstream instr_file(insts_path, std::ios::binary);
        if (!instr_file.is_open())
            throw std::runtime_error(std::string("Failed to open instr file: ") + insts_path);
        instr_file.seekg(0, std::ios::end);
        std::streamsize fsize = instr_file.tellg();
        instr_file.seekg(0, std::ios::beg);
        if (fsize % 4 != 0)
            throw std::runtime_error("Instruction file size is not a multiple of 4 bytes");
        instr_v.resize(fsize / 4);
        if (!instr_file.read(reinterpret_cast<char*>(instr_v.data()), fsize))
            throw std::runtime_error("Failed to read instruction file");
    }}

    int verbosity = {1 if npu_config.debug else 0};
    if (verbosity >= 1)
        std::cout << "Sequence instr count: " << instr_v.size() << std::endl;

    // Start the XRT test code
    // Get a device handle
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    // Load the xclbin
    if (verbosity >= 1)
        std::cout << "Loading xclbin." << std::endl;
    auto xclbin = xrt::xclbin(std::string(aie_path));

    if (verbosity >= 1)
        std::cout << "Kernel opcode: " << "MLIR_AIE" << std::endl;
    std::string Node = "MLIR_AIE";

    // Get the kernel from the xclbin
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                                    [Node, verbosity](xrt::xclbin::kernel &k) {{
                                    auto name = k.get_name();
                                    if (verbosity >= 1) std::cout << "Name: " << name << std::endl;
                                    return name.rfind(Node, 0) == 0;
                                    }});
    auto kernelName = xkernel.get_name();

    if (verbosity >= 1)
        std::cout << "Registering xclbin." << std::endl;

    device.register_xclbin(xclbin);

    // get a hardware context
    if (verbosity >= 1)
        std::cout << "Getting hardware context." << std::endl;
    xrt::hw_context context(device, xclbin.get_uuid());

    // get a kernel handle
    if (verbosity >= 1)
        std::cout << "Getting handle to kernel:" << kernelName << std::endl;
    auto kernel = xrt::kernel(context, kernelName);

    // get instruction sequence
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

    {' '.join(f'auto bo_{i} = xrt::bo(device, size{i}, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id({i+3}));' for i, ty in signature.items() if i not in constants and ty[0] == "*")}

    if (verbosity >= 1)
        std::cout << "Writing data into buffer objects." << std::endl;
    {' '.join(f'void *buf{i} = bo_{i}.map<void *>(); memcpy(buf{i}, arg{i}, size{i});' for i, ty in signature.items() if i not in constants and ty[0] == "*")}

    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    {' '.join(f'bo_{i}.sync(XCL_BO_SYNC_BO_TO_DEVICE);' for i, ty in signature.items() if i not in constants and ty[0] == "*")}

    if (verbosity >= 1)
        std::cout << "Running Kernel." << std::endl;
    unsigned int opcode = 3;
    {'auto start = std::chrono::high_resolution_clock::now();' if autotune_time else ''}
    auto run = kernel(opcode, bo_instr, instr_v.size(), {','.join(f'bo_{i}' for i, ty in signature.items() if i not in constants and ty[0] == "*")});
    // Throws unless the command reaches ERT_CMD_STATE_COMPLETED; an aborted run
    // must not fall through to the readback below and return zeros as a result.
    run.wait2();
    {'auto stop = std::chrono::high_resolution_clock::now(); float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();' if autotune_time else ''}

    {'std::ofstream file("data.txt"); file << npu_time << std::endl; file.close();' if autotune_time else ''}

    if (verbosity >= 1)
        std::cout << "Copying results." << std::endl;
    // TODO: Assuming the last tensor is the only output tensor.
    bo_{next((i for i, ty in reversed(signature.items()) if i not in constants and ty[0] == "*"), None)}.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(arg{next((i for i, ty in reversed(signature.items()) if i not in constants and ty[0] == "*"), None)}, buf{next((i for i, ty in reversed(signature.items()) if i not in constants and ty[0] == "*"), None)}, size{next((i for i, ty in reversed(signature.items()) if i not in constants and ty[0] == "*"), None)});

    if (verbosity >= 1)
        std::cout << "Launch finished." << std::endl;

    }} catch (const std::exception& e) {{
        std::string msg = std::string("XRT runtime error: ") + e.what();
        PyErr_SetString(PyExc_RuntimeError, msg.c_str());
    }}
  }}
}}

#include "npu_dispatch_common.h"

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  {"; ".join([f"long nelem{i} = getNumElements(_arg{i}); long ebytes{i} = getElementSizeInBytes(_arg{i}); if (nelem{i} == -1 || ebytes{i} == -1) return NULL; long tensor_volume{i} = nelem{i} * ebytes{i};" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, {', '.join(f"tensor_volume{i}" for i, ty in signature.items() if i not in constants and ty[0]=="*")}, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{"set_paths", py_set_paths, METH_VARARGS, "Set paths to aie.bin and insts.bin"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__npu_dispatch\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___npu_dispatch(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""


def _generate_elf_launcher(constants, signature, kernel_name):
    """Generate C++ launcher code using XRT ELF APIs (for NPU2/AIE2P only)."""
    arg_decls = ", ".join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    args_format = "".join([format_of(extracted_type(ty)) for ty in signature.values()])
    format = "iiiOOOO" + args_format
    args_list = (
        ", " + ", ".join(f"&_arg{i}" for i, ty in signature.items())
        if len(signature) > 0
        else ""
    )

    global autotune_time

    # Collect pointer (tensor) args excluding constants
    ptr_args = [
        (i, ty) for i, ty in signature.items() if i not in constants and ty[0] == "*"
    ]
    last_ptr_idx = next(
        (
            i
            for i, ty in reversed(signature.items())
            if i not in constants and ty[0] == "*"
        ),
        None,
    )

    # Build set_arg lines for kernel invocation
    set_arg_lines = "\n    ".join(
        f"run.set_arg({idx}, bo_{i});" for idx, (i, ty) in enumerate(ptr_args)
    )

    return f"""
#include <assert.h>
#include <fstream>
#include <iostream>
#include <stdbool.h>
#include <Python.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdexcept>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>

static char elf_path[1024] = {{0}};
static char elf_kernel_name[256] = {{0}};

static PyObject* py_set_paths(PyObject* self, PyObject* args) {{
    const char* elf;
    const char* kname;

    if (!PyArg_ParseTuple(args, "ss", &elf, &kname)) {{
        return NULL;
    }}

    strncpy(elf_path, elf, sizeof(elf_path) - 1);
    elf_path[sizeof(elf_path) - 1] = '\\0';
    strncpy(elf_kernel_name, kname, sizeof(elf_kernel_name) - 1);
    elf_kernel_name[sizeof(elf_kernel_name) - 1] = '\\0';

    Py_RETURN_NONE;
}}

// ELF-based XRT launch:
static void _launch(int gridX, int gridY, int gridZ, {', '.join(f"long size{i}" for i, ty in ptr_args)}, {arg_decls}) {{
  if (gridX*gridY*gridZ > 0) {{
    try {{

    int verbosity = {1 if npu_config.debug else 0};

    // Get a device handle
    unsigned int device_index = 0;
    if (verbosity >= 1)
        std::cout << "Opening device " << device_index << "..." << std::endl;
    auto device = xrt::device(device_index);

    // Load the ELF
    if (verbosity >= 1)
        std::cout << "Loading ELF: " << elf_path << std::endl;
    xrt::elf ctx_elf{{elf_path}};

    if (verbosity >= 1)
        std::cout << "Creating hw_context..." << std::endl;
    xrt::hw_context context = xrt::hw_context(device, ctx_elf);

    // Kernel name from ELF config (e.g., "main:vecadd")
    std::string kernelName = elf_kernel_name;
    if (verbosity >= 1)
        std::cout << "Kernel name: " << kernelName << std::endl;
    auto kernel = xrt::ext::kernel(context, kernelName);

    // Create buffer objects using xrt::ext::bo (no group_id needed)
    {' '.join(f'xrt::bo bo_{i} = xrt::ext::bo{{device, (size_t)size{i}}};' for i, ty in ptr_args)}

    if (verbosity >= 1)
        std::cout << "Writing data into buffer objects." << std::endl;
    {' '.join(f'void *buf{i} = bo_{i}.map<void *>(); memcpy(buf{i}, arg{i}, size{i});' for i, ty in ptr_args)}

    {' '.join(f'bo_{i}.sync(XCL_BO_SYNC_BO_TO_DEVICE);' for i, ty in ptr_args)}

    if (verbosity >= 1)
        std::cout << "Running Kernel." << std::endl;
    {'auto start = std::chrono::high_resolution_clock::now();' if autotune_time else ''}
    auto run = xrt::run(kernel);
    {set_arg_lines}
    run.start();
    run.wait2();
    {'auto stop = std::chrono::high_resolution_clock::now(); float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();' if autotune_time else ''}

    {'std::ofstream file("data.txt"); file << npu_time << std::endl; file.close();' if autotune_time else ''}

    if (verbosity >= 1)
        std::cout << "Copying results." << std::endl;
    // TODO: Assuming the last tensor is the only output tensor.
    bo_{last_ptr_idx}.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(arg{last_ptr_idx}, buf{last_ptr_idx}, size{last_ptr_idx});

    if (verbosity >= 1)
        std::cout << "Launch finished." << std::endl;

    }} catch (const std::exception& e) {{
        std::string msg = std::string("XRT runtime error: ") + e.what();
        PyErr_SetString(PyExc_RuntimeError, msg.c_str());
    }}
  }}
}}

#include "npu_dispatch_common.h"

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  {"; ".join([f"long nelem{i} = getNumElements(_arg{i}); long ebytes{i} = getElementSizeInBytes(_arg{i}); if (nelem{i} == -1 || ebytes{i} == -1) return NULL; long tensor_volume{i} = nelem{i} * ebytes{i};" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, {', '.join(f"tensor_volume{i}" for i, ty in signature.items() if i not in constants and ty[0]=="*")}, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{"set_paths", py_set_paths, METH_VARARGS, "Set path to ELF binary and kernel name"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__npu_dispatch\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___npu_dispatch(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""


def _aircc_compile(
    air_mlir_path, output_format, npu_version, air_proj_path, bf16_emulation=None
):
    """Run aircc on an AIR-dialect MLIR file to produce an NPU binary.

    Resolves the aircc binary + peano flag, builds the command for the requested
    ``output_format`` ("elf", "xclbin", or "pdi"), runs it, and returns the
    produced artifact paths. For "elf" also extracts the dispatch kernel name
    from ``full_elf_config.json``.

    Args:
        air_mlir_path: path to the AIR-dialect .mlir to compile.
        output_format: "elf", "xclbin", or "pdi".
        npu_version: target device string ("npu1"/"npu2") for ``--device``.
        air_proj_path: directory aircc writes artifacts into.

    Returns:
        elf:           {"elf_path": str, "elf_kernel_name": str}
        xclbin / pdi:  {"bin_path": str, "insts_path": str}

    Raises:
        subprocess.CalledProcessError: if aircc exits non-zero.
    """
    if bf16_emulation is None:
        bf16_emulation = npu_config.bf16_emulation
    aircc_binary_name = "aircc.exe" if IS_WINDOWS else "aircc"
    aircc_bin = _find_mlir_air_binary(aircc_binary_name)

    # Resolve the peano (llvm-aie) install root and pass it explicitly to
    # aircc. Without --peano, aiecc falls back to whichever opt/llc is on PATH
    # (e.g. a system /usr/bin/opt) which lacks the aie2p/aie2 target and fails
    # with "unrecognized architecture 'aie2p'".
    peano_flag = "--peano="
    # 1) LLVM_BINARY_DIR points to bin/, peano wants the parent. Only trust it
    #    if that parent is an AIE-capable LLVM, otherwise fall through so a
    #    misconfigured LLVM_BINARY_DIR doesn't feed aircc a bogus --peano.
    peano_dir = os.environ.get("LLVM_BINARY_DIR", "")
    if peano_dir:
        candidate = Path(peano_dir).parent
        if _is_peano_root(str(candidate)):
            peano_flag = f"--peano={candidate}"
    # 2) Auto-detect from the pip-installed llvm-aie package.
    if peano_flag == "--peano=":
        try:
            dist = importlib.metadata.distribution("llvm-aie")
            candidate = Path(dist.locate_file("")) / "llvm-aie"
            if _is_peano_root(str(candidate)):
                peano_flag = f"--peano={candidate}"
        except Exception:
            pass
    # 3) Fall back to the PEANO_INSTALL_DIR env var.
    if peano_flag == "--peano=":
        peano_env = os.environ.get("PEANO_INSTALL_DIR", "")
        if _is_peano_root(peano_env):
            peano_flag = f"--peano={peano_env}"

    # On Windows, add mlir_aie/bin to PATH so aircc can find aiecc.exe
    if IS_WINDOWS:
        # Ensure aiecc is findable
        try:
            import mlir_aie

            mlir_aie_bin = str(Path(mlir_aie.__path__[0]) / "bin")
        except ImportError:
            mlir_aie_bin = str(
                Path(aircc.__file__).resolve().parent.parent.parent / "mlir_aie" / "bin"
            )

        if os.path.isdir(mlir_aie_bin) and mlir_aie_bin not in os.environ.get(
            "PATH", ""
        ):
            os.environ["PATH"] = mlir_aie_bin + os.pathsep + os.environ.get("PATH", "")

    if output_format == "elf":
        elf_path = os.path.join(air_proj_path, "aie.elf")
        aircc_cmd = [
            aircc_bin,
            "--device",
            npu_version,
            "--no-xchesscc",
            "--no-xbridge",
            "--output-format",
            "elf",
            "--elf-name",
            elf_path,
            peano_flag,
            air_mlir_path,
        ]
    elif output_format == "pdi":
        # PDI output for alternative (non-XRT) runtimes: aircc emits a raw
        # aie.pdi (CDO->PDI via bootgen) plus the insts.bin sidecar. Uses
        # --pdi-name instead of -o. Requires mlir-air PR #1729.
        pdi_path = os.path.join(air_proj_path, "aie.pdi")
        insts_path = os.path.join(air_proj_path, "insts.bin")
        aircc_cmd = [
            aircc_bin,
            "--device",
            npu_version,
            "--no-xchesscc",
            "--no-xbridge",
            "--output-format",
            "pdi",
            "-i",
            insts_path,
            "--pdi-name",
            pdi_path,
            peano_flag,
            air_mlir_path,
        ]
    else:
        xclbin_path = os.path.join(air_proj_path, "aie.xclbin")
        insts_path = os.path.join(air_proj_path, "insts.bin")
        aircc_cmd = [
            aircc_bin,
            "--device",
            npu_version,
            "--no-xchesscc",
            "--no-xbridge",
            "--output-format",
            "xclbin",
            "-i",
            insts_path,
            "-o",
            xclbin_path,
            peano_flag,
            air_mlir_path,
        ]
    # Enable bf16 emulation: hardware truncates f32 -> bf16 before
    # multiply, with f32 accumulation.
    if bf16_emulation:
        aircc_cmd.insert(-1, "--bf16-emulation")
    # Explicitly set runtime loop tiling sizes to [4,4] (aircc
    # default changed from [4,4] to [] in mlir-air #1470).
    aircc_cmd.insert(-1, "--air-runtime-loop-tiling-sizes=4")
    aircc_cmd.insert(-1, "--air-runtime-loop-tiling-sizes=4")
    # Increase core stack size to 2048 bytes to accommodate
    # deeper call chains in register-intensive kernels.
    aircc_cmd.insert(-1, "--stack-size")
    aircc_cmd.insert(-1, "2048")
    _run_compile(aircc_cmd)

    if output_format == "elf":
        # Extract kernel name from ELF config.json. aircc writes
        # full_elf_config.json into its hardcoded default working dir
        # (<cwd>/air_project), NOT necessarily air_proj_path -- so check
        # air_proj_path first, then fall back to the cwd default.
        config_json_path = os.path.join(air_proj_path, "full_elf_config.json")
        if not os.path.isfile(config_json_path):
            fallback = os.path.join("air_project", "full_elf_config.json")
            if os.path.isfile(fallback):
                config_json_path = fallback
        elf_kernel_name = _extract_elf_kernel_name(config_json_path)
        return {"elf_path": elf_path, "elf_kernel_name": elf_kernel_name}
    bin_path = pdi_path if output_format == "pdi" else xclbin_path
    return {"bin_path": bin_path, "insts_path": insts_path}


# Global cache: maps input hash -> (loaded .pyd module, launch function)
# Persists across all NPULauncher instances for the process lifetime.
# Bypasses the expensive MLIR pipeline on repeated dispatches of the same kernel.
_global_module_cache = {}

# Last dispatched module — set after each dispatch so callers can capture it
# for direct fast-path calls (bypassing Triton JIT entirely).
_last_dispatched_module = None


def _get_cached_aircc_artifacts(cache, output_format):
    """Return cached aircc artifacts (elf/xclbin/pdi) or None if the set is incomplete.

    Keys mirror ``_put_aircc_artifacts``:
        elf:           {"elf_path", "elf_kernel_name_path"}
        xclbin / pdi:  {"bin_path", "insts_path"}
    """
    if output_format == "elf":
        elf_path = cache.get_file("aie.elf")
        kname_path = cache.get_file("elf_kernel_name.txt")
        if elf_path is None or kname_path is None:
            return None
        return {"elf_path": elf_path, "elf_kernel_name_path": kname_path}
    bin_name = "aie.pdi" if output_format == "pdi" else "aie.xclbin"
    bin_path = cache.get_file(bin_name)
    insts_path = cache.get_file("insts.bin")
    if bin_path is None or insts_path is None:
        return None
    return {"bin_path": bin_path, "insts_path": insts_path}


def _put_aircc_artifacts(cache, artifacts, output_format):
    """Persist aircc artifacts; return cached paths in the same shape as the getter.

    ``artifacts`` is the dict returned by ``_aircc_compile``.
    """
    if output_format == "elf":
        with open(artifacts["elf_path"], "rb") as f:
            elf_path = cache.put(f.read(), "aie.elf", binary=True)
        kname_path = cache.put(
            artifacts["elf_kernel_name"].encode(), "elf_kernel_name.txt"
        )
        return {"elf_path": elf_path, "elf_kernel_name_path": kname_path}
    bin_name = "aie.pdi" if output_format == "pdi" else "aie.xclbin"
    with open(artifacts["bin_path"], "rb") as f:
        bin_path = cache.put(f.read(), bin_name, binary=True)
    with open(artifacts["insts_path"], "rb") as f:
        insts_path = cache.put(f.read(), "insts.bin")
    return {"bin_path": bin_path, "insts_path": insts_path}


def compile_module(
    launcher_src,
    kernel_placeholder_name,
    output_format="xclbin",
    actual_sizes=None,
    on_cache_resolved=None,
    link_profile="xrt",
):
    """Lower a kernel and JIT-compile its host launcher shared object.

    ``link_profile`` selects the runtime the generated launcher links against:

    * ``"xrt"``  -- XRT (xclbin/ELF); links xrt_coreutil + uuid.
    * ``"hsa"``  -- HSA AIE dispatch via ROCR; links libhsa-runtime64. Requires
      ``output_format == "pdi"``.
    """
    if link_profile not in ("xrt", "hsa"):
        raise ValueError(f"link_profile must be 'xrt' or 'hsa'; got {link_profile!r}")
    if IS_WINDOWS and link_profile == "hsa":
        # Assert this up front, before resolving the ROCR SDK below, so the
        # user gets this clear message instead of a "ROCR not found" error.
        raise RuntimeError(
            "HSA (AMD_TRITON_NPU_RUNTIME=hsa) with AIE is only supported on Linux."
        )
    py_version = sys.version_info
    if platform.system() == "Windows":
        py_include_dir = os.path.join(sys.base_prefix, "include")
        py_lib_dir = os.path.join(sys.base_prefix, "libs")
        py_lib = "{name}{major}{minor}.lib".format(
            name="python", major=py_version.major, minor=py_version.minor
        )
    else:
        py_include_dir = os.path.join(
            sys.base_prefix,
            "include",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
        )
        py_lib_dir = os.path.join(sys.base_prefix, "lib")
        py_lib = "{name}{major}.{minor}".format(
            name="python", major=py_version.major, minor=py_version.minor
        )
    npu_backend_path = Path(__file__).resolve().parent
    include_dir = os.path.join(npu_backend_path, "include")
    # Runtime-specific SDK locations. Only resolve the one we actually link
    # against so an XRT-only or ROCR-only host still works.
    xrt_dir = _get_xrt_path() if link_profile == "xrt" else None
    # HSA launchers link the shared runtime library (built once against ROCR),
    # not ROCR directly, so all signatures share one process-global HsaRuntime.
    hsa_rt_dir = (
        _build_hsa_runtime_lib(include_dir, _get_rocr_install())
        if link_profile == "hsa"
        else None
    )
    aie_test_utils_dir = _get_aie_test_utils_path()

    def launch(
        gridX,
        gridY,
        gridZ,
        stream,
        cu_function,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):
        global _global_module_cache, _last_dispatched_module
        asm_src = cu_function
        kernel_name = kernel_metadata[6]  # see pack_metadata in compiler.py

        # Auto-generate a matmul tiling schedule when the IR is a plain matmul
        # and no user script is supplied. f32 matmul additionally turns on
        # bf16 emulation (no native f32 MAC on AIE). The emulation decision is
        # keyed off the detected input dtype, not off whether a schedule could
        # be derived, so an f32 matmul still enables it even when derivation
        # fails and we fall back to the default tiling.
        user_script = npu_config.transform_tiling_script
        matmul_params = None
        matmul_dtype = None
        if not user_script:
            _mm_info = _detect_matmul(asm_src.decode("utf-8", errors="ignore"))
            if _mm_info is not None:
                matmul_dtype = _mm_info["in_elem"]
            matmul_params = _matmul_transform_params(_mm_info, detect_npu_version())
        effective_bf16 = npu_config.bf16_emulation or matmul_dtype == "f32"

        # Fast path: check if we've already loaded the .pyd for this kernel.
        # The tiling script path is part of the key so the same kernel compiled
        # with different schedules (or scriptless vs a user script) does not
        # collide on the in-process module cache.
        input_key = hashlib.md5(
            asm_src + f"_{gridX}_{gridY}_{gridZ}_{kernel_name}"
            f"_{autotune_time}_{output_format}_{link_profile}"
            f"_{effective_bf16}_{user_script or ''}".encode()
        ).hexdigest()

        if input_key in _global_module_cache:
            mod = _global_module_cache[input_key]
            _last_dispatched_module = mod
            return mod.launch(
                gridX,
                gridY,
                gridZ,
                kernel_metadata,
                launch_metadata,
                launch_enter_hook,
                launch_exit_hook,
                *args,
            )

        src = launcher_src.replace(kernel_placeholder_name, kernel_name)

        air_proj_path = npu_config.air_project_path
        os.makedirs(air_proj_path, exist_ok=True)
        Path(os.path.join(air_proj_path, "asm_src.mlir")).write_bytes(asm_src)
        air_output = _ttshared_to_air(
            asm_src,
            gridX,
            gridY,
            gridZ,
            actual_sizes=actual_sizes,
            matmul_info=matmul_params,
        )
        with open(Path(os.path.join(air_proj_path, "asm_air_output.mlir")), "w") as f:
            f.write(str(air_output))

        npu_version = detect_npu_version(link_profile)
        key_data = (
            str(air_output)
            + f"_timing_{autotune_time}"
            + f"_format_{output_format}"
            + f"_link_{link_profile}"
            + f"_npu_{npu_version}"
            + f"_bf16emu_{effective_bf16}"
        )
        key = hashlib.md5(key_data.encode("utf-8")).hexdigest()

        cache = get_cache_manager(key)
        if on_cache_resolved is not None:
            on_cache_resolved(cache.cache_dir)
        name = "__npu_dispatch"
        filename = f"{name}.pyd" if IS_WINDOWS else f"{name}.so"
        cache_path = cache.get_file(filename)
        cached_artifacts = _get_cached_aircc_artifacts(cache, output_format)
        if cached_artifacts is not None:
            if output_format == "elf":
                cache_elf_path = cached_artifacts["elf_path"]
                cache_elf_kernel_path = cached_artifacts["elf_kernel_name_path"]
            else:
                cache_bin_path = cached_artifacts["bin_path"]
                cache_insts_path = cached_artifacts["insts_path"]

        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                launcher_src_path = os.path.join(tmpdir, "main.cxx")
                if IS_WINDOWS:
                    so_path = os.path.join(tmpdir, "xrt_dispatch.pyd")
                else:
                    so_path = os.path.join(tmpdir, "xrt_dispatch.exe")
                Path(launcher_src_path).write_text(src)
                # Compile the launcher shared library.
                if IS_WINDOWS:
                    cl_path = _find_msvc_cl()
                    msvc_env = _get_msvc_env(cl_path)
                    compile_flags = [
                        cl_path,
                        "/std:c++latest",
                        "/MD",
                        "/Zc:__cplusplus",
                        "/EHsc",
                        "/LD",
                        f"/Fe:{so_path}",
                        launcher_src_path,
                        f"/I{py_include_dir}",
                        f"/I{include_dir}",
                        f"/I{os.path.join(xrt_dir, 'include')}",
                        f"/link",
                        f"/LIBPATH:{py_lib_dir}",
                        f"/LIBPATH:{os.path.join(xrt_dir, 'lib')}",
                        f"{py_lib}",
                        "xrt_coreutil.lib",
                    ]
                    if output_format != "elf":
                        # xclbin mode previously needed test_utils for loading
                        # instruction binary, but that has been inlined.
                        pass
                else:
                    msvc_env = None
                    compile_flags = [
                        "g++",
                        "-std=c++23",
                        launcher_src_path,
                        f"-I{py_include_dir}",
                        f"-I{include_dir}",
                        f"-L{py_lib_dir}",
                        "-shared",
                        f"-l{py_lib}",
                        "-fPIC",
                        "-Wall",
                    ]
                    if link_profile == "hsa":
                        # Link the shared HSA runtime (libtriton_npu_hsa.so),
                        # which in turn links libhsa-runtime64. The launcher needs
                        # no ROCR headers -- only the C ABI in include/HsaRuntime
                        # (covered by -I{include_dir} above). rpath keeps both
                        # .so's loadable without LD_LIBRARY_PATH.
                        compile_flags += [
                            f"-L{hsa_rt_dir}",
                            f"-Wl,-rpath,{hsa_rt_dir}",
                            "-ltriton_npu_hsa",
                            "-lstdc++",
                        ]
                    else:
                        compile_flags += [
                            f"-I{os.path.join(xrt_dir, 'include')}",
                            f"-L{os.path.join(xrt_dir, 'lib')}",
                            "-luuid",
                            "-lxrt_coreutil",
                            "-lrt",
                            "-lstdc++",
                        ]
                    if output_format != "elf":
                        # xclbin mode previously needed test_utils for loading
                        # instruction binary, but that has been inlined.
                        pass
                    compile_flags += ["-o", so_path]
                _run_compile(compile_flags, env=msvc_env)

                ###### Compile to binary (ELF or xclbin + insts)
                air_mlir_path = os.path.join(air_proj_path, "asm_air_output.mlir")
                artifacts = _aircc_compile(
                    air_mlir_path,
                    output_format,
                    npu_version,
                    air_proj_path,
                    bf16_emulation=effective_bf16,
                )

                # Cache format-specific artifacts first, then the .so last.
                # This avoids partial cache entries if aircc or kernel name
                # extraction fails -- the .so is the gate for cache hits.
                cached_artifacts = _put_aircc_artifacts(cache, artifacts, output_format)
                if output_format == "elf":
                    cache_elf_path = cached_artifacts["elf_path"]
                    cache_elf_kernel_path = cached_artifacts["elf_kernel_name_path"]
                else:
                    cache_bin_path = cached_artifacts["bin_path"]
                    cache_insts_path = cached_artifacts["insts_path"]
                with open(so_path, "rb") as f:
                    cache_path = cache.put(f.read(), filename, binary=True)

                # Check for compile-only mode
                if npu_config.compile_only:
                    logger.debug("Compile-only mode: binaries cached at %s", cache_path)
                    if output_format == "elf":
                        logger.debug("  elf: %s", cache_elf_path)
                    else:
                        logger.debug("  %s: %s", output_format, cache_bin_path)
                        logger.debug("  insts: %s", cache_insts_path)
                    return None
        else:
            logger.debug(
                "got cache path: %s compilation is therefore skipped "
                "(delete cache path to force recompile).",
                cache_path,
            )

            # Check for compile-only mode (cache hit)
            if npu_config.compile_only:
                logger.debug(
                    "Compile-only mode (cache hit): binaries at %s", cache_path
                )
                if output_format == "elf":
                    logger.debug("  elf: %s", cache_elf_path)
                else:
                    logger.debug("  %s: %s", output_format, cache_bin_path)
                    logger.debug("  insts: %s", cache_insts_path)
                return None

        # Load and launch the compiled kernel.
        spec = importlib.util.spec_from_file_location(name, cache_path)
        if spec is None:
            raise RuntimeError(f"Cannot find {name} module in {cache_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if output_format == "elf":
            # Read the cached kernel name
            with open(cache_elf_kernel_path) as f:
                elf_kernel_name = f.read().strip()
            # Strip Windows extended-length path prefix (\\?\) which
            # confuses XRT's internal path parsing (stoul error).
            elf_path_str = cache_elf_path
            if IS_WINDOWS and elf_path_str.startswith("\\\\?\\"):
                elf_path_str = elf_path_str[4:]
            mod.set_paths(elf_path_str, elf_kernel_name)
        else:
            bin_path_str = cache_bin_path
            insts_path_str = cache_insts_path
            if IS_WINDOWS:
                if bin_path_str.startswith("\\\\?\\"):
                    bin_path_str = bin_path_str[4:]
                if insts_path_str.startswith("\\\\?\\"):
                    insts_path_str = insts_path_str[4:]
            mod.set_paths(bin_path_str, insts_path_str)

        # Cache the loaded module for fast subsequent dispatches
        _global_module_cache[input_key] = mod
        _last_dispatched_module = mod

        return mod.launch(
            gridX,
            gridY,
            gridZ,
            kernel_metadata,
            launch_metadata,
            launch_enter_hook,
            launch_exit_hook,
            *args,
        )

    return launch


class NPULauncher(object):
    # Placeholder replaced with the real kernel name at compile time.
    kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"

    # Instance attributes, declared here so every instance has a well-defined
    # shape even if compilation raises before they are assigned in _finalize.
    output_format: "str | None" = None
    npu_cache_dir: "str | None" = None
    launch = None  # callable(gridX, gridY, gridZ, stream, function, *args)

    def __init__(self, src, metadata, runtime="xrt"):
        """Build the host launcher for ``src`` targeting ``runtime``.

        ``runtime`` is "xrt" (xclbin/ELF via XRT) or "hsa" (PDI + insts via
        HSA/ROCR); it is bound by ``NPUDriver`` when it constructs the launcher.
        """
        constants, signature = extract_signature_and_constants(src)
        if runtime == "hsa":
            # HSA consumes PDI + insts and links ROCR. Import lazily so the
            # (large) HSA codegen module is only loaded when the HSA path is used.
            from .hsa_launcher import _generate_hsa_launcher

            self.output_format = "pdi"
            launcher_src = _generate_hsa_launcher(
                constants, signature, self.kernel_placeholder_name
            )
            link_profile = "hsa"
        else:
            # Detect output format: ELF for npu2, xclbin for npu1. Pass the
            # bound runtime so the format decision matches this launcher's
            # runtime rather than the (possibly different) global config.
            self.output_format = _get_output_format(runtime=runtime)
            if self.output_format == "elf":
                launcher_src = _generate_elf_launcher(
                    constants, signature, self.kernel_placeholder_name
                )
            else:
                launcher_src = _generate_launcher(
                    constants, signature, self.kernel_placeholder_name
                )
            link_profile = "xrt"

        self._finalize(src, launcher_src, link_profile=link_profile)

    def _finalize(self, src, launcher_src: str, link_profile: str = "xrt") -> None:
        """Extract padding sizes, wire the cache-dir callback, and JIT-compile.

        Uses ``self.output_format`` (set by ``__init__``). Shared by the XRT and
        HSA paths so the compile/caching tail is defined once; ``link_profile``
        selects which runtime the launcher links against.
        """
        actual_sizes = extract_actual_sizes(src)

        self.npu_cache_dir = None

        def _on_cache_resolved(cache_dir):
            self.npu_cache_dir = cache_dir

        self.launch = compile_module(
            launcher_src,
            self.kernel_placeholder_name,
            self.output_format,
            actual_sizes=actual_sizes,
            on_cache_resolved=_on_cache_resolved,
            link_profile=link_profile,
        )

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        self.launch(gridX, gridY, gridZ, stream, function, *args)


def get_npu_cache_dir(compiled_kernel):
    """Return the NPU binary cache directory for a compiled kernel.

    The NPU backend stores hardware-specific artifacts in a separate cache
    directory from Triton's main compiler cache. Depending on the selected
    output format, the directory contains either:

    * xclbin output: ``aie.xclbin``, ``insts.bin``, and
      ``__npu_dispatch.so``
    * elf output: ``aie.elf``, ``elf_kernel_name.txt``, and
      ``__npu_dispatch.so``

    This function returns the path to that directory.

    The directory is only populated after the first kernel invocation,
    since NPU binary compilation is deferred to launch time.

    Args:
        compiled_kernel: A triton.compiler.compiler.CompiledKernel instance
            compiled for the NPU backend.

    Returns:
        str | None: Absolute path to the NPU binary cache directory, or
            None if the kernel has not been launched yet or does not expose
            an NPU launcher via ``_run``.

    Raises:
        TypeError: If ``compiled_kernel._run`` exists but is not an
            ``NPULauncher`` instance.

    Example::

        compiled_kernel = my_kernel[grid](a, b, c, N, BLOCK_SIZE_N=1024)
        npu_cache = get_npu_cache_dir(compiled_kernel)
        print(f"NPU artifacts at: {npu_cache}")
    """
    launcher = getattr(compiled_kernel, "_run", None)
    if launcher is None:
        return None
    if not isinstance(launcher, NPULauncher):
        raise TypeError(
            f"Expected an NPULauncher but got {type(launcher).__name__}. "
            "Is the NPU backend active?"
        )
    return launcher.npu_cache_dir


class NPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    # Note:
    # nvidia and amd backends have their corresponding driver.c file that exposes
    # get_device_properties and load_binary using python bindings.
    # (see third_party/nvidia/backend/driver.c)
    # These methods are then used in compiler.py to initialize handles before running
    # the triton kernels.
    # Since we recompile the kernel every time (see compile_module above),
    # and the metadata generated by these functions aren't applicable to the npu
    # backend, just define the same functions with dummy implementation.
    @staticmethod
    def get_device_properties(device):
        return {
            "max_shared_mem": 2**20,
            "multiprocessor_count": None,
            "sm_clock_rate": None,
            "mem_clock_rate": None,
            "mem_bus_width": None,
        }

    # Important note:
    # Since we cannot easy pass function pointers around, we pass along the
    # assembly source code so that compile_module above can recompile the
    # module every time.
    @staticmethod
    def load_binary(name, kernel_asm, shared, device):
        return (
            None,  # module
            kernel_asm,  # function
            None,  # n_regs
            None,  # n_spills
            sys.maxsize,  # n_max_threads
        )


class NPUDriver(DriverBase):

    def __init__(self, runtime=None):
        """Create the NPU driver for a launch runtime.

        ``runtime`` selects how kernels are dispatched:

        * ``"xrt"`` -- XRT (xclbin on npu1, ELF on npu2).
        * ``"hsa"`` -- HSA/ROCR AIE dispatch (PDI + insts).

        Defaults to ``npu_config.runtime`` (the ``AMD_TRITON_NPU_RUNTIME`` env
        var, itself defaulting to ``"xrt"``), so ``NPUDriver()`` honors the
        environment while ``NPUDriver("hsa")`` / ``NPUDriver("xrt")`` force it.
        """
        super().__init__()
        if runtime is None:
            # Already validated + normalized by the config property.
            runtime = npu_config.runtime
        elif isinstance(runtime, str):
            runtime = runtime.lower()  # match npu_config.runtime normalization
        if runtime not in _VALID_RUNTIMES:
            raise ValueError(
                f"runtime must be one of {sorted(_VALID_RUNTIMES)}; got {runtime!r}"
            )
        # Exposed for introspection (e.g. triton.runtime.driver.active.runtime).
        self.runtime = runtime
        self.utils = NPUUtils()
        # Triton instantiates the launcher as ``launcher_cls(src, metadata)`` and
        # never passes the driver, so bind the chosen runtime here.
        self.launcher_cls = functools.partial(NPULauncher, runtime=runtime)
        self.binary_ext = "ttsharedir"

    # NPU driver won't be automatically chosen unless explicitly set through
    # triton.runtime.driver.set_active(NPUDriver()) / NPUDriver("hsa").
    @staticmethod
    def is_active():
        return False

    def do_bench(
        self,
        fn,
        warmup=25,
        rep=100,
        grad_to_none=None,
        quantiles=None,
        return_mode="mean",
    ):
        assert return_mode in ["min", "max", "mean", "median", "all"]

        global autotune_time
        autotune_time = True

        fn()

        # Estimate the runtime of the function
        estimate_us = 0.0
        for _ in range(5):
            fn()
            with open("data.txt", "r") as f:
                value_str = f.read().strip()
            value = float(value_str)
            estimate_us += value

        estimate_ms = estimate_us / (5 * 1000)

        from triton import knobs

        verbose = knobs.autotuning.print
        if verbose:
            print("NPU estimate ms: ", estimate_ms)
        # compute number of warmup and repeat
        n_warmup = max(1, int(25 / estimate_ms))
        n_repeat = max(5, int(100 / estimate_ms))

        # Warm-up
        for _ in range(n_warmup):
            fn()

        times = [0.0 for i in range(n_repeat)]
        # Benchmark
        for i in range(n_repeat):
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            fn()
            with open("data.txt", "r") as f:
                value_str = f.read().strip()
            times[i] = float(value_str) / 1000

        if verbose:
            print("NPU KERNEL TIME (ms): ", ", ".join(str(t) for t in times))
        autotune_time = False

        from triton.testing import _summarize_statistics

        return _summarize_statistics(times, quantiles, return_mode)

    def get_benchmarker(self):
        return self.do_bench

    def get_device_capability(self):
        return ("npu", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        # NPU doesn't have a device to return. Return something.
        return "npu"

    def set_current_device(self, device):
        # NPU doesn't have a device to set
        assert device == "npu"
        return

    def get_current_target(self):
        return GPUTarget("npu", 0, 0)

    def get_active_torch_device(self):
        import torch

        return torch.device("npu")

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)
