# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Triton XDNA Wheel Build Setup

This setup.py builds a triton wheel with the amd_triton_npu and triton_shared backends included.
It uses triton's native build system via pip wheel, then customizes the output.

Usage:
    # Build wheel using cibuildwheel
    cibuildwheel --platform linux

    # Build wheel directly (development)
    pip wheel . --no-build-isolation -w wheelhouse

    # Install in development mode (uses the old approach)
    pip install -e . --no-build-isolation
"""

import os
import glob
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from email.parser import Parser
import tarfile
import urllib.request
import shlex

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install


try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    from setuptools.command.bdist_wheel import bdist_wheel


# =============================================================================
# Platform Check
# =============================================================================

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform == "linux"

if not (IS_LINUX or IS_WINDOWS):
    sys.exit("ERROR: triton-xdna currently only supports Linux and Windows.")


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent.resolve()
THIRD_PARTY_DIR = BASE_DIR / "third_party"
TRITON_SOURCE_DIR = THIRD_PARTY_DIR / ("triton-windows" if IS_WINDOWS else "triton")
TRITON_SHARED_DIR = THIRD_PARTY_DIR / "triton_shared"
AMD_TRITON_NPU_DIR = BASE_DIR / "amd_triton_npu"

# same strategy as used in TheRock
# https://github.com/ROCm/TheRock/pull/4006/changes#diff-6812ec4cd824b4a56416cd4ca74afdece86fe8d39c813300eddc0d17194b9e80R153-R155
LLVM_BASE_URL = "https://oaitriton.blob.core.windows.net/public/llvm-builds"

# Patch configuration: (submodule_name, patch_file)
PATCHES = [
    ("triton-windows", "triton-windows.patch") if IS_WINDOWS else ("triton", "triton.patch"),
    ("triton_shared", "triton_shared.patch"),
]

# Marker file name to track if patches have been applied
PATCH_MARKER_FILE = ".patches_applied"


def find_triton_shared_opt_binary(triton_source_dir: Path = None) -> Path:
    """
    Find the triton-shared-opt binary in the build directory.

    The binary is located at:
    triton/build/cmake.*/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt

    triton\\build\\lib.win-amd64-cpython-312\\triton\\_C\\triton-shared-opt.exe

    Returns:
        Path to the binary if found, None otherwise.
    """
    if triton_source_dir is None:
        triton_source_dir = TRITON_SOURCE_DIR

    build_dir = triton_source_dir / "build"

    if not build_dir.exists():
        return None
    
    if IS_WINDOWS:
        for lib_dir in build_dir.glob("lib.*"):
            binary_path = (
                lib_dir
                / "triton"
                / "_C"
                / "triton-shared-opt.exe"
            )
        if binary_path.exists() and binary_path.is_file():
            return binary_path
    else:
        for cmake_dir in build_dir.glob("cmake.*"):
            binary_path = (
                cmake_dir
                / "third_party"
                / "triton_shared"
                / "tools"
                / "triton-shared-opt"
                / "triton-shared-opt"
            )
            if binary_path.exists() and binary_path.is_file():
                return binary_path

    return None


def check_env_flag(name: str, default: str = "") -> bool:
    """Check if an environment variable is set to a truthy value."""
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


# =============================================================================
# Patch Management
# =============================================================================


def apply_submodule_patches():
    """
    Apply local patches to third-party submodules before building.

    This function applies patch files from third_party/ to their respective
    submodules. It uses marker files to track whether patches have been applied
    to avoid re-applying them on subsequent builds.
    """
    print("=" * 60, file=sys.stderr)
    print("Checking/applying patches to submodules", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    for submodule_name, patch_name in PATCHES:
        submodule_dir = THIRD_PARTY_DIR / submodule_name
        patch_file = THIRD_PARTY_DIR / patch_name
        marker_file = submodule_dir / PATCH_MARKER_FILE

        print(f"\n[{submodule_name}]", file=sys.stderr)

        # Check if submodule directory exists
        if not submodule_dir.exists():
            print(
                f"  ⚠ Submodule directory not found: {submodule_dir}", file=sys.stderr
            )
            continue

        # Check if patch file exists
        if not patch_file.exists():
            print(f"  ⚠ Patch file not found: {patch_file}", file=sys.stderr)
            continue

        # Check if already applied (marker exists)
        if marker_file.exists():
            print(f"  ✓ Patches already applied (marker exists)", file=sys.stderr)
            continue

        # Check if patch can be applied (dry run)
        check_result = subprocess.run(
            ["git", "apply", "--check", str(patch_file)],
            cwd=submodule_dir,
            capture_output=True,
            text=True,
        )

        if check_result.returncode != 0:
            # Check if patch is already applied by trying reverse
            reverse_result = subprocess.run(
                ["git", "apply", "--check", "--reverse", str(patch_file)],
                cwd=submodule_dir,
                capture_output=True,
                text=True,
            )

            if reverse_result.returncode == 0:
                print(f"  ✓ Patch already applied", file=sys.stderr)
                marker_file.touch()
            else:
                print(
                    f"  ✗ Patch conflict: {check_result.stderr.strip()}",
                    file=sys.stderr,
                )
            continue

        # Apply the patch
        print(f"  Applying {patch_name}...", file=sys.stderr)
        apply_result = subprocess.run(
            ["git", "apply", str(patch_file)],
            cwd=submodule_dir,
            capture_output=True,
            text=True,
        )

        if apply_result.returncode == 0:
            print(f"  ✓ Patch applied successfully", file=sys.stderr)
            marker_file.touch()
        else:
            print(f"  ✗ Failed to apply patch: {apply_result.stderr}", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)


# =============================================================================
# Version Management
# =============================================================================


def parse_hash_file(filepath, key):
    """Parse a hash file and extract the value for a given key."""
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip()
    raise ValueError(f"Key '{key}' not found in {filepath}")


def get_version():
    """Generate version string from environment or git commit."""
    if "TRITON_XDNA_WHEEL_VERSION" in os.environ:
        version = os.environ["TRITON_XDNA_WHEEL_VERSION"].lstrip("v")
        if version:
            return version

    # Base version - use triton's version as base
    release_version = "3.6.0"

    # Get commit hash
    commit_hash = os.environ.get("TRITON_XDNA_PROJECT_COMMIT", "")
    if not commit_hash:
        try:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short=7", "HEAD"],
                    cwd=BASE_DIR,
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            commit_hash = "unknown"

    # Get timestamp
    now = datetime.now()
    timestamp = os.environ.get(
        "DATETIME", f"{now.year}{now.month:02}{now.day:02}{now.hour:02}"
    )

    return f"{release_version}.{timestamp}+{commit_hash}"


def _get_installed_version(pkg_name):
    """Return the installed version of a package, or None if not installed."""
    try:
        from importlib.metadata import version as _ver
        return _ver(pkg_name)
    except Exception:
        return None


def _is_locally_available(pkg_name):
    """Check if a package is locally available via .pth file (built from source)."""
    import site
    pth_name = f"{pkg_name}.pth"
    for sp in site.getsitepackages():
        if os.path.exists(os.path.join(sp, pth_name)):
            return True
    return False


def _make_version_spec(pkg_name, version, timestamp, short_commit, suffix=""):
    """Build a pip version specifier for a package.

    On Windows the CI publishes wheels with timestamps that may differ by ±1
    from the value recorded in the hash file.  If the package is already
    installed (via pip) we pin to its exact version.  If it is locally
    available via a .pth file (built from source) we skip it entirely.
    Otherwise we list the hash-file version so Linux CI keeps working.

    Returns the version spec string, or None if the package should be
    omitted from install_requires.
    """
    installed = _get_installed_version(pkg_name)
    if installed is not None:
        return f"{pkg_name}=={installed}"
    if _is_locally_available(pkg_name):
        return None  # available via .pth, no pip requirement needed
    full = f"{version}.{timestamp}+{short_commit}{suffix}"
    return f"{pkg_name}=={full}"


def get_install_requires():
    """Build install_requires list from hash files."""
    mlir_aie_hash_file = BASE_DIR / "utils" / "mlir-aie-hash.txt"
    mlir_air_hash_file = BASE_DIR / "utils" / "mlir-air-hash.txt"

    # Parse mlir-aie version
    mlir_aie_version = parse_hash_file(mlir_aie_hash_file, "Version")
    mlir_aie_timestamp = parse_hash_file(mlir_aie_hash_file, "Timestamp")
    mlir_aie_commit = parse_hash_file(mlir_aie_hash_file, "Commit")
    mlir_aie_short_commit = mlir_aie_commit[:7]
    mlir_aie_full_version = (
        f"{mlir_aie_version}.{mlir_aie_timestamp}+{mlir_aie_short_commit}.no.rtti"
    )

    # Parse mlir-air version
    mlir_air_version = parse_hash_file(mlir_air_hash_file, "Version")
    mlir_air_timestamp = parse_hash_file(mlir_air_hash_file, "Timestamp")
    mlir_air_commit = parse_hash_file(mlir_air_hash_file, "Commit")
    mlir_air_short_commit = mlir_air_commit[:7]
    mlir_air_full_version = (
        f"{mlir_air_version}.{mlir_air_timestamp}+{mlir_air_short_commit}.no.rtti"
    )

    return [
        f"mlir-aie=={mlir_aie_full_version}",
        "llvm-aie",
        f"mlir-air=={mlir_air_full_version}",
    ]


def get_triton_windows_llvm_hash(triton_dir: Path) -> str:
    """Read the LLVM hash from triton-windows cmake/llvm-hash.txt."""
    hash_file = triton_dir / "cmake" / "llvm-hash.txt"
    if not hash_file.exists():
        raise RuntimeError(f"LLVM hash file not found: {hash_file}")
    return hash_file.read_text().strip()


def download_llvm_for_triton_windows(triton_dir: Path) -> Path:
    """Download and extract pre-built LLVM binaries for triton-windows.

    triton-windows requires a specific LLVM version that matches the hash
    in cmake/llvm-hash.txt. Pre-built binaries are hosted at oaitriton.blob.core.windows.net.
    """
    full_hash = get_triton_windows_llvm_hash(triton_dir)
    short_hash = full_hash[:8]

    llvm_dir = triton_dir.parent / f"llvm-{short_hash}-windows-x64"
    llvm_hash_marker = llvm_dir / ".llvm-hash"

    if llvm_hash_marker.exists():
        installed_hash = llvm_hash_marker.read_text().strip()
        if installed_hash == full_hash:
            print(f"LLVM already downloaded: {llvm_dir}")
            return llvm_dir

    if llvm_dir.exists():
        shutil.rmtree(llvm_dir)

    filename = f"llvm-{short_hash}-windows-x64.tar.gz"
    download_url = f"{LLVM_BASE_URL}/{filename}"

    print(f"Downloading LLVM for triton-windows...")
    print(f"  Hash: {short_hash}")
    print(f"  URL: {download_url}")

    with tempfile.TemporaryDirectory() as temp_dir:
        download_path = Path(temp_dir) / filename

        print("  Downloading (this may take a few minutes, ~500MB)...")
        try:
            urllib.request.urlretrieve(download_url, download_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download LLVM from {download_url}: {e}\n"
                "You may need to download manually and extract to "
                f"{llvm_dir}"
            )

        print("  Extracting...")
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(triton_dir.parent, filter="data")

        if not llvm_dir.exists():
            raise RuntimeError(f"Extracted LLVM directory not found: {llvm_dir}")

        llvm_hash_marker.write_text(full_hash)

    print(f"  LLVM downloaded to: {llvm_dir}")
    return llvm_dir


def run_command(args: list[str | Path], cwd: Path, env: dict[str, str] | None = None):
    args = [str(arg) for arg in args]
    full_env = dict(os.environ)
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    if env:
        print(f":: Env:")
        for k, v in env.items():
            print(f"  {k}={v}")
        full_env.update(env)
    subprocess.check_call(args, cwd=str(cwd), env=full_env)


# =============================================================================
# Wheel Building
# =============================================================================


class TritonXdnaBdistWheel(bdist_wheel):
    """
    Custom bdist_wheel that:
    1. Builds triton wheel with external plugins via TRITON_PLUGIN_DIRS
    2. Unpacks the wheel and modifies it
    3. Adds triton-shared-opt binary
    4. Updates metadata with our dependencies
    5. Repacks as final wheel
    """

    def run(self):
        # Apply patches before building
        apply_submodule_patches()

        print("=" * 60, file=sys.stderr)
        print("Building Triton XDNA wheel", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        # Create temp directory for wheel building
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Step 1: Build triton wheel with plugins
            if IS_WINDOWS:
                triton_wheel = self._build_triton_windows(tmpdir)
            else:
                triton_wheel = self._build_triton_wheel(tmpdir)

            # Step 2: Unpack the wheel
            unpack_dir = tmpdir / "unpacked"
            self._unpack_wheel(triton_wheel, unpack_dir)

            # Step 3: Add triton-shared-opt binary
            self._add_triton_shared_opt(unpack_dir)

            # Step 4: Rename package from triton to triton-xdna
            self._rename_package(unpack_dir)

            # Step 5: Update metadata with dependencies
            self._update_metadata(unpack_dir)

            # Step 6: Repack as final wheel with new name
            final_wheel = self._repack_wheel(unpack_dir, tmpdir, triton_wheel.name)

            # Step 7: Copy to dist directory
            self._copy_to_dist(final_wheel)

        print("=" * 60, file=sys.stderr)
        print("Triton XDNA wheel build complete!", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    def _build_triton_windows(self, tmpdir: Path) -> Path:
        """Build triton wheel with external plugins."""
        print("\n[1/5] Building triton wheel with plugins...", file=sys.stderr)

        wheel_dir = tmpdir / "triton_wheel"
        wheel_dir.mkdir(parents=True, exist_ok=True)

        # Get the project root (works both on host and in container)
        # Use the current working directory if it looks like a container mount
        cwd = Path.cwd()
        if (cwd / "third_party" / "triton-windows").exists():
            project_root = cwd
        else:
            project_root = BASE_DIR

        # Set plugin directories relative to project root
        triton_shared_dir = project_root / "third_party" / "triton_shared"
        triton_dir = project_root / "third_party" / "triton-windows"
        amd_npu_dir = project_root / "amd_triton_npu"

        plugin_dirs = f"{triton_shared_dir};{amd_npu_dir}"
        print(f"  TRITON_PLUGIN_DIRS={plugin_dirs}", file=sys.stderr)
        print(f"  Project root: {project_root}", file=sys.stderr)

        llvm_build_dir = download_llvm_for_triton_windows(triton_dir)

        # Prepare environment for triton-windows build.
        # Note: MSVC environment (vcvars64.bat) must already be set up.
        windows_env = dict(os.environ)
        windows_env.update(
            {
                "PYTHONUTF8": "1",
                "LLVM_BUILD_DIR": str(llvm_build_dir),
                "LLVM_INCLUDE_DIRS": str(llvm_build_dir / "include"),
                "LLVM_LIBRARY_DIR": str(llvm_build_dir / "lib"),
                "LLVM_SYSPATH": str(llvm_build_dir),
                "TRITON_BUILD_PROTON": "OFF",
                "TRITON_APPEND_CMAKE_ARGS": "-DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=FALSE",
                "TRITON_PLUGIN_DIRS": plugin_dirs,
                # Override package name to "triton" for consistency with Linux
                "TRITON_WHEEL_NAME": "triton",
                # cl MUST be used or it will fail with a -fPIC error
                "CC": "cl.exe",
                "CXX": "cl.exe"
            }
        )

        print("+++ Installing build dependencies:")
        run_command(
            [sys.executable, "-m", "pip", "install", "build", "wheel"],
            cwd=triton_dir,
        )

        print("+++ Building triton:")
        run_command(
            [sys.executable, "-m", "build", "--wheel", "-v"],
            cwd=triton_dir,
            env=windows_env,
        )

        # Find the built wheel
        wheel_dir = triton_dir / "dist"
        wheels = list(wheel_dir.glob("triton-*.whl"))
        if not wheels:
            raise RuntimeError(f"No triton wheel found in {wheel_dir}")

        triton_wheel = wheels[0]
        print(f"  Built: {triton_wheel.name}", file=sys.stderr)
        return triton_wheel

    def _build_triton_wheel(self, tmpdir: Path) -> Path:
        """Build triton wheel with external plugins."""
        print("\n[1/5] Building triton wheel with plugins...", file=sys.stderr)

        wheel_dir = tmpdir / "triton_wheel"
        wheel_dir.mkdir(parents=True, exist_ok=True)

        # Set up environment - use paths relative to triton source for container compatibility
        env = os.environ.copy()

        # Get the project root (works both on host and in container)
        # Use the current working directory if it looks like a container mount
        cwd = Path.cwd()
        if (cwd / "third_party" / "triton").exists():
            project_root = cwd
        else:
            project_root = BASE_DIR

        # Set plugin directories relative to project root
        triton_shared_dir = project_root / "third_party" / "triton_shared"
        amd_npu_dir = project_root / "amd_triton_npu"

        plugin_dirs = f"{triton_shared_dir};{amd_npu_dir}"
        env["TRITON_PLUGIN_DIRS"] = plugin_dirs
        print(f"  TRITON_PLUGIN_DIRS={plugin_dirs}", file=sys.stderr)
        print(f"  Project root: {project_root}", file=sys.stderr)

        # Set build flags (clang+lld not used on Windows; MSVC is used instead)
        if not IS_WINDOWS and check_env_flag("TRITON_BUILD_WITH_CLANG_LLD", "true"):
            env["TRITON_BUILD_WITH_CLANG_LLD"] = "true"

        triton_source = project_root / "third_party" / "triton"

        # Use subprocess with pip wheel - this properly inherits environment
        # Find a working Python with pip
        if IS_WINDOWS:
            python_candidates = [
                sys.executable,
                "python",
            ]
        else:
            python_candidates = [
                sys.executable,
                "/opt/python/cp313-cp313/bin/python",
                "/opt/python/cp312-cp312/bin/python",
                "/opt/python/cp311-cp311/bin/python",
                "python3",
                "python",
            ]

        python_exe = None
        for candidate in python_candidates:
            try:
                result = subprocess.run(
                    [candidate, "-c", "import pip; print('ok')"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                if result.returncode == 0:
                    python_exe = candidate
                    break
            except Exception:
                continue

        if python_exe is None:
            raise RuntimeError("Could not find Python interpreter with pip")

        print(f"  Using Python: {python_exe}", file=sys.stderr)

        cmd = [
            python_exe,
            "-m",
            "pip",
            "wheel",
            str(triton_source),
            "--no-build-isolation",
            "-w",
            str(wheel_dir),
            "-v",
        ]

        print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
        subprocess.check_call(cmd, env=env)

        # Find the built wheel
        wheels = list(wheel_dir.glob("triton-*.whl"))
        if not wheels:
            raise RuntimeError(f"No triton wheel found in {wheel_dir}")

        triton_wheel = wheels[0]
        print(f"  Built: {triton_wheel.name}", file=sys.stderr)
        return triton_wheel

    def _unpack_wheel(self, wheel_path: Path, unpack_dir: Path):
        """Unpack the wheel."""
        print("\n[2/5] Unpacking wheel...", file=sys.stderr)

        unpack_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(wheel_path, "r") as zf:
            zf.extractall(unpack_dir)

        print(f"  Unpacked to: {unpack_dir}", file=sys.stderr)

    def _add_triton_shared_opt(self, unpack_dir: Path):
        """Add triton-shared-opt binary to the unpacked wheel."""
        print("\n[3/7] Adding triton-shared-opt binary...", file=sys.stderr)

        triton_shared_opt_binary = find_triton_shared_opt_binary(TRITON_SOURCE_DIR)

        if triton_shared_opt_binary is None:
            # Warning: triton-shared-opt binary not found in D:\winxdna\third_party\triton/build/
            print(
                f"  Warning: triton-shared-opt binary not found in {TRITON_SOURCE_DIR}/build/",
                file=sys.stderr,
            )
            return

        print(f"  Found binary: {triton_shared_opt_binary}", file=sys.stderr)

        # Copy binary to triton package
        triton_shared_dst = unpack_dir / "triton" / "triton_shared"
        triton_shared_dst.mkdir(parents=True, exist_ok=True)

        # Use a platform-appropriate binary name (Windows requires .exe)
        binary_name = "triton-shared-opt.exe" if IS_WINDOWS else "triton-shared-opt"
        dst_binary = triton_shared_dst / binary_name
        shutil.copy2(triton_shared_opt_binary, dst_binary)

        # Ensure the binary has execute permissions (not needed on Windows)
        if not IS_WINDOWS:
            os.chmod(dst_binary, 0o755)

        print(f"  Copied triton-shared-opt to {dst_binary}", file=sys.stderr)

    def _rename_package(self, unpack_dir: Path):
        """Rename package from triton to triton-xdna and update version."""
        print("\n[4/7] Renaming package from triton to triton-xdna...", file=sys.stderr)

        # Find the dist-info directory
        dist_info_dirs = list(unpack_dir.glob("triton-*.dist-info"))
        if not dist_info_dirs:
            raise RuntimeError("Could not find triton dist-info directory")

        old_dist_info = dist_info_dirs[0]
        old_name = old_dist_info.name

        # Get our custom version with timestamp and commit hash
        our_version = get_version()
        print(f"  Using version: {our_version}", file=sys.stderr)

        # Create new dist-info directory name with our version
        new_dist_info_name = f"triton_xdna-{our_version}.dist-info"
        new_dist_info = unpack_dir / new_dist_info_name

        # Rename the directory
        old_dist_info.rename(new_dist_info)
        print(f"  Renamed {old_name} -> {new_dist_info_name}", file=sys.stderr)

        # Update METADATA file - change Name and Version fields
        metadata_path = new_dist_info / "METADATA"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.startswith("Name: "):
                    new_lines.append("Name: triton-xdna\n")
                elif line.startswith("Version: "):
                    new_lines.append(f"Version: {our_version}\n")
                else:
                    new_lines.append(line)

            with open(metadata_path, "w") as f:
                f.writelines(new_lines)
            print("  Updated Name and Version in METADATA", file=sys.stderr)

        # Update WHEEL file if needed (usually doesn't need changes)
        wheel_path = new_dist_info / "WHEEL"
        if wheel_path.exists():
            with open(wheel_path, "r") as f:
                content = f.read()
            # WHEEL file typically doesn't contain the package name, but check anyway
            if "triton" in content.lower() and "triton-xdna" not in content.lower():
                content = content.replace("triton", "triton-xdna")
                with open(wheel_path, "w") as f:
                    f.write(content)
                print("  Updated WHEEL file", file=sys.stderr)

    def _update_metadata(self, unpack_dir: Path):
        """Update wheel metadata with our dependencies and version."""
        print("\n[5/7] Updating wheel metadata...", file=sys.stderr)

        # Find the dist-info directory (now named triton_xdna-*)
        dist_info_dirs = list(unpack_dir.glob("triton_xdna-*.dist-info"))
        if not dist_info_dirs:
            raise RuntimeError("Could not find triton_xdna dist-info directory")

        dist_info = dist_info_dirs[0]

        # Update METADATA
        metadata_path = dist_info / "METADATA"
        with open(metadata_path, "r") as f:
            content = f.read()

        # Get our dependencies
        our_deps = get_install_requires()

        # METADATA format: headers, then blank line, then body
        # We need to insert Requires-Dist lines in the header section
        # (before the first blank line)
        lines = content.split("\n")
        new_lines = []
        header_ended = False
        deps_inserted = False

        for i, line in enumerate(lines):
            # Check if we've reached the end of headers (blank line)
            if not header_ended and line.strip() == "":
                # Insert our dependencies before the blank line
                if not deps_inserted:
                    for dep in our_deps:
                        dep_line = f"Requires-Dist: {dep}"
                        # Only add if not already present
                        if dep_line not in content:
                            new_lines.append(dep_line)
                    deps_inserted = True
                header_ended = True

            new_lines.append(line)

        # If no blank line was found (unusual), append at end of headers
        if not deps_inserted:
            # Find the last header line and insert after it
            insert_pos = len(new_lines)
            for dep in our_deps:
                dep_line = f"Requires-Dist: {dep}"
                if dep_line not in content:
                    new_lines.insert(insert_pos, dep_line)
                    insert_pos += 1

        new_content = "\n".join(new_lines)

        with open(metadata_path, "w") as f:
            f.write(new_content)

        print(f"  Updated METADATA in {dist_info.name}", file=sys.stderr)
        print(f"  Added dependencies: {our_deps}", file=sys.stderr)

    def _repack_wheel(
        self, unpack_dir: Path, tmpdir: Path, original_wheel_name: str
    ) -> Path:
        """Repack the modified wheel with triton-xdna name and our version."""
        print("\n[6/7] Repacking wheel...", file=sys.stderr)

        # Find dist-info to get RECORD file (now named triton_xdna-*)
        dist_info_dirs = list(unpack_dir.glob("triton_xdna-*.dist-info"))
        if not dist_info_dirs:
            raise RuntimeError("Could not find triton_xdna dist-info directory")

        dist_info = dist_info_dirs[0]
        record_file = dist_info / "RECORD"

        # Update RECORD with new files
        self._update_record(unpack_dir, record_file)

        # Parse the original wheel name to extract tags
        # Format: triton-{version}-{python_tag}-{abi_tag}-{platform_tag}.whl
        # e.g., triton-3.6.0-cp313-cp313-linux_x86_64.whl
        import re

        wheel_pattern = re.compile(r"triton-[^-]+-([^-]+)-([^-]+)-(.+)\.whl")
        match = wheel_pattern.match(original_wheel_name)
        if not match:
            raise RuntimeError(f"Could not parse wheel name: {original_wheel_name}")

        python_tag = match.group(1)
        abi_tag = match.group(2)
        platform_tag = match.group(3)

        # Get our custom version with timestamp and commit hash
        our_version = get_version()

        # Create new wheel filename with our version
        # Note: wheel filenames use underscores for package names with hyphens
        new_wheel_name = (
            f"triton_xdna-{our_version}-{python_tag}-{abi_tag}-{platform_tag}.whl"
        )
        wheel_path = tmpdir / "final" / new_wheel_name
        wheel_path.parent.mkdir(parents=True, exist_ok=True)

        # Create the wheel
        with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in unpack_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(unpack_dir)
                    zf.write(file_path, arcname)

        print(f"  Created: {wheel_path.name}", file=sys.stderr)
        return wheel_path

    def _update_record(self, unpack_dir: Path, record_file: Path):
        """Update RECORD file with all files."""
        import hashlib
        import base64

        records = []

        for file_path in unpack_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(unpack_dir)

                # Skip RECORD itself
                if rel_path.name == "RECORD":
                    continue

                # Calculate hash
                with open(file_path, "rb") as f:
                    content = f.read()
                    sha256 = hashlib.sha256(content).digest()
                    hash_str = (
                        base64.urlsafe_b64encode(sha256).rstrip(b"=").decode("ascii")
                    )

                size = len(content)
                records.append(f"{rel_path},sha256={hash_str},{size}")

        # Add RECORD entry for itself (no hash)
        record_rel = record_file.relative_to(unpack_dir)
        records.append(f"{record_rel},,")

        with open(record_file, "w") as f:
            f.write("\n".join(records) + "\n")

    def _copy_to_dist(self, wheel_path: Path):
        """Copy the final wheel to the dist directory."""
        print("\n[7/7] Copying wheel to dist directory...", file=sys.stderr)

        dist_dir = Path(self.dist_dir) if self.dist_dir else Path("dist")
        dist_dir.mkdir(parents=True, exist_ok=True)

        final_path = dist_dir / wheel_path.name
        shutil.copy2(wheel_path, final_path)

        print(f"  Final wheel: {final_path}", file=sys.stderr)


def _is_triton_installed():
    """Check if triton (or triton-windows) is already installed."""
    try:
        import triton
        return True
    except Exception:
        return False


def _copy_backend_to_triton(backend_src, backend_name):
    """Copy a backend directory into the installed triton's backends dir."""
    import triton
    triton_dir = Path(triton.__file__).parent
    dst = triton_dir / "backends" / backend_name
    dst.mkdir(parents=True, exist_ok=True)

    # Copy Python files
    for py_file in ["compiler.py", "driver.py"]:
        src_file = backend_src / py_file
        if src_file.exists():
            shutil.copy2(src_file, dst / py_file)

    # Ensure __init__.py exists (may not be in source tree)
    init_file = backend_src / "__init__.py"
    if init_file.exists():
        shutil.copy2(init_file, dst / "__init__.py")
    else:
        (dst / "__init__.py").touch()

    # Copy name.conf
    name_conf = backend_src / "name.conf"
    if name_conf.exists():
        shutil.copy2(name_conf, dst / "name.conf")

    # Copy include/ and transform_library/ directories
    for subdir in ["include", "transform_library"]:
        src_subdir = backend_src / subdir
        if src_subdir.exists():
            dst_subdir = dst / subdir
            if dst_subdir.exists():
                shutil.rmtree(dst_subdir)
            shutil.copytree(src_subdir, dst_subdir)

    # Copy xclbin_assemble.py if it exists
    xclbin_file = backend_src / "xclbin_assemble.py"
    if xclbin_file.exists():
        shutil.copy2(xclbin_file, dst / "xclbin_assemble.py")

    print(f"Copied {backend_name} backend to {dst}")


class TritonXdnaDevelop(develop):
    """Development install - uses pip install on triton like before."""

    def run(self):
        if _is_triton_installed():
            # Triton (or triton-windows) is already installed as a wheel.
            # Just copy the backend files into triton's backends directory.
            print("Found pre-installed triton, copying backend files...")
            backend_src = AMD_TRITON_NPU_DIR / "backend"
            _copy_backend_to_triton(backend_src, "amd_triton_npu")
            self._copy_triton_shared_opt()
        else:
            # Build triton from source with plugins
            apply_submodule_patches()
            env = os.environ.copy()
            plugin_dirs = f"{TRITON_SHARED_DIR};{AMD_TRITON_NPU_DIR}"
            env["TRITON_PLUGIN_DIRS"] = plugin_dirs

            if not IS_WINDOWS and check_env_flag("TRITON_BUILD_WITH_CLANG_LLD", "true"):
                env["TRITON_BUILD_WITH_CLANG_LLD"] = "true"

            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                str(TRITON_SOURCE_DIR),
                "--no-build-isolation",
                "-v",
            ]
            subprocess.check_call(cmd, env=env)
            self._copy_triton_shared_opt()

        super().run()

    def _copy_triton_shared_opt(self):
        """Copy triton-shared-opt to installed triton."""
        try:
            import triton

            triton_dir = Path(triton.__file__).parent
        except ImportError:
            print(
                "Warning: triton not installed, skipping triton-shared-opt copy",
                file=sys.stderr,
            )
            return

        triton_shared_opt_binary = find_triton_shared_opt_binary()

        if triton_shared_opt_binary is None:
            print(
                f"Warning: triton-shared-opt binary not found",
                file=sys.stderr,
            )
            return

        triton_shared_dst = triton_dir / "triton_shared"
        triton_shared_dst.mkdir(parents=True, exist_ok=True)

        binary_name = "triton-shared-opt.exe" if IS_WINDOWS else "triton-shared-opt"
        dst_binary = triton_shared_dst / binary_name
        shutil.copy2(triton_shared_opt_binary, dst_binary)
        if not IS_WINDOWS:
            os.chmod(dst_binary, 0o755)


class TritonXdnaInstall(install):
    """Custom install that builds triton with plugins."""

    def run(self):
        if _is_triton_installed():
            # Triton (or triton-windows) is already installed as a wheel.
            # Just copy the backend files into triton's backends directory.
            print("Found pre-installed triton, copying backend files...")
            backend_src = AMD_TRITON_NPU_DIR / "backend"
            _copy_backend_to_triton(backend_src, "amd_triton_npu")

            # Copy triton-shared-opt
            import triton
            triton_dir = Path(triton.__file__).parent
            triton_shared_opt_binary = find_triton_shared_opt_binary()
            if triton_shared_opt_binary is not None:
                triton_shared_dst = triton_dir / "triton_shared"
                triton_shared_dst.mkdir(parents=True, exist_ok=True)
                binary_name = "triton-shared-opt.exe" if IS_WINDOWS else "triton-shared-opt"
                dst_binary = triton_shared_dst / binary_name
                shutil.copy2(triton_shared_opt_binary, dst_binary)
                if not IS_WINDOWS:
                    os.chmod(dst_binary, 0o755)
        else:
            # Build triton from source with plugins
            apply_submodule_patches()

            env = os.environ.copy()
            plugin_dirs = f"{TRITON_SHARED_DIR};{AMD_TRITON_NPU_DIR}"
            env["TRITON_PLUGIN_DIRS"] = plugin_dirs

            pip_install_target = env.get("PIP_INSTALL_TARGET")

            if not IS_WINDOWS and check_env_flag("TRITON_BUILD_WITH_CLANG_LLD", "true"):
                env["TRITON_BUILD_WITH_CLANG_LLD"] = "true"

            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                str(TRITON_SOURCE_DIR),
                "--no-build-isolation",
                "-v",
            ]

            if pip_install_target:
                cmd += ["--target", pip_install_target]

            subprocess.check_call(cmd, env=env)

            # Copy triton-shared-opt
            if pip_install_target:
                triton_dir = Path(pip_install_target) / "triton"
            else:
                import triton
                triton_dir = Path(triton.__file__).parent

            triton_shared_opt_binary = find_triton_shared_opt_binary()
            if triton_shared_opt_binary is not None:
                triton_shared_dst = triton_dir / "triton_shared"
                triton_shared_dst.mkdir(parents=True, exist_ok=True)
                binary_name = "triton-shared-opt.exe" if IS_WINDOWS else "triton-shared-opt"
                dst_binary = triton_shared_dst / binary_name
                shutil.copy2(triton_shared_opt_binary, dst_binary)
                if not IS_WINDOWS:
                    os.chmod(dst_binary, 0o755)

        super().run()


# =============================================================================
# Setup
# =============================================================================

setup(
    name="triton-xdna",
    version=get_version(),
    author="AMD Inc.",
    author_email="",
    description="Triton compiler with MLIR-AIR backend for AMD NPU devices",
    long_description=(BASE_DIR / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/amd/Triton-XDNA",
    license="MIT",
    packages=[],  # No packages - we build from triton
    install_requires=get_install_requires(),
    python_requires=">=3.10",
    cmdclass={
        "bdist_wheel": TritonXdnaBdistWheel,
        "develop": TritonXdnaDevelop,
        "install": TritonXdnaInstall,
    },
    entry_points={
        "triton.backends": [
            "amd_triton_npu = triton.backends.amd_triton_npu",
            "triton_shared = triton.backends.triton_shared",
        ],
    },
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    extras_require={
        "build": [
            "cmake>=3.20",
            "ninja",
            "lit",
        ],
        "tests": [
            "pytest",
            "pytest-xdist",
            "numpy",
            "scipy>=1.7.1",
        ],
    },
)
