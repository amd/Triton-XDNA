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

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install


try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    from setuptools.command.bdist_wheel import bdist_wheel


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent.resolve()
THIRD_PARTY_DIR = BASE_DIR / "third_party"
TRITON_SOURCE_DIR = THIRD_PARTY_DIR / "triton"
TRITON_SHARED_DIR = THIRD_PARTY_DIR / "triton_shared"
AMD_TRITON_NPU_DIR = BASE_DIR / "amd_triton_npu"

# Patch configuration: (submodule_name, patch_file)
PATCHES = [
    ("triton", "triton.patch"),
    ("triton_shared", "triton_shared.patch"),
]

# Marker file name to track if patches have been applied
PATCH_MARKER_FILE = ".patches_applied"


def find_triton_shared_opt_binary(triton_source_dir: Path = None) -> Path:
    """
    Find the triton-shared-opt binary in the build directory.

    The binary is located at:
    triton/build/cmake.*/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt

    Returns:
        Path to the binary if found, None otherwise.
    """
    if triton_source_dir is None:
        triton_source_dir = TRITON_SOURCE_DIR

    build_dir = triton_source_dir / "build"

    if not build_dir.exists():
        return None

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


def get_install_requires():
    """Build install_requires list from hash files."""
    mlir_aie_hash_file = BASE_DIR / "utils" / "mlir-aie-hash.txt"
    mlir_air_hash_file = BASE_DIR / "utils" / "mlir-air-hash.txt"
    llvm_aie_hash_file = BASE_DIR / "utils" / "llvm-aie-hash.txt"

    # Parse mlir-aie version
    mlir_aie_version = parse_hash_file(mlir_aie_hash_file, "Version")
    mlir_aie_timestamp = parse_hash_file(mlir_aie_hash_file, "Timestamp")
    mlir_aie_commit = parse_hash_file(mlir_aie_hash_file, "Commit")
    mlir_aie_short_commit = mlir_aie_commit[:7]
    mlir_aie_full_version = (
        f"{mlir_aie_version}.{mlir_aie_timestamp}+{mlir_aie_short_commit}.no.rtti"
    )

    # Parse llvm-aie version
    llvm_aie_version = parse_hash_file(llvm_aie_hash_file, "Version")
    llvm_aie_timestamp = parse_hash_file(llvm_aie_hash_file, "Timestamp")
    llvm_aie_commit = parse_hash_file(llvm_aie_hash_file, "Commit")
    llvm_aie_full_version = f"{llvm_aie_version}.{llvm_aie_timestamp}+{llvm_aie_commit}"

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
        f"llvm-aie=={llvm_aie_full_version}",
        f"mlir-air=={mlir_air_full_version}",
    ]


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

        # Set build flags
        if check_env_flag("TRITON_BUILD_WITH_CLANG_LLD", "true"):
            env["TRITON_BUILD_WITH_CLANG_LLD"] = "true"

        triton_source = project_root / "third_party" / "triton"

        # Use subprocess with pip wheel - this properly inherits environment
        # Find a working Python with pip
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

        # Get project root
        cwd = Path.cwd()
        if (cwd / "third_party" / "triton").exists():
            project_root = cwd
        else:
            project_root = BASE_DIR

        triton_source = project_root / "third_party" / "triton"
        triton_shared_opt_binary = find_triton_shared_opt_binary(triton_source)

        if triton_shared_opt_binary is None:
            print(
                f"  Warning: triton-shared-opt binary not found in {triton_source}/build/",
                file=sys.stderr,
            )
            return

        print(f"  Found binary: {triton_shared_opt_binary}", file=sys.stderr)

        # Copy binary to triton package
        triton_shared_dst = unpack_dir / "triton" / "triton_shared"
        triton_shared_dst.mkdir(parents=True, exist_ok=True)

        dst_binary = triton_shared_dst / "triton-shared-opt"
        shutil.copy2(triton_shared_opt_binary, dst_binary)

        # Ensure the binary has execute permissions
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


class TritonXdnaDevelop(develop):
    """Development install - uses pip install on triton like before."""

    def run(self):
        # Apply patches before building
        apply_submodule_patches()

        env = os.environ.copy()
        plugin_dirs = f"{TRITON_SHARED_DIR};{AMD_TRITON_NPU_DIR}"
        env["TRITON_PLUGIN_DIRS"] = plugin_dirs

        if check_env_flag("TRITON_BUILD_WITH_CLANG_LLD", "true"):
            env["TRITON_BUILD_WITH_CLANG_LLD"] = "true"

        # Install triton with plugins
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

        # Copy triton-shared-opt
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

        dst_binary = triton_shared_dst / "triton-shared-opt"
        shutil.copy2(triton_shared_opt_binary, dst_binary)
        os.chmod(dst_binary, 0o755)


class TritonXdnaInstall(install):
    """Custom install that builds triton with plugins."""

    def run(self):
        # Apply patches before building
        apply_submodule_patches()

        env = os.environ.copy()
        plugin_dirs = f"{TRITON_SHARED_DIR};{AMD_TRITON_NPU_DIR}"
        env["TRITON_PLUGIN_DIRS"] = plugin_dirs

        pip_install_target = env.get("PIP_INSTALL_TARGET")

        if check_env_flag("TRITON_BUILD_WITH_CLANG_LLD", "true"):
            env["TRITON_BUILD_WITH_CLANG_LLD"] = "true"

        # Install triton with plugins
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
            dst_binary = triton_shared_dst / "triton-shared-opt"
            shutil.copy2(triton_shared_opt_binary, dst_binary)
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
