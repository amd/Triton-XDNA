#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Apply local patches to third-party submodules before building.

This script applies patch files from third_party/ to their respective submodules.
It uses marker files to track whether patches have been applied to avoid
re-applying them on subsequent builds.

Usage:
    python scripts/apply_patches.py [--reset] [--force]

Options:
    --reset    Reset submodules to clean state before applying patches
    --force    Force re-apply patches even if marker exists
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent.resolve()
THIRD_PARTY_DIR = BASE_DIR / "third_party"

# Patch configuration: (submodule_name, patch_file)
PATCHES = [
    ("triton", "triton.patch"),
    ("triton_shared", "triton_shared.patch"),
]

# Marker file name to track if patches have been applied
MARKER_FILE = ".patches_applied"


def run_git(args: list, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in the specified directory."""
    cmd = ["git"] + args
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )


def reset_submodule(submodule_dir: Path) -> bool:
    """Reset a submodule to its clean state."""
    print(f"  Resetting {submodule_dir.name}...", file=sys.stderr)

    try:
        # Discard all local changes
        run_git(["checkout", "."], cwd=submodule_dir)
        # Remove untracked files
        run_git(["clean", "-fd"], cwd=submodule_dir)

        # Remove marker file if it exists
        marker = submodule_dir / MARKER_FILE
        if marker.exists():
            marker.unlink()

        print(f"  ✓ Reset {submodule_dir.name}", file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to reset {submodule_dir.name}: {e.stderr}", file=sys.stderr)
        return False


def check_patch_applicable(patch_file: Path, target_dir: Path) -> tuple[bool, str]:
    """
    Check if a patch can be applied.

    Returns:
        (can_apply, reason) tuple
    """
    # --ignore-whitespace tolerates CRLF/LF differences in context lines,
    # which matters on Windows where git checkout converts text files to
    # CRLF by default but our patches were generated with LF.
    result = run_git(
        ["apply", "--check", "--ignore-whitespace", str(patch_file)],
        cwd=target_dir,
        check=False,
    )

    if result.returncode == 0:
        return True, "Patch can be applied"

    # Check if patch is already applied by trying reverse
    result_reverse = run_git(
        ["apply", "--check", "--reverse", "--ignore-whitespace", str(patch_file)],
        cwd=target_dir,
        check=False,
    )

    if result_reverse.returncode == 0:
        return False, "Patch already applied"

    return False, f"Patch conflict: {result.stderr.strip()}"


def apply_patch(patch_file: Path, target_dir: Path) -> bool:
    """Apply a patch file to the target directory."""
    try:
        run_git(["apply", "--ignore-whitespace", str(patch_file)], cwd=target_dir)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to apply patch: {e.stderr}", file=sys.stderr)
        return False


def apply_patches(force: bool = False, reset: bool = False) -> bool:
    """
    Apply all configured patches to their submodules.

    Args:
        force: Force re-apply patches even if marker exists
        reset: Reset submodules before applying patches

    Returns:
        True if all patches were applied successfully, False otherwise
    """
    print("=" * 60, file=sys.stderr)
    print("Applying patches to submodules", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    all_success = True

    for submodule_name, patch_name in PATCHES:
        submodule_dir = THIRD_PARTY_DIR / submodule_name
        patch_file = THIRD_PARTY_DIR / patch_name
        marker_file = submodule_dir / MARKER_FILE

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

        # Reset if requested
        if reset:
            if not reset_submodule(submodule_dir):
                all_success = False
                continue

        # Check if already applied (marker exists)
        if marker_file.exists() and not force:
            print(f"  ✓ Patches already applied (marker exists)", file=sys.stderr)
            continue

        # Check if patch can be applied
        can_apply, reason = check_patch_applicable(patch_file, submodule_dir)

        if not can_apply:
            if "already applied" in reason.lower():
                print(f"  ✓ {reason}", file=sys.stderr)
                # Create marker file
                marker_file.touch()
            else:
                print(f"  ✗ {reason}", file=sys.stderr)
                all_success = False
            continue

        # Apply the patch
        print(f"  Applying {patch_name}...", file=sys.stderr)
        if apply_patch(patch_file, submodule_dir):
            print(f"  ✓ Patch applied successfully", file=sys.stderr)
            # Create marker file
            marker_file.touch()
        else:
            all_success = False

    print("\n" + "=" * 60, file=sys.stderr)
    if all_success:
        print("All patches applied successfully!", file=sys.stderr)
    else:
        print("Some patches failed to apply.", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    return all_success


def reset_all_submodules() -> bool:
    """Reset all configured submodules to clean state."""
    print("=" * 60, file=sys.stderr)
    print("Resetting submodules to clean state", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    all_success = True

    for submodule_name, _ in PATCHES:
        submodule_dir = THIRD_PARTY_DIR / submodule_name

        if not submodule_dir.exists():
            print(
                f"  ⚠ Submodule directory not found: {submodule_dir}", file=sys.stderr
            )
            continue

        if not reset_submodule(submodule_dir):
            all_success = False

    return all_success


def main():
    parser = argparse.ArgumentParser(
        description="Apply local patches to third-party submodules"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset submodules to clean state before applying patches",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-apply patches even if marker exists",
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Only reset submodules, don't apply patches",
    )

    args = parser.parse_args()

    if args.reset_only:
        success = reset_all_submodules()
    else:
        success = apply_patches(force=args.force, reset=args.reset)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
