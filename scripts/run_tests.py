#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


# Example directories skipped by default unless explicitly selected via --select
DEFAULT_SKIPPED_EXAMPLES = {
    "layernorm",
    "load_2d_block",
    "multi_drivers",
}


def discover_example_dirs(examples_dir: Path, selected: list[str]) -> list[Path]:
    """
    Return a list of subdirectories under examples_dir to test.
    If selected list is non-empty, only include those subdirectory names.
    If no selection is provided, skip a default set of example directories.
    """
    dirs = []
    for entry in sorted(examples_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "__pycache__":
            continue
        # If user provided selections, include only those, even if normally skipped
        if selected:
            if entry.name not in selected:
                continue
        else:
            # No selection: skip default examples
            if entry.name in DEFAULT_SKIPPED_EXAMPLES:
                continue
        dirs.append(entry)
    return dirs


def discover_python_files(example_dir: Path) -> list[Path]:
    """
    Return a list of .py files directly under example_dir (non-recursive).
    """
    return sorted([p for p in example_dir.glob("*.py") if p.is_file()])


def run_python_file(
    py_file: Path,
    cwd: Path,
    transform_file: str | None,
    timeout_sec: int,
    verbose: bool,
    log_f,
) -> tuple[int, str, str]:
    """
    Run a single python file with optional AIR_TRANSFORM_TILING_SCRIPT environment variable.
    Returns (returncode, stdout, stderr).
    """
    env = os.environ.copy()
    if transform_file:
        # Set relative filename, since we're executing with cwd=example directory
        env["AIR_TRANSFORM_TILING_SCRIPT"] = transform_file

    cmd = [sys.executable, py_file.name]

    if log_f:
        log_f.flush()
        log_f.write("\n\n")
        log_f.write("=" * 80 + "\n")
        log_f.write(f"Running {py_file.name} (cwd={cwd})\n")
        log_f.write("=" * 80 + "\n\n")
        if transform_file:
            log_f.write(f"Environment: AIR_TRANSFORM_TILING_SCRIPT={transform_file}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        # Compose timeout outputs - normalize to str
        stdout = (
            e.stdout.decode()
            if isinstance(e.stdout, (bytes, bytearray))
            else (e.stdout or "")
        )
        stderr = (
            e.stderr.decode()
            if isinstance(e.stderr, (bytes, bytearray))
            else (e.stderr or "")
        )
        if verbose:
            print(f"⏰ TIMEOUT: {py_file.name}")
            if stdout:
                print("stdout:\n" + stdout)
            if stderr:
                print("stderr:\n" + stderr)
        if log_f:
            log_f.write("stdout:\n")
            log_f.write(stdout)
            log_f.write("\nstderr:\n")
            log_f.write(stderr)
        return (-1, stdout, stderr)

    if verbose:
        print(f"Command: {' '.join(cmd)}")
        if result.stdout:
            print("stdout:\n" + result.stdout)
        if result.stderr:
            print("stderr:\n" + result.stderr)

    if log_f:
        log_f.write("stdout:\n")
        log_f.write(result.stdout)
        log_f.write("\nstderr:\n")
        log_f.write(result.stderr)

    return (result.returncode, result.stdout, result.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Run each example under the examples directory and report pass/fail."
    )
    parser.add_argument(
        "--examples",
        dest="examples_dir",
        default="examples",
        help="Path to the examples directory (default: examples)",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Optional path to a log file. If it contains {}, it will be formatted with the current date/time.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=450,
        help="Per-test timeout in seconds (default: 450)",
    )
    parser.add_argument(
        "-t",
        "--select",
        default=[],
        action="append",
        help="Select example subdirectory to execute. Can be passed multiple times. Default: all subdirectories.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (prints stdout/stderr)",
    )
    parser.add_argument(
        "--device",
        default="aie2",
        choices=["aie2", "aie2p"],
        help="Target device architecture: aie2 (npu1) or aie2p (npu2). Default: aie2",
    )
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir).absolute()
    if not examples_dir.exists() or not examples_dir.is_dir():
        print(f"❌ Examples directory not found or not a directory: {examples_dir}")
        return 2

    # Prepare logging if requested
    log_f = None
    if args.log:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_path = args.log.format(now)
        log_f = open(log_path, "w")
        # Intentionally leave open for entire run

    # Determine transform filename based on device
    transform_filename = f"transform_{args.device}.mlir"

    print("Starting example test run...")
    print(f"Examples dir: {examples_dir}")
    print(f"Target device: {args.device}")
    print(f"Transform file: {transform_filename}")
    print("-" * 50)

    example_dirs = discover_example_dirs(examples_dir, args.select)
    total_files = 0
    passed = 0
    failed = 0
    timeouts = 0
    skipped = 0

    for ex_dir in example_dirs:
        print(f"📁 Example: {ex_dir.name}")
        py_files = discover_python_files(ex_dir)
        if not py_files:
            print("   (no python files found)")
            print()
            continue

        # Check if transform file exists for the target device
        transform_path = ex_dir / transform_filename
        if transform_path.exists():
            transform_file = transform_filename
            print(
                f"   {transform_filename} detected; will set AIR_TRANSFORM_TILING_SCRIPT"
            )
        else:
            # No transform file for this device - skip the example
            print(
                f"   ⏭️  SKIP: {transform_filename} not found for device {args.device}"
            )
            skipped += 1
            print()
            continue

        for py_file in py_files:
            total_files += 1
            print(f"   🔄 Running: {py_file.name}")

            rc, stdout, stderr = run_python_file(
                py_file=py_file,
                cwd=ex_dir,
                transform_file=transform_file,
                timeout_sec=args.timeout,
                verbose=args.verbose,
                log_f=log_f,
            )

            if rc == 0:
                print(f"   ✅ PASS: {py_file.name}")
                passed += 1
            elif rc == -1:
                print(f"   ⏰ TIMEOUT: {py_file.name}")
                timeouts += 1
            else:
                print(f"   ❌ FAIL: {py_file.name} (exit code {rc})")
                failed += 1
            print()

    # Summary
    print("-" * 50)
    print("Test Results:")
    print(f"  ✅ Passed:  {passed}")
    print(f"  ❌ Failed:  {failed}")
    print(f"  ⏰ Timeouts: {timeouts}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"  📊 Total:   {total_files}")

    if log_f:
        log_f.write("\n" + "-" * 80 + "\n")
        log_f.write("Summary:\n")
        log_f.write(f"Passed: {passed}\n")
        log_f.write(f"Failed: {failed}\n")
        log_f.write(f"Timeouts: {timeouts}\n")
        log_f.write(f"Skipped: {skipped}\n")
        log_f.write(f"Total: {total_files}\n")
        # Intentionally not closing for entire program run

    # Exit code: any failures or timeouts => 1, else 0
    if failed == 0 and timeouts == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"💔 {failed} failed, {timeouts} timed out")
        return 1


if __name__ == "__main__":
    sys.exit(main())
