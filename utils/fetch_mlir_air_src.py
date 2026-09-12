#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fetch mlir-air's sources at the same commit the wheel is pinned to.

`utils/mlir-air-hash.txt` already pins mlir-air: `env_setup.sh` turns its
`Commit:` field into the wheel version (`...+6746658.no.rtti`). Some examples
need mlir-air's *sources* too -- `examples/llama32_1b_q4nx` builds the fused
decode from `fused_decode.build_module()` and compiles its AIE kernels from
`kernels/*.cc`. Those live in `programming_examples/`, which the wheel does
not ship.

Rather than vendor a copy that silently drifts, this checks out the same
commit. One pin, two artifacts: the compiled toolchain and the sources that
generated it, which cannot disagree.

Blobless and sparse -- about 1.7 MB of the two directories actually used,
against 1.2 GB for a full checkout. Idempotent: if the checkout is already at
the pinned commit it does nothing.

    python3 utils/fetch_mlir_air_src.py           # fetch or verify
    python3 utils/fetch_mlir_air_src.py --path    # print the location, fetch if needed
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Xilinx/mlir-air.git"
HASH_FILE = "utils/mlir-air-hash.txt"
DEST = "third_party/mlir-air-src"

# Only what the examples import or compile. `llms` carries the model drivers
# and `shared/`; `fused_decode` carries the builder and the AIE kernels.
SPARSE_PATHS = (
    "programming_examples/fused_decode",
    "programming_examples/llms",
)


def repo_root():
    return Path(__file__).resolve().parent.parent


def pinned_commit():
    path = repo_root() / HASH_FILE
    for line in path.read_text().splitlines():
        if line.startswith("Commit:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"no 'Commit:' line in {path}")


def _git(*args, cwd=None, check=True, quiet=False):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=quiet,
        text=True,
    )


def at_commit(dest, commit):
    """True if `dest` is a checkout already sitting on `commit`."""
    if not (dest / ".git").exists():
        return False
    try:
        head = _git("rev-parse", "HEAD", cwd=dest, quiet=True).stdout.strip()
    except subprocess.CalledProcessError:
        return False
    # The pin is abbreviated; compare on the shorter of the two.
    return head.startswith(commit) or commit.startswith(head[: len(commit)])


def fetch(dest=None, commit=None, quiet=False):
    """Ensure a sparse checkout of mlir-air at the pinned commit. Returns its path."""
    dest = Path(dest) if dest else repo_root() / DEST
    commit = commit or pinned_commit()

    if at_commit(dest, commit):
        if not quiet:
            print(f"[air-src] {dest} already at {commit}", flush=True)
        return dest

    if not (dest / ".git").exists():
        if dest.exists() and any(dest.iterdir()):
            raise SystemExit(f"{dest} exists and is not a git checkout; remove it")
        if not quiet:
            print(f"[air-src] cloning {REPO_URL} (blobless, sparse)...", flush=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git(
            "clone",
            "--filter=blob:none",
            "--sparse",
            "--no-checkout",
            REPO_URL,
            str(dest),
        )
        _git("sparse-checkout", "set", *SPARSE_PATHS, cwd=dest)

    if not quiet:
        print(f"[air-src] checking out {commit}...", flush=True)
    # The commit may predate what a shallow clone fetched, or postdate it.
    try:
        _git("checkout", "--detach", commit, cwd=dest, quiet=True)
    except subprocess.CalledProcessError:
        _git("fetch", "--filter=blob:none", "origin", commit, cwd=dest)
        _git("checkout", "--detach", commit, cwd=dest)

    if not quiet:
        print(f"[air-src] {dest} at {commit}", flush=True)
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--commit", default=None, help="override the pin (for testing)")
    ap.add_argument(
        "--path",
        action="store_true",
        help="print the checkout path on stdout (fetching if needed)",
    )
    args = ap.parse_args(argv)
    dest = fetch(args.dest, args.commit, quiet=args.path)
    if args.path:
        print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
