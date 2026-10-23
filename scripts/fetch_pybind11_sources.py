#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Check out the Triton and triton_shared sources for the pybind11 build.

The commits come from utils/pybind11-pins.txt and land in third_party/pybind11/.
setup.py calls this when TRITON_XDNA_BINDINGS=pybind11; it can also be run on
its own. A checkout already at its pinned commit is left alone, patches and
all; one at any other commit is reset to the pin.
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PINS_FILE = BASE_DIR / "utils" / "pybind11-pins.txt"
DEST_DIR = BASE_DIR / "third_party" / "pybind11"

REPOS = {
    "triton": "https://github.com/triton-lang/triton.git",
    "triton_shared": "https://github.com/facebookincubator/triton-shared.git",
}


def read_pins(path: Path = PINS_FILE) -> dict:
    pins = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, sha = line.partition(":")
        pins[name.strip()] = sha.strip()
    missing = set(REPOS) - set(pins)
    if missing:
        sys.exit(f"{path} has no pin for: {', '.join(sorted(missing))}")
    return pins


def _git(args, cwd: Path) -> str:
    return subprocess.run(
        ["git"] + args, cwd=str(cwd), check=True, capture_output=True, text=True
    ).stdout.strip()


def _head(repo: Path):
    try:
        return _git(["rev-parse", "HEAD"], repo)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def fetch(name: str, url: str, sha: str) -> Path:
    dest = DEST_DIR / name
    if _head(dest) == sha:
        print(f"  {name}: at {sha[:12]}", file=sys.stderr)
        return dest

    print(f"  {name}: fetching {sha[:12]} from {url}", file=sys.stderr)
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        _git(["init", "-q"], dest)
        _git(["remote", "add", "origin", url], dest)
    _git(["fetch", "-q", "--depth", "1", "origin", sha], dest)
    _git(["checkout", "-q", "-f", "--detach", "FETCH_HEAD"], dest)
    _git(["clean", "-q", "-fdx"], dest)
    return dest


def fetch_all() -> None:
    pins = read_pins()
    for name, url in REPOS.items():
        fetch(name, url, pins[name])


if __name__ == "__main__":
    fetch_all()
