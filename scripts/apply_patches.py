#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Apply local patches to third-party submodules before building.

Thin CLI over scripts/patching.py, which holds the actual implementation and is
shared with setup.py so the two cannot drift apart again.

Usage:
    python scripts/apply_patches.py [--reset] [--force] [--reset-only]

Options:
    --reset       Reset submodules to clean state before applying patches
    --force       Force re-apply patches even if marker exists
    --reset-only  Only reset submodules, don't apply patches
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patching import PatchError, apply_patches, reset_all_submodules


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
        return 0 if reset_all_submodules() else 1

    try:
        apply_patches(force=args.force, reset=args.reset)
    except PatchError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
