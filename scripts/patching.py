#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Apply the local patches in third_party/ to their submodules.

This is the single implementation, shared by ``setup.py`` (build time) and
``scripts/apply_patches.py`` (the standalone CLI). It previously existed as two
near-identical copies that had drifted apart in two ways that both broke
Windows: only one passed ``--ignore-whitespace``, and only the other knew that
Windows builds ``triton-windows`` rather than ``triton``.

Line endings are the crux. The patches are authored with LF, but git's Windows
default ``core.autocrlf=true`` rewrites both the patch files and the submodule
working trees to CRLF on checkout, and a CRLF/LF mismatch on either side makes
``git apply`` reject the hunks. Three defences, in order:

1. ``.gitattributes`` pins ``third_party/*.patch`` to LF, so the patch side is
   fixed no matter how the user has git configured. It cannot help the
   submodule side -- attributes do not cross into a submodule's own checkout.
2. ``--ignore-whitespace``, which tolerates EOL differences in context lines.
3. ``--3way``, which matches on the blob SHAs recorded in the patch's ``index``
   lines instead of on textual context, and so survives both EOL differences
   and modest context drift.

A patch that still will not apply raises :class:`PatchError`. It must not be
skipped: an unpatched triton_shared builds for roughly twenty minutes and then
fails with a C2397 narrowing-conversion error in ``PtrAnalysis.cpp`` that the
patch exists to prevent, which is a miserable way to learn that patching was
silently a no-op.
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIR = BASE_DIR / "third_party"

IS_WINDOWS = sys.platform == "win32"

# Marker file dropped in a submodule once its patch is applied, so repeated
# builds do not re-run the patch machinery.
MARKER_FILE = ".patches_applied"

# (submodule directory, patch file). Windows builds triton-windows instead of
# triton -- keep this in step with TRITON_SOURCE_DIR in setup.py.
PATCHES = [
    (
        ("triton-windows", "triton-windows.patch")
        if IS_WINDOWS
        else ("triton", "triton.patch")
    ),
    ("triton_shared", "triton_shared.patch"),
]


class PatchError(RuntimeError):
    """A patch could not be applied and the build must not continue."""


def _git(args, cwd, check=False):
    return subprocess.run(
        ["git"] + args, cwd=str(cwd), capture_output=True, text=True, check=check
    )


def _log(message):
    print(message, file=sys.stderr)


def reset_submodule(submodule_dir: Path) -> bool:
    """Discard local changes in *submodule_dir*, dropping any applied patch."""
    _log(f"  Resetting {submodule_dir.name}...")
    try:
        _git(["checkout", "."], cwd=submodule_dir, check=True)
        _git(["clean", "-fd"], cwd=submodule_dir, check=True)
    except subprocess.CalledProcessError as e:
        _log(f"  x Failed to reset {submodule_dir.name}: {e.stderr}")
        return False

    marker = submodule_dir / MARKER_FILE
    if marker.exists():
        marker.unlink()
    _log(f"  + Reset {submodule_dir.name}")
    return True


def reset_all_submodules() -> bool:
    """Reset every configured submodule to a clean state."""
    _log("=" * 60)
    _log("Resetting submodules to clean state")
    _log("=" * 60)

    ok = True
    for submodule_name, _ in PATCHES:
        submodule_dir = THIRD_PARTY_DIR / submodule_name
        if not submodule_dir.exists():
            _log(f"  ! Submodule directory not found: {submodule_dir}")
            continue
        if not reset_submodule(submodule_dir):
            ok = False
    return ok


def _already_applied(patch_file: Path, target_dir: Path) -> bool:
    """True if *patch_file* is already present in *target_dir*."""
    return (
        _git(
            ["apply", "--check", "--reverse", "--ignore-whitespace", str(patch_file)],
            cwd=target_dir,
        ).returncode
        == 0
    )


# git apply strategies, in escalating order of tolerance. Each entry is
# (extra flags, short description used in diagnostics).
_STRATEGIES = [
    (["--ignore-whitespace"], "ignore-whitespace"),
    (["--3way", "--ignore-whitespace"], "3-way merge"),
]


def _is_clean(target_dir: Path) -> bool:
    """True if the submodule working tree has no local modifications."""
    result = _git(["status", "--porcelain"], cwd=target_dir)
    return result.returncode == 0 and not result.stdout.strip()


def _apply_one(patch_file: Path, target_dir: Path) -> None:
    """Apply *patch_file* in *target_dir*, or raise :class:`PatchError`.

    Failure is atomic when we started from a clean tree: --3way can apply a
    patch *with conflict markers* and still report failure, and leaving a
    half-merged vendored submodule behind would be worse than not trying.
    """
    started_clean = _is_clean(target_dir)

    attempts = []
    for flags, label in _STRATEGIES:
        if "--3way" in flags and not started_clean:
            # Refuse to 3-way merge into a tree we cannot safely roll back.
            attempts.append(
                f"    via {label}: skipped, {target_dir.name} has local changes"
            )
            continue
        # Dry-run first so a failing strategy cannot leave a partial apply
        # behind. --3way is exempt: it stages merge results and has no
        # meaningful --check, so it is attempted directly.
        if "--3way" not in flags:
            check = _git(
                ["apply", "--check"] + flags + [str(patch_file)], cwd=target_dir
            )
            if check.returncode != 0:
                attempts.append(f"    via {label}: {check.stderr.strip()}")
                continue

        result = _git(["apply"] + flags + [str(patch_file)], cwd=target_dir)
        if result.returncode == 0:
            _log(f"  + Applied {patch_file.name} ({label})")
            return
        attempts.append(f"    via {label}: {result.stderr.strip()}")

    if started_clean and not _is_clean(target_dir):
        # A strategy (in practice --3way) left partial or conflicted content.
        # We know the tree was clean going in, so this restores exactly the
        # prior state rather than discarding anyone's work.
        _log(f"  Rolling back partial apply in {target_dir.name}")
        _git(["reset", "--hard"], cwd=target_dir)

    detail = "\n".join(attempts)
    raise PatchError(
        f"Could not apply {patch_file.name} to third_party/{target_dir.name}.\n"
        f"{detail}\n"
        "\n"
        "The usual cause on Windows is git's CRLF conversion rewriting the\n"
        "submodule working tree. Configure LF and re-checkout, matching what\n"
        "CI does (.github/workflows/nightly-wheels.yml):\n"
        "    git config --global core.autocrlf false\n"
        "    git config --global core.eol lf\n"
        "    git submodule foreach --recursive git reset --hard\n"
        "Otherwise the submodule pin and the patch have genuinely diverged."
    )


def apply_patches(force: bool = False, reset: bool = False) -> None:
    """Apply every configured patch. Raises :class:`PatchError` on failure.

    Args:
        force: re-apply even when the marker file says it was already done.
        reset: reset each submodule to a clean state first.
    """
    _log("=" * 60)
    _log("Checking/applying patches to submodules")
    _log("=" * 60)

    for submodule_name, patch_name in PATCHES:
        submodule_dir = THIRD_PARTY_DIR / submodule_name
        patch_file = THIRD_PARTY_DIR / patch_name
        marker_file = submodule_dir / MARKER_FILE

        _log(f"\n[{submodule_name}]")

        if not submodule_dir.exists():
            # An uninitialised submodule is not a patch failure -- the build
            # will report it far more clearly than we can here.
            _log(f"  ! Submodule directory not found: {submodule_dir}")
            continue

        if not patch_file.exists():
            raise PatchError(f"Patch file not found: {patch_file}")

        if reset and not reset_submodule(submodule_dir):
            raise PatchError(f"Failed to reset third_party/{submodule_name}")

        if marker_file.exists() and not force:
            _log("  + Patches already applied (marker exists)")
            continue

        if _already_applied(patch_file, submodule_dir):
            _log("  + Patch already applied")
            marker_file.touch()
            continue

        _log(f"  Applying {patch_name}...")
        _apply_one(patch_file, submodule_dir)
        marker_file.touch()

    _log("\n" + "=" * 60)
    _log("All patches applied successfully.")
    _log("=" * 60)
