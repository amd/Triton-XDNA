# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Locate MSVC and reproduce a developer-prompt environment.

Used from two places that both need a working ``cl.exe``:

* ``setup.py``, to build triton-windows. It loads this module by path
  (importlib) rather than importing it, so nothing here may import triton.
* ``driver.py``, to JIT-compile the NPU dispatch shim at kernel launch.

Both previously assumed the caller had already run ``vcvars64.bat``; outside
such a shell the build failed with a CMake message about the ``CXX``
environment variable that never mentioned Visual Studio. Rather than
hand-assembling INCLUDE/LIB -- which is easy to get subtly wrong, and misses
tools like ``rc.exe`` and ``mt.exe`` that the LLVM build needs -- we run the
real ``vcvars64.bat`` and capture the environment it produces.

Only the standard library is used, and every entry point returns ``None``
rather than raising, so callers can fall back.
"""

import functools
import os
import shutil
import subprocess
import sys

IS_WINDOWS = sys.platform == "win32"

# Component id for the x64 C++ toolset. Asking vswhere to require it avoids
# selecting a Visual Studio install that has no C++ workload.
_VC_TOOLS_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"

# Printed between vcvars' own chatter and the `set` dump so we can tell the
# two apart. vcvars writes a banner to stdout and offers no quiet flag; a
# redirect would be simpler but is awkward to quote portably through cmd.
_ENV_MARKER = "__TRITON_XDNA_ENV_BEGIN__"


def find_vswhere():
    """Path to vswhere.exe, or None. It ships with VS 2017+ at a fixed location."""
    if not IS_WINDOWS:
        return None
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(
        program_files_x86, "Microsoft Visual Studio", "Installer", "vswhere.exe"
    )
    return vswhere if os.path.isfile(vswhere) else None


@functools.lru_cache(maxsize=1)
def find_vs_install():
    """Installation root of the newest VS with the x64 C++ toolset, or None.

    Deliberately not pinned to a Visual Studio year: ``-latest`` picks up
    VS 2022 and VS 18 alike.
    """
    vswhere = find_vswhere()
    if vswhere is None:
        return None
    try:
        out = subprocess.check_output(
            [
                vswhere,
                "-latest",
                "-products",
                "*",
                "-requires",
                _VC_TOOLS_COMPONENT,
                "-property",
                "installationPath",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines[0] if lines else None


@functools.lru_cache(maxsize=1)
def find_vcvars():
    """Path to vcvars64.bat for the newest suitable VS install, or None."""
    vs_path = find_vs_install()
    if vs_path is None:
        return None
    vcvars = os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvars64.bat")
    return vcvars if os.path.isfile(vcvars) else None


@functools.lru_cache(maxsize=1)
def vcvars_env():
    """Environment produced by vcvars64.bat as a dict, or None.

    Spawning cmd is not cheap, so the result is cached for the process.
    """
    vcvars = find_vcvars()
    if vcvars is None:
        return None
    # shell=True and a single command string: passing the quoted .bat path as a
    # list element makes subprocess escape the quotes for cmd, which then
    # cannot find the file.
    command = f'"{vcvars}" && echo {_ENV_MARKER} && set'
    try:
        out = subprocess.check_output(
            command,
            shell=True,
            text=True,
            errors="replace",
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    if _ENV_MARKER not in out:
        return None

    env = {}
    for line in out.split(_ENV_MARKER, 1)[1].splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value
    # A vcvars environment without INCLUDE did not actually initialise.
    return env if env.get("INCLUDE") else None


def in_developer_environment():
    """True if the current process already looks like a VS developer prompt."""
    return bool(os.environ.get("INCLUDE"))


def find_cl(env=None):
    """Absolute path to cl.exe, or None.

    Looks on the current PATH first (already inside a developer prompt), then
    in *env* if given, then in a freshly captured vcvars environment.
    """
    if not IS_WINDOWS:
        return None

    on_path = shutil.which("cl.exe") or shutil.which("cl")
    if on_path:
        return on_path

    for candidate_env in (env, vcvars_env()):
        if candidate_env:
            found = shutil.which("cl.exe", path=candidate_env.get("PATH", ""))
            if found:
                return found
    return None


SETUP_HINT = (
    "Visual Studio with the C++ toolset was not found. Triton-XDNA needs MSVC "
    "to compile NPU dispatch code.\n"
    "Options:\n"
    "  1. Install Visual Studio with the 'Desktop development with C++' "
    "workload (https://visualstudio.microsoft.com/)\n"
    "  2. Install the Build Tools for Visual Studio "
    "(https://visualstudio.microsoft.com/visual-cpp-build-tools/)\n"
    "  3. Run from an 'x64 Native Tools Command Prompt' so vcvars64.bat has "
    "already been applied"
)
