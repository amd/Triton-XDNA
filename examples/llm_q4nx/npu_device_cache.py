# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Building a `FusedDevice` once: the toolchain aircc reads, and a cache
directory named by a digest of everything the build is made from.

Shared by the devices built here from Triton kernels (`gemma4_triton_fused`,
`gemma4_vision_device`).
"""

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def toolchain():
    """Peano and mlir-aie, as the NPU backend resolves them."""
    from triton.backends.amd_triton_npu.driver import find_peano_root
    from triton.language.extra.npu.fused_prefill import _mlir_aie_root

    peano = find_peano_root()
    if not peano:
        raise RuntimeError("no Peano (llvm-aie) install found for the AIE kernels")
    return str(peano), _mlir_aie_root()


def versions():
    """The installed compiler stack, by distribution."""
    import importlib.metadata as md

    out = []
    for names in (
        ("mlir_air", "mlir-air"),
        ("mlir_aie_no_rtti", "mlir_aie"),
        ("llvm-aie",),
    ):
        for n in names:
            try:
                out.append(f"{n}=={md.version(n)}")
                break
            except md.PackageNotFoundError:
                pass
    return out


@contextlib.contextmanager
def toolchain_env():
    """aircc reads the toolchain from the environment and leaves the working
    directory where it compiled."""
    peano, aie = toolchain()
    saved = {
        k: os.environ.get(k) for k in ("PEANO_INSTALL_DIR", "MLIR_AIE_INSTALL_DIR")
    }
    cwd = os.getcwd()
    os.environ.update(PEANO_INSTALL_DIR=peano, MLIR_AIE_INSTALL_DIR=aie)
    try:
        yield
    finally:
        os.chdir(cwd)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


#: Backend modules that run a build but take no part in making it.
RUNTIME_ONLY = ("fused_program.py", "npu_runner.py")


def fingerprint(sources, extra=()):
    """Digest of what a device is made from: `sources` (files) and `extra`
    (anything JSON), the NPU backend and Triton that compile it, and the
    compiler stack (its versions and where it is installed)."""
    import triton
    import triton.backends.amd_triton_npu as backend

    h = hashlib.sha256()
    h.update(json.dumps(list(extra)).encode())
    h.update(json.dumps([triton.__version__, *versions(), *toolchain()]).encode())
    backend_dir = Path(backend.__file__).parent
    backend_sources = sorted(
        p
        for p in backend_dir.rglob("*")
        if p.suffix in (".py", ".mlir") and p.name not in RUNTIME_ONLY
    )
    for p in [*sources, *backend_sources]:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def cached(name, digest, files, build, cache_root=None):
    """`build(out)` into `<cache root>/<name>_<digest>` unless a complete one
    (every name in `files`) is there. Returns `build_dir` and `cached`."""
    import fcntl

    from triton.language.extra.npu.fused_prefill import default_cache_root

    root = Path(cache_root or default_cache_root())
    root.mkdir(parents=True, exist_ok=True)
    current = root / f"{name}_{digest}"

    def complete(d):
        return all((Path(d) / f).is_file() for f in files)

    with open(root / f".{current.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if current.is_dir() and complete(current):
            return dict(build_dir=str(current), cached=True)
        tmp = Path(tempfile.mkdtemp(prefix=f"{current.name}.", dir=root))
        try:
            build(tmp)
            if not complete(tmp):
                raise RuntimeError(f"{name} build incomplete: {tmp}")
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        if current.exists():
            shutil.rmtree(current)
        os.replace(tmp, current)
    return dict(build_dir=str(current), cached=False)
