# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""``tl.extra.npu.fused_prefill`` -- mlir-air's one-device chunked prefill.

The prefill counterpart of `fused_decode`. mlir-air's Gemma4-E2B
`fused_prefill/` configures the array once and runs a prompt in fixed-size
chunks, one instruction stream per op. Its builder compiles the AIE kernels and
every op, and checks that the ops agree on one device configuration.

This op calls that builder rather than restating it, and adds what a caller
here needs around it: the toolchain this backend resolves (Peano and mlir-aie),
a content fingerprint so an earlier build is reused instead of recompiled, and
a completeness check before a build is handed out.

The builder runs in a subprocess because it changes directory per op and runs a
process pool, neither of which belongs in the caller's process.

The builder's shapes come from Gemma4-E2B's constants, so `PrefillConfig`
refuses any other model instead of building something that would run and
produce wrong output.
"""

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

#: Models mlir-air has a fused prefill builder for.
MODELS = ("gemma4-e2b",)

#: Sources outside `fused_prefill/` that the builder compiles, relative to
#: mlir-air's `programming_examples/`. Whole directories, because the kernels
#: include headers that sit beside them.
KERNEL_SOURCE_DIRS = (
    "matrix_multiplication/bf16_in_fp32_out",
    "flash_attention/kernel_fusion_based",
)

#: Distributions whose version changes what the builder emits.
TOOLCHAIN = ("mlir_air", "mlir_aie_no_rtti", "mlir_aie", "llvm-aie")

_BUILD_LOCK = threading.Lock()


class PrefillConfigError(ValueError):
    pass


class PrefillConfig:
    """What a fused prefill build depends on besides its sources."""

    def __init__(self, model, jobs=8):
        if model not in MODELS:
            raise PrefillConfigError(
                f"no fused prefill for {model!r}; mlir-air has one for {MODELS}"
            )
        if int(jobs) < 1:
            raise PrefillConfigError(f"jobs must be >= 1, got {jobs}")
        self.model = model
        self.jobs = int(jobs)

    def __repr__(self):
        return f"PrefillConfig(model={self.model!r}, jobs={self.jobs})"


def _toolchain_versions():
    out = {}
    for dist in TOOLCHAIN:
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def _hash_tree(h, root, base):
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(base)).encode())
            h.update(p.read_bytes())


def fingerprint(config, fused_prefill_dir, peano_root):
    """Digest of everything the build reads: sources, model constants, tools."""
    d = Path(fused_prefill_dir).resolve()
    examples = d.parents[2]
    h = hashlib.sha256()
    h.update(config.model.encode())
    _hash_tree(h, d, examples)
    # The model's constants: `device.py` and `build.py` size every op from them.
    h.update((d.parent / "gemma4_e2b_q4nx_weights.py").read_bytes())
    for rel in KERNEL_SOURCE_DIRS:
        src = examples / rel
        if not src.is_dir():
            raise FileNotFoundError(
                f"fused prefill kernel sources missing: {src}. Fetch them with "
                f"utils/fetch_mlir_air_src.py."
            )
        _hash_tree(h, src, examples)
    h.update(json.dumps(_toolchain_versions(), sort_keys=True).encode())
    h.update(str(peano_root).encode())
    return h.hexdigest()[:16]


def _mlir_aie_root():
    env = os.environ.get("MLIR_AIE_INSTALL_DIR")
    if env:
        return env
    import aie

    # aie/__init__.py sits in <root>/python/aie/ in both the wheel and an install.
    return str(Path(aie.__file__).resolve().parents[2])


def _complete(build_dir):
    """True if `build_dir` holds a manifest and every artifact it names."""
    m = Path(build_dir) / "manifest.json"
    if not m.is_file():
        return False
    man = json.loads(m.read_text())
    names = list(man["gemm"]) + [man["lm"]]
    names += [n for pts in man["attn"].values() for n in pts.values()]
    return all(
        (Path(build_dir) / f"{n}{ext}").is_file()
        for n in names
        for ext in (".xclbin", ".insts.bin")
    )


def default_cache_root():
    base = os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")
    return os.path.join(base, "npu_fused_prefill")


def fused_prefill(config, fused_prefill_dir, cache_root=None, rebuild=False):
    """Build (or reuse) mlir-air's fused prefill for `config`.

    Returns a dict: ``build_dir`` (what `FusedPrefill(build_dir)` loads),
    ``manifest``, ``fingerprint``, ``cached`` (True when nothing was built) and
    ``config``.

    Args:
        config: a `PrefillConfig`.
        fused_prefill_dir: mlir-air's `programming_examples/llms/
            gemma4_e2b_q4nx/fused_prefill`.
        cache_root: where builds live, one directory per fingerprint. Defaults
            to `npu_fused_prefill/` under Triton's cache directory.
        rebuild: build even if a complete build with this fingerprint exists.
    """
    from triton.backends.amd_triton_npu.driver import find_peano_root

    if not isinstance(config, PrefillConfig):
        raise PrefillConfigError(
            f"config must be a PrefillConfig, got {type(config).__name__}"
        )
    d = Path(fused_prefill_dir).resolve()
    if not (d / "build.py").is_file():
        raise FileNotFoundError(f"no fused prefill builder at {d}")
    peano = find_peano_root()
    if not peano:
        raise RuntimeError("no Peano (llvm-aie) install found for the AIE kernels")
    fp = fingerprint(config, d, peano)
    root = Path(cache_root or default_cache_root())
    out = root / f"{config.model}_{fp}"
    with _BUILD_LOCK:
        if out.is_dir() and _complete(out) and not rebuild:
            cached = True
        else:
            root.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=root))
            env = dict(
                os.environ,
                PEANO_INSTALL_DIR=str(peano),
                MLIR_AIE_INSTALL_DIR=_mlir_aie_root(),
            )
            cmd = [
                sys.executable,
                str(d / "build.py"),
                str(tmp),
                "-j",
                str(config.jobs),
            ]
            r = subprocess.run(cmd, env=env, cwd=tmp, capture_output=True, text=True)
            if r.returncode != 0 or not _complete(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
                raise RuntimeError(
                    f"fused prefill build failed (rc={r.returncode}):\n"
                    f"{r.stdout[-2000:]}\n{r.stderr[-4000:]}"
                )
            shutil.rmtree(tmp / "work", ignore_errors=True)
            (tmp / "fingerprint.json").write_text(
                json.dumps(
                    dict(
                        fingerprint=fp,
                        model=config.model,
                        toolchain=_toolchain_versions(),
                        peano=str(peano),
                        sources=str(d),
                    ),
                    indent=1,
                )
            )
            if out.exists():
                shutil.rmtree(out)
            # Atomic on one filesystem: a concurrent reader sees the old
            # directory or the new one, never a partial build.
            os.replace(tmp, out)
            cached = False
    return dict(
        build_dir=str(out),
        manifest=json.loads((out / "manifest.json").read_text()),
        fingerprint=fp,
        cached=cached,
        config=config,
    )
