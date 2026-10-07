# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""``tl.extra.npu.fused_prefill`` -- mlir-air's one-device chunked prefill.

The prefill counterpart of `fused_decode`. mlir-air's fused prefill
(`llms/shared/fused_prefill/`) configures the array once and runs a prompt in
fixed-size chunks, one instruction stream per op. Its builder compiles the AIE
kernels and every op, and checks that the ops agree on one device
configuration. The dense models are rows of its model table
(`dense_main.py build MODEL`); Gemma4-E2B has its own spec and `build.py`
beside its driver, on the same shared builder.

This op calls that builder rather than restating it, and adds what a caller
here needs around it: the toolchain this backend resolves (Peano and mlir-aie),
a content fingerprint so an earlier build is reused instead of recompiled, and
a completeness check before a build is handed out.

The builder runs in a subprocess because it changes directory per op and runs a
process pool, neither of which belongs in the caller's process.

The builder's shapes come from each model's constants, so `PrefillConfig`
refuses a model mlir-air has no entry for instead of building something that
would run and produce wrong output.
"""

import fcntl
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

#: Models mlir-air has a fused prefill for, by this repository's model name:
#: the key of the model in `shared/fused_prefill/models.py`, or None for
#: Gemma4-E2B, whose builder is `gemma4_e2b_q4nx/fused_prefill/build.py`.
MODELS = {
    "gemma4-e2b": None,
    "llama-3.2-1b": "llama32_1b_q4nx",
    "llama-3.2-3b": "llama32_3b_q4nx",
    "llama-3.1-8b": "llama31_8b_q4nx",
    "qwen3-4b": "qwen3_4b_q4nx",
    "qwen3-8b": "qwen3_8b_q4nx",
    "phi4-mini": "phi4_mini_q4nx",
    "gemma3-4b": "gemma3_4b_q4nx",
}

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
                f"no fused prefill for {model!r}; mlir-air has one for "
                f"{', '.join(MODELS)}"
            )
        if int(jobs) < 1:
            raise PrefillConfigError(f"jobs must be >= 1, got {jobs}")
        self.model = model
        self.jobs = int(jobs)
        #: The model's key in mlir-air's dense model table, or None.
        self.dense = MODELS[model]

    @property
    def package(self):
        """The `llms/` directory holding this model's builder: `shared` for a
        dense model, `airsrc.fused_prefill_dir(config.package)`."""
        return "shared" if self.dense else "gemma4_e2b_q4nx"

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


def fingerprint(config, fused_prefill_dir, peano_root, aie_root):
    """Digest of everything the build reads: sources, model constants, tools."""
    d = Path(fused_prefill_dir).resolve()
    examples = d.parents[2]
    shared = examples / "llms" / "shared" / "fused_prefill"
    h = hashlib.sha256()
    h.update(config.model.encode())
    # The shared builder and device, which every model's build runs on; for a
    # dense model `d` is that directory, and its model table is in it.
    if not shared.is_dir():
        raise FileNotFoundError(f"mlir-air's shared fused prefill missing: {shared}")
    _hash_tree(h, shared, examples)
    if not config.dense:
        _hash_tree(h, d, examples)
        # The model's constants: its spec sizes every op from them.
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
    # The builder compiles against this installation's AIE headers.
    h.update(str(Path(aie_root).resolve()).encode())
    return h.hexdigest()[:16]


def _mlir_aie_root():
    env = os.environ.get("MLIR_AIE_INSTALL_DIR")
    if env:
        return env
    import aie

    # aie/__init__.py sits in <root>/python/aie/ in both the wheel and an install.
    return str(Path(aie.__file__).resolve().parents[2])


def _complete(build_dir):
    """True if `build_dir` holds a manifest, every artifact it names, and the
    host library the runtime loads when it is constructed."""
    m = Path(build_dir) / "manifest.json"
    if not m.is_file() or not (Path(build_dir) / "libhostops.so").is_file():
        return False
    man = json.loads(m.read_text())
    # A dense model's LM head runs as GEMMs, so it has no "lm" op.
    names = list(man["gemm"]) + ([man["lm"]] if man.get("lm") else [])
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
        fused_prefill_dir: the model's builder directory under mlir-air's
            `programming_examples/llms/`: `<config.package>/fused_prefill`.
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
    builder = "dense_main.py" if config.dense else "build.py"
    if not (d / builder).is_file():
        raise FileNotFoundError(f"no fused prefill builder at {d / builder}")
    peano = find_peano_root()
    if not peano:
        raise RuntimeError("no Peano (llvm-aie) install found for the AIE kernels")
    aie_root = _mlir_aie_root()
    fp = fingerprint(config, d, peano, aie_root)
    root = Path(cache_root or default_cache_root())
    root.mkdir(parents=True, exist_ok=True)
    # `current` names the published generation. Generations are never modified
    # or removed once published, so a reader holding one keeps a complete
    # build; publishing a new one only switches the link.
    current = root / f"{config.model}_{fp}"
    with _BUILD_LOCK, open(root / f".{current.name}.lock", "w") as lock:
        # Builders in other processes wait here, then reuse what the first
        # one published.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if current.is_dir() and _complete(current) and not rebuild:
            cached = True
        else:
            gen = Path(tempfile.mkdtemp(prefix=f"{current.name}.g", dir=root))
            env = dict(
                os.environ, PEANO_INSTALL_DIR=str(peano), MLIR_AIE_INSTALL_DIR=aie_root
            )
            cmd = [sys.executable, str(d / builder)]
            cmd += ["build", config.dense] if config.dense else []
            cmd += [str(gen), "-j", str(config.jobs)]
            r = subprocess.run(cmd, env=env, cwd=gen, capture_output=True, text=True)
            if r.returncode != 0 or not _complete(gen):
                shutil.rmtree(gen, ignore_errors=True)
                raise RuntimeError(
                    f"fused prefill build failed (rc={r.returncode}):\n"
                    f"{r.stdout[-2000:]}\n{r.stderr[-4000:]}"
                )
            shutil.rmtree(gen / "work", ignore_errors=True)
            (gen / "fingerprint.json").write_text(
                json.dumps(
                    dict(
                        fingerprint=fp,
                        model=config.model,
                        toolchain=_toolchain_versions(),
                        peano=str(peano),
                        mlir_aie=aie_root,
                        sources=str(d),
                    ),
                    indent=1,
                )
            )
            if current.is_dir() and not current.is_symlink():
                # A build published as a plain directory becomes a generation
                # of its own, so the link below can take its name.
                current.rename(root / f"{current.name}.g{os.urandom(4).hex()}")
            link = root / f".{current.name}.link"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(gen.name)
            os.replace(link, current)
            cached = False
        build_dir = current.resolve()
    return dict(
        build_dir=str(build_dir),
        manifest=json.loads((build_dir / "manifest.json").read_text()),
        fingerprint=fp,
        cached=cached,
        config=config,
    )
