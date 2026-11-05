# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""``tl.extra.npu.flash_attention``: mlir-air's flash attention for one shape.

mlir-air's `flash_attention/kernel_fusion_based/attn_npu2_seqfirst` writes the
dataflow (Q/K/V tiling, the K/V cascade across core rows, the online softmax)
with mlir-air's Python builder; the per-core math is the C++ microkernel
`attn_npu2.cc`. This op builds that design for one `AttentionConfig` and
returns the ELF.

As with `fused_decode`, the module comes from mlir-air's builder and is
lowered by this backend's `_aircc_compile` through `FusedDecodeOp`. The
microkernel is compiled with mlir-air's helper against the Peano and mlir-aie
this backend resolves. As in `fused_prefill`, each build is stored under a
content fingerprint and published only once complete.

Q, K and V are row-major `[rows, heads * head_dim]` bf16, and so is the output.
The kernel scales the scores by `1/sqrt(head_dim)` itself; a caller that wants
another scale multiplies Q by `scale * sqrt(head_dim)` first.
"""

import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
from pathlib import Path

from .fused_prefill import _hash_tree, _mlir_aie_root, _toolchain_versions

#: Query rows per launch tile, and the four 64-row q tiles inside it.
Q_TILE = 256
Q_TILES = 4
#: Key/value rows per cascade stage; also the only head dim the kernel takes.
KV_STAGE_ROWS = 64

#: The builder and the microkernel, relative to `flash_attention_dir`.
SOURCES = ("attn_npu2_seqfirst.py", "attn_npu2.cc")

#: aircc options mlir-air's own callers build this design with: no ping-pong
#: transform, and runtime loops tiled 1x1.
AIRCC_ARGS = ("--omit-ping-pong-transform=all",)
RUNTIME_LOOP_TILING = (1, 1)

_BUILD_LOCK = threading.Lock()


class AttentionConfigError(ValueError):
    pass


class AttentionConfig:
    """One flash attention shape.

    Args:
        rows: query rows, which are also the key/value rows (self-attention).
            Queries are padded to a multiple of 256; keys and values are not,
            because the kernel has no key mask and a padded key would take part
            in the softmax. So `rows` must divide into whole cascade stages:
            a multiple of 256 (four stages) or of 192 (three).
        heads: query heads.
        kv_heads: key/value heads; `heads` must be a multiple. Defaults to
            `heads` (multi-head attention).
        head_dim: 64, the kernel's K/V tile.
        bfp16: build the microkernel's matmuls in block-float mode. Faster
            but less accurate; enable it only where the caller's accuracy
            check passes with it.
    """

    def __init__(self, rows, heads, kv_heads=None, head_dim=64, bfp16=False):
        rows, heads = int(rows), int(heads)
        kv_heads = heads if kv_heads is None else int(kv_heads)
        if head_dim != KV_STAGE_ROWS:
            raise AttentionConfigError(
                f"head_dim must be {KV_STAGE_ROWS}, the kernel's K/V tile; got {head_dim}"
            )
        if heads < 2 or heads % 2 or kv_heads < 1 or heads % kv_heads:
            raise AttentionConfigError(
                f"heads={heads}, kv_heads={kv_heads}: heads must be even (two "
                f"per unroll) and a multiple of kv_heads"
            )
        if self._stages(rows) is None:
            raise AttentionConfigError(
                f"rows={rows} does not divide into whole cascade stages "
                f"({KV_STAGE_ROWS * 4} or {KV_STAGE_ROWS * 3} rows), and keys "
                f"cannot be padded"
            )
        self.rows, self.heads, self.kv_heads = rows, heads, kv_heads
        self.head_dim, self.bfp16 = head_dim, bool(bfp16)

    @staticmethod
    def _stages(rows):
        for s in (4, 3):
            if rows > 0 and rows % (KV_STAGE_ROWS * s) == 0:
                return s
        return None

    @property
    def q_rows(self):
        """Rows of the Q and output buffers: `rows`, padded to the q tile."""
        return math.ceil(self.rows / Q_TILE) * Q_TILE

    def builder_kwargs(self):
        """`attn_npu2_seqfirst.build_module`'s arguments for this shape."""
        return dict(
            lk=self.rows,
            lkp=self.head_dim,
            lq=self.q_rows,
            lqp=Q_TILE,
            dk=self.head_dim,
            dv=self.head_dim,
            num_q_tiles=Q_TILES,
            num_cascade_stages=self._stages(self.rows),
            num_heads=self.heads,
            num_kv_heads=self.kv_heads,
            causal=False,
            num_heads_per_unroll=2,
            fused_qkv=False,
            n_images=1,
            # The q loop inside the segment needs fewer waves, but past four q
            # tiles the runtime sequence needs more buffer descriptors at once
            # than a shim tile has (16).
            q_in_segment=self.q_rows <= Q_TILES * Q_TILE,
        )

    def __repr__(self):
        return (
            f"AttentionConfig(rows={self.rows}, heads={self.heads}, "
            f"kv_heads={self.kv_heads}, head_dim={self.head_dim}, bfp16={self.bfp16})"
        )


def fingerprint(config, flash_attention_dir, peano_root, aie_root):
    """Digest of everything the build reads: shape, sources, tools."""
    d = Path(flash_attention_dir).resolve()
    h = hashlib.sha256()
    h.update(json.dumps(config.builder_kwargs(), sort_keys=True).encode())
    h.update(b"bfp16" if config.bfp16 else b"bf16")
    for name in SOURCES:
        if not (d / name).is_file():
            raise FileNotFoundError(
                f"flash attention source missing: {d / name}. Fetch it with "
                f"utils/fetch_mlir_air_src.py."
            )
    _hash_tree(h, d, d.parent)
    h.update(json.dumps(_toolchain_versions(), sort_keys=True).encode())
    h.update(str(peano_root).encode())
    h.update(str(Path(aie_root).resolve()).encode())
    return h.hexdigest()[:16]


def _complete(build_dir):
    m = Path(build_dir) / "kernel.json"
    return m.is_file() and (Path(build_dir) / "attn.elf").is_file()


def default_cache_root():
    base = os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")
    return os.path.join(base, "npu_flash_attention")


@contextlib.contextmanager
def _scoped_env(values):
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _cwd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _build(config, d, gen, peano, aie_root):
    """Compile the microkernel and lower the module into `gen`."""
    import importlib.util
    import sys

    from triton.backends.amd_triton_npu.fused_decode_op import FusedDecodeOp

    examples = d.parents[1]
    # mlir-air's examples import each other as `air_examples.*`.
    pkg = sys.modules.get("air_examples")
    if pkg is None:
        import types

        pkg = sys.modules.setdefault("air_examples", types.ModuleType("air_examples"))
        pkg.__path__ = [str(examples)]
    from air_examples.llms.shared.infra.external_kernels import compile_attn_npu2

    spec = importlib.util.spec_from_file_location(
        "_npu_flash_attention_builder", d / SOURCES[0]
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    env = dict(PEANO_INSTALL_DIR=str(peano), MLIR_AIE_INSTALL_DIR=str(aie_root))
    with _scoped_env(env), _cwd(gen):
        compile_attn_npu2(
            head_dim=config.head_dim,
            bfp16=config.bfp16,
            force=True,
            pretranspose_k=True,
        )
        op = FusedDecodeOp.from_builder(
            lambda: builder.build_module(**config.builder_kwargs()),
            kernel_objects=[str(gen / "attn_npu2.o")],
            name="flash_attention",
            stack_size=2048,
            aircc_args=AIRCC_ARGS,
            runtime_loop_tiling_sizes=RUNTIME_LOOP_TILING,
        )
        with _scoped_env({"AMD_TRITON_NPU_AIR_PROJECT_PATH": str(gen / "work")}):
            art = op.lower(output_format="elf")
    shutil.copy2(art["elf_path"], gen / "attn.elf")
    return art["elf_kernel_name"]


def flash_attention(config, flash_attention_dir, cache_root=None, rebuild=False):
    """Build (or reuse) mlir-air's flash attention for `config`.

    Returns a dict: ``elf_path`` and ``kernel`` (what `MultiLaunchRunner`
    loads), ``q_rows`` (rows of the Q and output buffers), ``fingerprint``,
    ``cached`` (True when nothing was built) and ``config``.

    Args:
        config: an `AttentionConfig`.
        flash_attention_dir: mlir-air's `programming_examples/flash_attention/
            kernel_fusion_based`.
        cache_root: where builds live, one directory per fingerprint. Defaults
            to `npu_flash_attention/` under Triton's cache directory.
        rebuild: build even if a complete build with this fingerprint exists.
    """
    from triton.backends.amd_triton_npu.driver import find_peano_root

    if not isinstance(config, AttentionConfig):
        raise AttentionConfigError(
            f"config must be an AttentionConfig, got {type(config).__name__}"
        )
    d = Path(flash_attention_dir).resolve()
    peano = find_peano_root()
    if not peano:
        raise RuntimeError("no Peano (llvm-aie) install found for the microkernel")
    aie_root = _mlir_aie_root()
    fp = fingerprint(config, d, peano, aie_root)
    # Absolute: the build runs from inside the generation directory.
    root = Path(cache_root or default_cache_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # `current` names the published generation, as in `fused_prefill`: a
    # generation is never modified once published, and publishing a new one
    # only switches the link.
    current = root / f"{config.q_rows}x{config.rows}_{fp}"
    with _BUILD_LOCK, open(root / f".{current.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if current.is_dir() and _complete(current) and not rebuild:
            cached = True
        else:
            gen = Path(tempfile.mkdtemp(prefix=f"{current.name}.g", dir=root))
            try:
                kernel = _build(config, d, gen, peano, aie_root)
            except BaseException:
                shutil.rmtree(gen, ignore_errors=True)
                raise
            shutil.rmtree(gen / "work", ignore_errors=True)
            (gen / "kernel.json").write_text(
                json.dumps(
                    dict(
                        kernel=kernel,
                        fingerprint=fp,
                        config=repr(config),
                        toolchain=_toolchain_versions(),
                        peano=str(peano),
                        mlir_aie=aie_root,
                        sources=str(d),
                    ),
                    indent=1,
                )
            )
            if current.is_dir() and not current.is_symlink():
                current.rename(root / f"{current.name}.g{os.urandom(4).hex()}")
            link = root / f".{current.name}.link"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(gen.name)
            os.replace(link, current)
            cached = False
        build_dir = current.resolve()
    meta = json.loads((build_dir / "kernel.json").read_text())
    return dict(
        elf_path=str(build_dir / "attn.elf"),
        kernel=meta["kernel"],
        q_rows=config.q_rows,
        fingerprint=fp,
        cached=cached,
        config=config,
    )
