#!/usr/bin/env python3
"""Check `dense_kv_layout` against each dense model's own `seed_kv`, byte for byte.

`test_kv_layout.py` for the models mlir-air's shared decode engine builds. The
layout is mlir-air's, restated here, and restating it wrongly does not crash:
the first token comes from the prefill and only the decode reads the cache, so
a wrong offset shows up as fluent nonsense after it. So the check is equality
of the bytes against upstream's code, not a tolerance on logits.

For each model, upstream's decoder `seed_kv` runs as it is, on an object that
holds only what it reads, and `dense_kv_layout.scatter` fills the same K/V. The
geometry comes from mlir-air's decode builder, and is also checked against
`DenseKVGeometry.for_model`, which the iGPU path uses when no decoder exists.

Needs no hardware, but does need mlir-air's sources and its Python bindings.
Each model runs in its own process: the drivers share module names.

Not picked up by `scripts/run_tests.py` (`llm_q4nx` is a library). Exits 77
when mlir-air's sources are not present.

    python test_dense_kv_layout.py [--model llama-3.2-1b]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = (
    "llama-3.2-1b",
    "llama-3.2-3b",
    "llama-3.1-8b",
    "qwen3-4b",
    "qwen3-8b",
    "phi4-mini",
    "gemma3-4b",
    "qwen2.5-3b",
    "qwen2.5-7b",
)

#: A small cache, and a prompt that fills part of it: the scatter is per row,
#: so a ragged P against the window is the case worth having.
ATTN_MAXL, P = 48, 19


class _HostBO:
    """Where `seed_kv` uploads: host memory taking writes at byte offsets."""

    def __init__(self, nbytes):
        self.buf = np.zeros(nbytes, np.uint8)

    def write(self, buf, off=0):
        data = np.ascontiguousarray(buf).view(np.uint8).reshape(-1)
        self.buf[off : off + data.size] = data

    def sync(self, *a):
        pass


def _builder(spec):
    """mlir-air's decode builder module, configured for `spec`'s model."""
    import airsrc

    fdd = airsrc.fused_decode_dir(spec.engine)
    sys.path.insert(0, fdd)
    os.environ.update({k: str(v) for k, v in spec.decode_env.items()})
    os.environ["DECODE_MODEL"] = spec.name
    s = importlib.util.spec_from_file_location(
        "fused_decode", os.path.join(fdd, "fused_decode.py")
    )
    fd = importlib.util.module_from_spec(s)
    s.loader.exec_module(fd)
    return fd


def _upstream_slab(cls, fd, n, ks, vs, bf16):
    """The decoder's `seed_kv` itself, on an object with only what it reads."""
    from dense_kv_layout import DenseKVGeometry

    geom = DenseKVGeometry(n, ATTN_MAXL, fd.NGRP, fd.REGION_W)
    from kv_bos import KVBuffers

    dec = object.__new__(cls)
    # The decoder's own KV buffers, over host memory.
    kvb = KVBuffers(fd, ATTN_MAXL, lambda nbytes, arg: _HostBO(nbytes))
    to = 0
    for name, value in dict(
        np=np,
        bf16=bf16,
        xrt=SimpleNamespace(
            xclBOSyncDirection=SimpleNamespace(XCL_BO_SYNC_BO_TO_DEVICE=to)
        ),
        TO=to,
        kvb=kvb,
        kvc=kvb.kvc,
        fd=SimpleNamespace(REGION_W=fd.REGION_W, NGRP=fd.NGRP),
        REGION_W=fd.REGION_W,
        NGRP=fd.NGRP,
        ATTN_MAXL=ATTN_MAXL,
        cur_maxl=ATTN_MAXL,
        N_LAYERS=n,
        UNI=n,
        UNI_DEC=n,
        LREG=geom.lreg,
        _geom=SimpleNamespace(lreg=lambda maxl: 2 * fd.NGRP * maxl * fd.REGION_W),
        KV=np.zeros(geom.slab_shape(), dtype=bf16),
    ).items():
        setattr(dec, name, value)
    import inspect

    if "P" in inspect.signature(cls.seed_kv).parameters:
        dec.seed_kv(np.stack(ks), np.stack(vs), P)
        return np.asarray(dec.KV).reshape(geom.slab_shape())
    dec.seed_kv(ks, vs)
    kv = np.concatenate([bo.buf for bo in kvb.bos]).view(bf16)
    return kv[: np.prod(geom.slab_shape())].reshape(geom.slab_shape())


def check_model(name):
    import harness
    import registry
    from ml_dtypes import bfloat16

    from dense_kv_layout import K_REGION, V_REGION, DenseKVGeometry, scatter

    spec = registry.spec(name)
    try:
        fd = _builder(spec)
        air = harness.air_inference_module(spec)
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: mlir-air's {name} sources are not importable: {e}")
        return 77
    n_kv, dh = fd.NUM_KV_HEADS, fd.DH_A
    n = fd.MODEL["UNI_DEC"]
    ok = True

    ours = DenseKVGeometry.for_model(name, n_kv, dh, n, ATTN_MAXL, spec.engine)
    same = (ours.ngrp, ours.region_w) == (fd.NGRP, fd.REGION_W)
    print(
        f"  {name}: builder NGRP={fd.NGRP} REGION_W={fd.REGION_W}, for_model "
        f"{ours.ngrp} {ours.region_w}  {'ok' if same else 'FAIL'}"
    )
    ok &= same

    rng = np.random.default_rng(0)
    ks = [rng.standard_normal((P, n_kv * dh), dtype=np.float32) for _ in range(n)]
    vs = [rng.standard_normal((P, n_kv * dh), dtype=np.float32) for _ in range(n)]
    cls = getattr(air, spec.decoder_class or "FusedDecoder")
    want = _upstream_slab(cls, fd, n, ks, vs, bfloat16)

    got = np.zeros(ours.slab_shape(), dtype=bfloat16)
    for L in range(n):
        scatter(got, L, K_REGION, ks[L].astype(bfloat16), ours)
        scatter(got, L, V_REGION, vs[L].astype(bfloat16), ours)
    bad = [
        L
        for L in range(n)
        if not np.array_equal(want[L].view(np.uint16), got[L].view(np.uint16))
    ]
    print(
        f"  {name}: {n} layers, ATTN_MAXL={ATTN_MAXL}, P={P}, "
        f"{cls.__name__}.seed_kv  "
        + ("identical" if not bad else f"DIFFER on layers {bad}")
    )
    ok &= not bad
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    args = ap.parse_args()
    if args.model:
        return check_model(args.model)

    rcs = []
    for name in MODELS:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--model", name])
        rcs.append(r.returncode)
    if all(rc == 77 for rc in rcs):
        return 77
    ok = all(rc in (0, 77) for rc in rcs)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
