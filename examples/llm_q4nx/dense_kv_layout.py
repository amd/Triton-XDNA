# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The KV cache layout of mlir-air's fused decode for the dense (GQA) models.

`kv_layout` is Gemma4-E2B's; this is the same thing for the models the shared
decode engine builds. The prefills (Triton on the host or NPU, mlir-air's fused
NPU prefill, the iGPU one) and the decodes (mlir-air's NPU decode, the iGPU
one) all read and write this layout, so a prefill on one device hands its KV
to a decode on the other in place. Nothing else should compute an offset into
the cache.

As `fused_decode.py` builds it:

    slab[L] = [ K_g0 | K_g1 | ... | V_g0 | V_g1 | ... ]  (then unused padding
                                                         up to the layer stride)

Each region is `attn_maxl x region_w` bf16, row `t` holding position `t`.
Group `g` is one AIE column of attention CUs; its `region_w` lanes are the KV
heads those CUs serve, in head order. So row `t` of `K_g` is
`k[t, g * region_w : (g + 1) * region_w]` of the `[P, n_kv * dh]` roped K, and
V is the raw projection. Unlike Gemma4, nothing is padded or permuted within a
row.

`test_dense_kv_layout.py` holds this against each model's own `seed_kv`.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

K_REGION, V_REGION = 0, 1


@dataclass(frozen=True)
class DenseKVGeometry:
    """Where row `t` of KV head `h` lives in a `[n_layers, lreg]` slab.

    `lreg`, the layer stride, may exceed the `2 * ngrp * attn_maxl * region_w`
    the regions use: a decoder whose current window is narrower than the
    context it was allocated for keeps the wider stride.
    """

    n_layers: int
    attn_maxl: int
    ngrp: int
    region_w: int
    lreg: int = 0

    def __post_init__(self):
        used = 2 * self.ngrp * self.region_stride
        if not self.lreg:
            object.__setattr__(self, "lreg", used)
        elif self.lreg < used:
            raise ValueError(f"layer stride {self.lreg} < the {used} its regions use")

    @property
    def region_stride(self):
        """Elements from one region's base to the next's."""
        return self.attn_maxl * self.region_w

    @property
    def kv_width(self):
        """`n_kv * dh`: one position's K (or V) across every head."""
        return self.ngrp * self.region_w

    def slab_shape(self):
        return (self.n_layers, self.lreg)

    def region_offset(self, region, g):
        """Elements from a layer's base to group `g`'s K or V region."""
        return ((self.ngrp if region == V_REGION else 0) + g) * self.region_stride

    def sink(self, host):
        """The dict `LlamaPrefill.bind_kv_sink` takes, over `host` (flat)."""
        return dict(
            buf=host.reshape(-1),
            region_w=self.region_w,
            ngrp=self.ngrp,
            region_stride=self.region_stride,
            lreg=self.lreg,
        )

    @classmethod
    def from_sink(cls, sink, n_layers=None):
        """The geometry of a decoder's sink (`harness._xrt_kv_sink`,
        `hsa_decode`'s `kv_sink`)."""
        rs, rw = sink["region_stride"], sink["region_w"]
        if rs % rw:
            raise ValueError(f"region stride {rs} is not a multiple of {rw}")
        if n_layers is None:
            n_layers = sink.get("n_layers") or sink["buf"].size // sink["lreg"]
        return cls(n_layers, rs // rw, sink["ngrp"], rw, sink["lreg"])

    @classmethod
    def for_model(cls, model_name, n_kv, dh, n_layers, attn_maxl, engine=None):
        """The geometry mlir-air's decode builds for `model_name`.

        From its `_MODELS` entry (`N_ATTN_CU`, `KV_PER_CU`, `DH_A`), checked
        against the model's own `n_kv` and `dh`. The grouping is the builder's
        placement: four CUs sit two to a column, two CUs share one.
        """
        e = decode_model_entry(model_name, engine)
        n_cu, kv_per_cu, dh_a = e["N_ATTN_CU"], e["KV_PER_CU"], e["DH_A"]
        if n_cu * kv_per_cu != n_kv or dh_a != dh:
            raise ValueError(
                f"{model_name}: mlir-air builds {n_cu} CUs x {kv_per_cu} kv heads "
                f"x {dh_a}, the model has {n_kv} x {dh}"
            )
        if n_cu not in (2, 4):
            raise ValueError(f"{model_name}: no column placement for {n_cu} CUs")
        ngrp = n_cu // 2
        return cls(n_layers, attn_maxl, ngrp, n_kv * dh // ngrp)


def decode_model_entry(model_name, engine=None):
    """The literal fields of `fused_decode.py`'s `_MODELS[model_name]`.

    Parsed rather than imported: importing the builder resolves a whole build
    configuration from the environment at module scope.
    """
    import airsrc

    src = Path(airsrc.fused_decode_dir(engine or "fused_decode")) / "fused_decode.py"
    for node in ast.parse(src.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "_MODELS" for t in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == model_name:
                    out = {}
                    for kw in getattr(v, "keywords", []):
                        try:
                            out[kw.arg] = ast.literal_eval(kw.value)
                        except ValueError:
                            pass
                    return out
            break
    raise KeyError(f"{model_name!r} is not in {src}'s _MODELS")


def region_view(slab, layer_idx, region, g, geom):
    """`[attn_maxl, region_w]` view of one group's K or V region of a
    `[n_layers, lreg]` slab, numpy or torch. No copy."""
    base = geom.region_offset(region, g)
    return slab[layer_idx, base : base + geom.region_stride].reshape(
        geom.attn_maxl, geom.region_w
    )


def scatter(slab, layer_idx, region, rows, geom, pos=0):
    """`rows`, `[P, kv_width]`, into positions `[pos, pos + P)` of one layer's
    K or V. The host writer; the iGPU's is `gpu_kernels.qkv_post_gqa`."""
    P, rw = rows.shape[0], geom.region_w
    if pos + P > geom.attn_maxl:
        raise ValueError(
            f"positions {pos}..{pos + P - 1} exceed the cache's {geom.attn_maxl} rows; "
            "rebuild the decode at a larger ATTN_MAXL or shorten the prompt"
        )
    for g in range(geom.ngrp):
        dst = region_view(slab, layer_idx, region, g, geom)
        dst[pos : pos + P] = rows[:, g * rw : (g + 1) * rw]


def gather(slab, layer_idx, region, geom, P):
    """Positions `[0, P)` of one layer's K or V as `[P, kv_width]`. A copy."""
    parts = [
        region_view(slab, layer_idx, region, g, geom)[:P] for g in range(geom.ngrp)
    ]
    if hasattr(parts[0], "cpu"):
        import torch

        return torch.cat(parts, dim=1)
    import numpy as np

    return np.concatenate(parts, axis=1)
