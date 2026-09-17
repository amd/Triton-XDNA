# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""What the decode build needs to know about a model.

mlir-air's `fused_decode` engine drives every model family from one builder;
which one it builds is module-level state read from the environment at import
time (`_MODELS` / `MODEL_NAME` in `fused_decode.py`). `build_module()` takes no
arguments, so the environment is the only interface there is.

Each spec below is copied **verbatim** from that model's Makefile in
mlir-air -- its `DECODE_ENV`, its `-DMODEL_TYPE`. These are not defaults chosen
for being reasonable: reconstructing them produces a module that builds
cleanly and then decodes to garbage, because the AIE kernels are compiled
against constants the builder also derives. So they are copied, and a new model
is added by reading its Makefile, not by reasoning about it.

`GLU_SLICE_EXPECTED` is deliberately absent: it is *derived* from the builder
under the model's own environment rather than recorded here, exactly as
mlir-air's Makefile derives it (see `decode_kernels.glu_slice_expected`).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """One model family, as the decode build and the example driver see it.

    Attributes:
        name: this repo's name for it, and the `--model` value.
        decode_env: environment the builder is imported under. Verbatim from
            mlir-air's Makefile for this model, including `DECODE_MODEL`.
        model_type: the `-DMODEL_TYPE` the AIE kernels are compiled with.
        air_package: the `llms/<pkg>` directory holding mlir-air's own driver.
        air_inference: that driver's module filename.
        tokenizer_fallback: used only when mlir-air's driver is not loaded,
            i.e. `--prefill-only`.
    """

    name: str
    decode_env: dict
    model_type: str
    air_package: str
    air_inference: str
    tokenizer_fallback: str
    #: llms packages to put on sys.path, beyond `air_package`.
    extra_packages: tuple = field(default_factory=tuple)


#: The 1B's environment has no DECODE_MODEL because the builder's default is
#: llama-3.2-1b, and no UNIFIED because its Makefile sets neither. Stated
#: explicitly all the same: a spec that relies on someone else's default is a
#: spec that breaks silently when the default moves.
LLAMA_3_2_1B = ModelSpec(
    name="llama-3.2-1b",
    decode_env=dict(
        DECODE_MODEL="llama-3.2-1b",
        VOCAB_CHUNK_I2="18",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        W_DUAL_CHAN="1",
        PROJ_RC_CACHE="1",
    ),
    model_type="LLAMA_3_2_1B",
    air_package="llama32_1b_q4nx",
    air_inference="llama32_1b_q4nx_inference.py",
    tokenizer_fallback="~/q4nx_data/tokenizer/Llama-3.2-1B",
    extra_packages=("llama32_1b",),
)


SPECS = {s.name: s for s in (LLAMA_3_2_1B,)}

#: What `--model` defaults to where a single model is implied.
DEFAULT = LLAMA_3_2_1B.name


def spec(name=None):
    """Look up a model spec by name, with a listing on a miss."""
    name = name or DEFAULT
    try:
        return SPECS[name]
    except KeyError:
        raise SystemExit(
            f"unknown model {name!r}; known: {', '.join(sorted(SPECS))}"
        ) from None
