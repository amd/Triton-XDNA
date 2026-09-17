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
        driver_api: which shape mlir-air's own driver for this model exposes.
            They are not uniform, and the difference is in how our prefill is
            handed to their decode:

            * `"npz"` -- the 1B. `generate(ids, n, seq_len, kv_path, ...)`
              reads a KV handoff file, so the harness writes one and
              neutralizes the step that would have produced it.
            * `"prefiller"` -- the 3B. `generate_stream(dec, tokenizer, ids, n,
              prefiller=...)` takes the prefill object directly, calling
              `clear_context()`, `prefill()` and `kv_view()` on it -- which is
              the interface our prefill already has. No file, nothing to
              neutralize.
    """

    name: str
    decode_env: dict
    model_type: str
    air_package: str
    air_inference: str
    tokenizer_fallback: str
    driver_api: str = "npz"
    #: For `driver_api="prefiller"`: the decoder class that driver exposes.
    #: Named per model (`FusedDecode3B`, `FusedDecode8B`), so it is recorded
    #: rather than derived from the model name.
    decoder_class: str = ""
    #: llms packages to put on sys.path, beyond `air_package`.
    extra_packages: tuple = field(default_factory=tuple)


#: Two places where this spec deliberately does *not* match mlir-air's
#: Makefiles character for character. Both matter when adding a model.
#:
#: `DECODE_MODEL` is pinned here, and upstream does not set it at all: the 1B is
#: the builder's own default (`MODEL_NAME = os.environ.get("DECODE_MODEL",
#: "llama-3.2-1b")`), so neither `llms/llama32_1b_q4nx/Makefile` nor
#: `fused_decode/Makefile` names it. Relying on someone else's default is a
#: thing that breaks silently when the default moves, so every spec names its
#: own model even where upstream can get away with not.
#:
#: `UNIFIED` is absent here because it is absent upstream, and that absence is a
#: fact about this model rather than an omission -- the 3B, Qwen3 and Gemma3
#: Makefiles all set `UNIFIED=1`. Copy each new model's from its Makefile; do
#: not carry this one's over.
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


#: Llama-3.2-3B. Same architecture as the 1B -- SwiGLU, one norm pair per
#: block, no qk-norm -- at 28 layers of 3072 with 128-wide heads.
#:
#: Its environment is assembled from *two* places in
#: `llms/llama32_3b_q4nx/Makefile`, and reading only the obvious one gets it
#: wrong: `DECODE_ENV` (line 51) carries most of it, but `W_DUAL_CHAN=1` reaches
#: the builder through a bare `export` (line 49) instead. That variable picks
#: the DDR weight layout, so dropping it builds cleanly and decodes to garbage.
#:
#: Two differences from the 1B, both deliberate:
#:
#: * `UNIFIED=1` is set here and unset for the 1B -- the per-model fact the 1B's
#:   comment above warns not to carry over in either direction.
#: * `PROJ_RC_CACHE` is absent, because the 3B's Makefile does not set it. It is
#:   read by the builder (`fused_decode.py:1114`) *and* by `proj_qmm.cc`, so the
#:   two must agree -- which they do, both taking the same default. Pinning it
#:   here would not protect that agreement, it would break step 2 of the
#:   verification: if upstream moves the default, air's own build moves with it
#:   and ours would silently stop matching.
LLAMA_3_2_3B = ModelSpec(
    name="llama-3.2-3b",
    decode_env=dict(
        DECODE_MODEL="llama-3.2-3b",
        VOCAB_CHUNK_I2="9",
        UNIFIED="1",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        W_DUAL_CHAN="1",
    ),
    model_type="LLAMA_3_2_3B",
    air_package="llama32_3b_q4nx",
    air_inference="llama32_3b_q4nx_inference.py",
    tokenizer_fallback="~/q4nx_data/tokenizer/Llama-3.2-3B",
    extra_packages=("llama32_3b", "llama32_1b_q4nx"),
    driver_api="prefiller",
    decoder_class="FusedDecode3B",
)


#: Llama-3.1-8B. The 1B's architecture again, at 32 layers of 4096. Nothing
#: new in the forward; what it exercises is size -- the resident decode weights
#: and, on the host, a bf16 dequantization of an 8B model.
#:
#: Two values its Makefile moves that the smaller two leave alone, and both are
#: derived rather than recorded here:
#:
#: * `DECODE_STACK=8064` lowers the AIE core stack from the builder's 10240
#:   default. `decode_build` reads `fused_decode.STACK_SIZE` after importing the
#:   builder under this environment, as mlir-air's own driver does, so it
#:   follows from `DECODE_STACK` below rather than being repeated.
#: * `DECODE_WGROUP=8` splits the DDR weight slab into groups.
#:
#: `W_DUAL_CHAN=1` again arrives by a bare `export`, not in `DECODE_ENV`.
LLAMA_3_1_8B = ModelSpec(
    name="llama-3.1-8b",
    decode_env=dict(
        DECODE_MODEL="llama-3.1-8b",
        VOCAB_CHUNK_I2="16",
        UNIFIED="1",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        DECODE_STACK="8064",
        DECODE_WGROUP="8",
        W_DUAL_CHAN="1",
    ),
    model_type="LLAMA_3_1_8B",
    air_package="llama31_8b_q4nx",
    air_inference="llama31_8b_q4nx_inference.py",
    tokenizer_fallback="~/q4nx_data/tokenizer/Llama-3.1-8B",
    extra_packages=("llama32_3b", "llama32_1b_q4nx"),
    driver_api="prefiller",
    decoder_class="FusedDecode8B",
)


SPECS = {s.name: s for s in (LLAMA_3_2_1B, LLAMA_3_2_3B, LLAMA_3_1_8B)}

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
