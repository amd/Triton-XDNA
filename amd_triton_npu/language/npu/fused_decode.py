# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""``tl.extra.npu.fused_decode`` -- one dispatch for a whole decoder stack.

The op every Q4NX example's decode goes through. What it replaces is a set of
**process-global environment variables**: `DECODE_MODEL`, `VOCAB_CHUNK_I2`,
`NLAYERS`, `DECODE_GOLDEN_L` and the rest were read by mlir-air's builder at
import time, which made the configuration a property of the *process* rather
than of the call. Two consequences, both gone here:

* a caller could only ever build one configuration, so `decode_build.py` forked
  a subprocess per context length;
* nothing checked the values, so a typo produced a module that built cleanly and
  then decoded to garbage.

Here they are arguments. `DecodeConfig` names them, validates what it can, and
`fused_decode()` applies them to mlir-air's builder for the duration of one
build and restores the environment afterwards. Building several configurations
in one process works -- mlir-air's module re-reads its constants on each fresh
`exec_module`, verified across model, layer count, vocab chunking and context
length.

This is stage one of three, and the staging is deliberate:

1. *this file* -- the op exists in the language, carries its own configuration,
   and lowers through this backend's `_aircc_compile` like any other AIR design.
   The graph is still built by mlir-air's Python builder.
2. the graph moves into Triton: the layer loop, the KV append and the per-token
   instruction patch expressed as Triton IR and lowered triton-shared -> linalg
   -> AIR, rather than assembled with `air.dialects`.
3. the six hand-written AIE kernels (`proj_qmm`, `attn_qk`, `attn_kv`, `rope`,
   `glu`, `rms_residual`) stop being Peano-compiled C++ linked into the module.

Stage 2's real obstacle is not Triton but the transform scripts: the matmul
schedule this backend reuses cannot currently absorb even a K loop (it pins
`iterator_interchange` to four loops), so a whole decode graph coming out of
triton-shared needs schedules written for it.
"""

import contextlib
import hashlib
import os
import threading

#: Knobs mlir-air's builder reads from the environment, and the attribute on
#: `DecodeConfig` that supplies each. Anything not listed here is left alone, so
#: a builder that grows a knob keeps its own default until it is named.
_ENV = {
    "DECODE_MODEL": "model",
    "DECODE_GOLDEN_L": "context_length",
    "VOCAB_CHUNK_I2": "vocab_chunk",
    "NLAYERS": "layers_per_dispatch",
    "UNIFIED": "unified",
    "LM_HEAD": "lm_head",
    "DECODE_GOLDEN": "golden",
    "W_DUAL_CHAN": "dual_channel",
    "DECODE_WGROUP": "weight_group",
    "DECODE_STACK": "stack_size",
    "PROJ_RC_CACHE": "proj_rc_cache",
}


class DecodeConfigError(ValueError):
    """The requested decode configuration cannot be built."""


class DecodeConfig:
    """What one fused decode is built for.

    Every field is copied from the model's Makefile in mlir-air rather than
    reasoned out -- the AIE kernels are compiled against constants the builder
    derives from these, so a plausible-looking wrong value builds cleanly and
    decodes to nonsense. `examples/llm_q4nx/registry.py` holds the per-model
    values and the reasoning for each.

    Args:
        model: mlir-air's name for the family, e.g. ``"qwen3-4b"``. Required and
            never defaulted: the builder's own default is Llama-3.2-1B, and
            silently building the wrong model is the failure this op exists to
            make impossible.
        model_type: the ``-DMODEL_TYPE`` this model's AIE kernels were compiled
            with, e.g. ``"QWEN3_4B"``. Checked against the objects that are
            linked; see `_verify_kernels`.
        context_length: the ``L`` the template is built for. The artifact serves
            every length up to ``16*ceil(L/16)``.
        vocab_chunk: this model's ``VOCAB_CHUNK_I2``. Its legal values depend on
            the model's own core geometry, so it does not transfer between
            models even within a family.
        layers_per_dispatch, unified, lm_head, golden, dual_channel: the
            remaining Makefile knobs, as strings or ints.
        weight_group: layers per weight buffer, for models past the 4 GiB
            one-BO limit. ``0`` disables it.
        stack_size: AIE core stack. ``None`` takes the builder's own, which is
            what every model but the 8B wants.
        proj_rc_cache: the builder's ``PROJ_RC_CACHE``, which selects the cached
            -- rather than recomputed -- reduction in ``proj_qmm.cc``. The
            builder and the kernel read it separately and must agree, so a model
            that moves it off the default has to say so here. ``None`` leaves the
            builder's default (1), which is what every model currently takes.
    """

    def __init__(
        self,
        model,
        model_type,
        context_length,
        vocab_chunk,
        layers_per_dispatch=1,
        unified=None,
        lm_head=0,
        golden=1,
        dual_channel=1,
        weight_group=None,
        stack_size=None,
        proj_rc_cache=None,
    ):
        if not model:
            raise DecodeConfigError(
                "model is required. mlir-air's builder defaults to "
                "'llama-3.2-1b', and inheriting that silently builds the wrong "
                "model -- name it."
            )
        if not model_type:
            raise DecodeConfigError(
                "model_type is required: it is the -DMODEL_TYPE the AIE kernels "
                "were compiled with, and it is what lets `fused_decode` refuse "
                "another model's objects. rms_residual.o and rope.o differ "
                "between models while keeping the same filenames, so linking "
                "the wrong ones is not an error -- it is wrong output."
            )
        if int(context_length) < 1:
            raise DecodeConfigError(
                f"context_length must be >= 1, got {context_length}"
            )
        if int(vocab_chunk) < 1:
            raise DecodeConfigError(f"vocab_chunk must be >= 1, got {vocab_chunk}")
        self.model = str(model)
        self.model_type = str(model_type)
        self.context_length = int(context_length)
        self.vocab_chunk = int(vocab_chunk)
        self.layers_per_dispatch = int(layers_per_dispatch)
        self.unified = unified
        self.lm_head = lm_head
        self.golden = golden
        self.dual_channel = dual_channel
        self.weight_group = weight_group
        self.stack_size = stack_size
        self.proj_rc_cache = proj_rc_cache

    #: The context length the built artifact actually serves, which is rounded
    #: up to a multiple of 16. Two requests that round to the same value produce
    #: the same artifact -- worth knowing before building both.
    @property
    def attn_maxl(self):
        return 16 * ((self.context_length + 15) // 16)

    def fingerprint(self):
        """Short digest of every field that changes the artifact.

        Used to name the build. `model` and `context_length` are in the name
        already; this covers the rest, because two builds differing only in,
        say, `weight_group` are different artifacts that would otherwise land on
        the same path.
        """
        parts = "|".join(f"{k}={v}" for k, v in sorted(self.env().items()))
        return hashlib.sha256(parts.encode()).hexdigest()[:8]

    def env(self):
        """This configuration as the environment mlir-air's builder reads."""
        out = {}
        for var, attr in _ENV.items():
            value = getattr(self, attr)
            if value is not None:
                out[var] = str(value)
        return out

    def __repr__(self):
        return (
            f"DecodeConfig(model={self.model!r}, model_type={self.model_type!r}, "
            f"context_length="
            f"{self.context_length}, attn_maxl={self.attn_maxl}, "
            f"vocab_chunk={self.vocab_chunk})"
        )


#: One build at a time per process. `_environment` mutates `os.environ` and the
#: builder reads it while it executes, so two concurrent calls would interleave
#: their models and context lengths -- one resolving the other's configuration,
#: and the restores clobbering each other. The window is the env swap plus the
#: import plus the compile, so the lock covers all three rather than just the
#: swap.
_BUILD_LOCK = threading.Lock()


@contextlib.contextmanager
def _environment(values):
    """Apply `values` for the duration of one build, then restore.

    Scoped rather than assigned, because the environment is process-global and
    a build that leaked its model name into it would change what the *next*
    build produced -- which is the failure mode this op removes.
    """
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


#: Left beside the objects by whoever compiled them, naming the `-DMODEL_TYPE`
#: they were compiled with. `examples/llm_q4nx/decode_kernels.py` writes one per
#: model; mlir-air's own Makefile writes none.
_KERNEL_STAMP = ".decode_kernels.{model_type}.json"
_KERNEL_STAMP_GLOB = ".decode_kernels.*.json"

#: What `lower` accepts. `None` is not among them: the op always names a format,
#: because the two it can produce go to different runtimes.
_OUTPUT_FORMATS = frozenset({"xclbin", "elf", "pdi"})


def _verify_kernels(kernel_objects, model_type):
    """Refuse another model's AIE objects.

    `rms_residual.o` and `rope.o` are compiled per model and keep the same
    filenames, so handing this op one model's objects with another's config
    links, builds, and decodes to garbage -- the silent failure the rest of this
    file exists to remove, and one it would otherwise have kept.

    The objects do not say what they were built for, so the check is on the
    stamp left beside them. A stamp for a *different* model is refused. No stamp
    at all is allowed, because mlir-air's own Makefile writes none and building
    against those is legitimate -- the cost is that the check cannot run, which
    is why the stamped path is the one the examples use.
    """
    import glob

    missing = [o for o in kernel_objects if not os.path.exists(o)]
    if missing:
        raise DecodeConfigError(
            "these AIE objects do not exist:\n  " + "\n  ".join(missing)
        )

    for directory in sorted(
        {os.path.dirname(os.path.abspath(o)) for o in kernel_objects}
    ):
        stamps = glob.glob(os.path.join(directory, _KERNEL_STAMP_GLOB))
        if not stamps:
            continue
        want = os.path.join(directory, _KERNEL_STAMP.format(model_type=model_type))
        if want not in stamps:
            found = ", ".join(os.path.basename(s).split(".")[2] for s in sorted(stamps))
            raise DecodeConfigError(
                f"the AIE objects in {directory} were compiled for {found}, not "
                f"{model_type}. rms_residual.o and rope.o differ between models "
                f"and keep the same names, so linking these would build cleanly "
                f"and decode to garbage. Build this model's kernels first."
            )


def _load_builder(fused_decode_dir):
    """mlir-air's `fused_decode` module, freshly executed.

    A fresh module object each time, deliberately: the builder resolves its
    geometry into module-level constants while it executes, so re-executing is
    what makes a second configuration in the same process read its own values
    rather than the first one's. Verified across model, layer count, vocab
    chunking and context length.
    """
    import importlib.util
    import sys

    path = os.path.join(fused_decode_dir, "fused_decode.py")
    if not os.path.exists(path):
        raise DecodeConfigError(
            f"{path} does not exist. It comes from mlir-air's sources; fetch "
            f"them at the pinned commit with utils/fetch_mlir_air_src.py."
        )
    if fused_decode_dir not in sys.path:
        sys.path.insert(0, fused_decode_dir)
    name = f"_air_fused_decode_{len(sys.modules)}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fused_decode(
    config,
    fused_decode_dir,
    kernel_objects=(),
    output_format="xclbin",
    name=None,
):
    """Build one fused decode artifact for `config`.

    Returns the dict `FusedDecodeOp.lower` returns -- the binary and its
    instruction stream.

    Args:
        config: a `DecodeConfig`.
        fused_decode_dir: mlir-air's `programming_examples/fused_decode`.
        kernel_objects: the AIE objects to link, built against this model's
            `-DMODEL_TYPE`. They are model-specific: `rms_residual.o` and
            `rope.o` differ between models, and linking another model's is not
            an error, it is wrong output.
        output_format: ``"xclbin"``, ``"elf"`` or ``"pdi"``.
        name: artifact name. Defaults to the model, the context length and a
            digest of every other field that changes the artifact -- two builds
            differing only in `vocab_chunk` or `weight_group` would otherwise
            share a project directory, and the second would overwrite the first
            while the first call's returned paths still pointed at it.
    """
    from triton.backends.amd_triton_npu.fused_decode_op import FusedDecodeOp

    if not isinstance(config, DecodeConfig):
        raise DecodeConfigError(
            f"config must be a DecodeConfig, got {type(config).__name__}. The "
            f"configuration is the op's argument, not the process environment."
        )

    # `_aircc_compile` treats anything that is not "elf" or "pdi" as "xclbin",
    # so a typo silently produces the wrong kind of artifact rather than
    # failing. Checked here instead.
    if output_format not in _OUTPUT_FORMATS:
        raise DecodeConfigError(
            f"output_format={output_format!r} is not one of "
            f"{sorted(_OUTPUT_FORMATS)}. Anything unrecognised would lower as "
            f"xclbin without complaint."
        )

    _verify_kernels(list(kernel_objects), config.model_type)

    with _BUILD_LOCK, _environment(config.env()):
        builder = _load_builder(fused_decode_dir)
        if builder.MODEL_NAME != config.model:
            raise DecodeConfigError(
                f"asked for {config.model!r} but mlir-air's builder resolved "
                f"{builder.MODEL_NAME!r}. It is one of "
                f"{sorted(getattr(builder, '_MODELS', {}))}."
            )
        op = FusedDecodeOp.from_builder(
            builder.build_module,
            kernel_objects=list(kernel_objects),
            name=name
            or (
                f"fused_decode_{config.model}_L{config.context_length}"
                f"_{config.fingerprint()}"
            ),
            stack_size=builder.STACK_SIZE,
        )
        artifacts = op.lower(output_format=output_format)

    artifacts["config"] = config
    artifacts["attn_maxl"] = builder.ATTN_MAXL
    return artifacts
