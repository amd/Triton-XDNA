# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""``tl.extra.npu.fused_decode`` -- one dispatch for a whole decoder stack.

The op every Q4NX example's decode goes through. What it replaces is a set of
**process-global environment variables**: `DECODE_MODEL`, `NLAYERS`,
`DECODE_GOLDEN_L` and the rest were read by mlir-air's builder at
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

#: Knobs every one of mlir-air's fused-decode builders reads from the
#: environment, and the attribute on `DecodeConfig` that supplies each. Anything
#: not listed is left alone, so a builder that grows a knob keeps its own
#: default until it is named.
_COMMON_ENV = {
    "DECODE_MODEL": "model",
    "DECODE_GOLDEN_L": "context_length",
    "NLAYERS": "layers_per_dispatch",
    "LM_HEAD": "lm_head",
    "DECODE_GOLDEN": "golden",
    "DECODE_UNI_DEC": "decode_waves",
    "PROJ_RC_CACHE": "proj_rc_cache",
}

#: The four knobs that moved. In the shared engine, mlir-air `deffe6f1` made
#: them properties of the model -- they live in its `_MODELS` entry, reached
#: through the builder's merged `MODEL` dict, and the environment is no longer
#: read for them. In the PLE fork, which predates that commit, they are still
#: environment variables.
#:
#: So they are named once here and placed differently per engine: `env` for a
#: builder that reads them, `model_table` for one that does not. Getting this
#: backwards is silent either way -- setting an ignored variable builds the
#: wrong geometry, checking an absent table key checks nothing -- which is why
#: it is a property of the engine rather than a guess made per call.
_MOVED_KNOBS = {
    "VOCAB_CHUNK_I2": "vocab_chunk",
    "W_DUAL_CHAN": "dual_channel",
    "DECODE_WGROUP": "weight_group",
    "DECODE_STACK": "stack_size",
}

#: Which slab each decode wave attends. Only the PLE fork has KV sharing, and
#: only it reads this; the shared engine never looks at it. Unset is the
#: IDENTITY map, so a model whose layers share a cache and does not name this
#: builds a wave that reads its own empty slab -- `DecodeConfig` refuses the
#: combination rather than letting it default.
_PLE_ONLY = {"DECODE_KV_SRC": "kv_src"}


class DecodeEngine:
    """One of mlir-air's fused-decode builders.

    There are two, and they are not interchangeable:

    * ``"fused_decode"`` -- the shared engine, which drives every model whose
      geometry its `_MODELS` table describes.
    * ``"ple"`` -- `fused_decode_ple`, a fork carrying the per-layer-embedding
      branch and the per-layer class map that Gemma4-E2B needs. Its `_MODELS`
      is a superset of the shared engine's, so it can build the other models
      too; upstream keeps that property deliberately, as the fork's only
      defence against drifting from its parent.

    What differs here is not just the filename. The fork predates `deffe6f1`,
    so the four knobs in `_MOVED_KNOBS` are environment variables to it and
    table entries to the shared engine -- see that constant.

    Attributes:
        name: the `engine` value on `DecodeConfig`.
        directory: the engine's directory under `programming_examples`.
        module: its builder module's filename.
        env: knobs this engine reads from the environment, as
            ``{variable: DecodeConfig attribute}``.
        model_table: knobs this engine takes from its `_MODELS` entry instead,
            same shape. Checked rather than set.
    """

    def __init__(self, name, directory, module, env, model_table):
        self.name = name
        self.directory = directory
        self.module = module
        self.env = dict(env)
        self.model_table = dict(model_table)


#: The engines `DecodeConfig(engine=...)` accepts.
ENGINES = {
    "fused_decode": DecodeEngine(
        name="fused_decode",
        directory="fused_decode",
        module="fused_decode.py",
        env=_COMMON_ENV,
        model_table=_MOVED_KNOBS,
    ),
    "ple": DecodeEngine(
        name="ple",
        directory="fused_decode_ple",
        module="fused_decode_ple.py",
        env={**_COMMON_ENV, **_MOVED_KNOBS, **_PLE_ONLY},
        model_table={},
    ),
}

#: What a caller gets if it does not choose. The shared engine: it is what
#: every model here but Gemma4-E2B uses, and what `fused_decode` meant before
#: there was a second one.
DEFAULT_ENGINE = "fused_decode"


def _check_model_table(builder, config):
    """Refuse to build when our recorded knobs differ from `_MODELS`.

    Only for engines that take those knobs from the table -- for one that still
    reads the environment, `model_table` is empty and this does nothing, because
    the values were applied rather than resolved.

    A mismatch means the model's entry upstream moved, or was transcribed
    wrongly here. Either way the build would quietly use the builder's value
    and the caller's record would be a lie -- which is the failure mode this
    whole module exists to avoid, one step removed.
    """
    table = getattr(builder, "MODEL", None)
    if not table:  # a builder without the table: nothing to check against
        return
    for var, attr in ENGINES[config.engine].model_table.items():
        want = getattr(config, attr, None)
        if want is None or var not in table:
            continue
        if int(want) != int(table[var]):
            raise DecodeConfigError(
                f"{config.model!r} records {var}={want}, but mlir-air's "
                f"_MODELS entry says {table[var]}. The {config.engine!r} "
                f"builder no longer reads this from the environment, so the "
                f"build would use {table[var]} and the recorded value would be "
                f"silently wrong. Reconcile the spec with the model's entry "
                f"upstream."
            )


class DecodeConfigError(ValueError):
    """The requested decode configuration cannot be built."""


def _normalize_kv_src(kv_src, engine, decode_waves):
    """Validate a KV sharing map and return the builder's spelling of it.

    The builder checks the same two things, but it checks them a minute into a
    lowering and after the environment has been mutated, so they are checked
    here too. `None` passes through: no map is the identity map.
    """
    if kv_src is None:
        return None
    if "DECODE_KV_SRC" not in ENGINES[engine].env:
        raise DecodeConfigError(
            f"engine={engine!r} does not read DECODE_KV_SRC, so a kv_src given "
            f"here would be dropped and the build would use the identity map. "
            f"A model that needs a sharing map needs an engine that reads one."
        )
    src = kv_src.split(",") if isinstance(kv_src, str) else list(kv_src)
    try:
        src = [int(s) for s in src]
    except (TypeError, ValueError):
        raise DecodeConfigError(f"kv_src must be ints, got {kv_src!r}") from None
    if not src:
        raise DecodeConfigError(
            "kv_src is empty, which the builder reads as the identity map. "
            "Pass None to mean that, so it cannot be an accident."
        )
    wrong = [(i, s) for i, s in enumerate(src) if not 0 <= s <= i]
    if wrong:
        raise DecodeConfigError(
            f"kv_src names one source slab per wave, and a wave can only read "
            f"a slab at or before its own. Out of range: "
            f"{', '.join(f'wave {i} -> {s}' for i, s in wrong)}"
        )
    if decode_waves is not None and len(src) != int(decode_waves):
        raise DecodeConfigError(
            f"kv_src has {len(src)} entries but the template has "
            f"{int(decode_waves)} decode waves. It names one source per wave."
        )
    return ",".join(str(s) for s in src)


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
        layers_per_dispatch, lm_head, golden: the remaining Makefile knobs the
            builder still reads from the environment, as strings or ints.
        engine: which of mlir-air's builders drives this model, a key of
            `ENGINES`. ``"fused_decode"`` unless the model needs the
            per-layer-embedding fork; see `DecodeEngine`.
        dual_channel: this model's ``W_DUAL_CHAN``. Selects the shim channel
            split and the DDR weight cascade order, so the artifact and the
            host that feeds it must agree. Set or checked depending on the
            engine; see `_MOVED_KNOBS`.
        decode_waves: ``DECODE_UNI_DEC`` -- decoder layers in the unified
            sequence. ``None`` takes the model's own `UNI_DEC`, which is what
            every model built through the shared engine wants. Named explicitly
            only where a Makefile does: the PLE engine defaults it to 1 for its
            single-layer gate, so a full-depth Gemma4 template has to ask.
        unified: accepted and unused. `UNIFIED` is set by every one of
            mlir-air's own Makefiles and read by none of its code, at this pin
            or any recent one -- so it is kept only so a spec transcribed from
            a Makefile does not have to drop a line, and is deliberately in no
            engine's `env` map.
        weight_group: layers per weight buffer, for models past the 4 GiB
            one-BO limit. ``0`` disables it.
        stack_size: AIE core stack. ``None`` takes the builder's own, which is
            what every model but the 8B wants.
        kv_src: ``DECODE_KV_SRC`` -- the slab each decode wave attends, as a
            sequence of ints or a comma-separated string, one per wave. Needed
            by a model whose later layers project no K/V and read an earlier
            layer's cache. ``None`` leaves the builder's identity map, which is
            correct only when no layer shares a cache: a wave pointed at its own
            slab with nothing to write there attends a zero key at the current
            position, which reads as a fluent opening that degrades into
            repetition. Accepted only by an engine that reads it; see
            `_PLE_ONLY`.
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
        engine=DEFAULT_ENGINE,
        decode_waves=None,
        kv_src=None,
    ):
        if engine not in ENGINES:
            raise DecodeConfigError(
                f"engine={engine!r} is not one of {sorted(ENGINES)}. It selects "
                f"which of mlir-air's builders lowers this model, and they do "
                f"not read the same knobs from the same places."
            )
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
        kv_src = _normalize_kv_src(kv_src, engine, decode_waves)
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
        self.engine = engine
        self.decode_waves = decode_waves
        #: Normalized to the builder's own comma-separated spelling, so `env()`
        #: needs no per-field formatting and `fingerprint()` separates two
        #: templates that differ only in their map.
        self.kv_src = kv_src

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

        The engine is hashed in alongside the environment rather than left to
        it: the two builders emit different designs for the same model, and for
        the shared engine `env()` does not even contain the knobs in
        `_MOVED_KNOBS`, so without this a pair of engines could fingerprint
        alike.
        """
        parts = "|".join(
            f"{k}={v}" for k, v in sorted({**self.env(), "ENGINE": self.engine}.items())
        )
        return hashlib.sha256(parts.encode()).hexdigest()[:8]

    def env(self):
        """This configuration as the environment mlir-air's builder reads.

        Which variables that is depends on the engine -- the PLE fork takes
        four knobs here that the shared engine takes from its model table.
        """
        out = {}
        for var, attr in ENGINES[self.engine].env.items():
            value = getattr(self, attr)
            if value is not None:
                out[var] = str(value)
        return out

    def __repr__(self):
        return (
            f"DecodeConfig(model={self.model!r}, model_type={self.model_type!r}, "
            f"engine={self.engine!r}, context_length="
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
def _environment(values, owned=()):
    """Apply `values` for the duration of one build, then restore.

    Scoped rather than assigned, because the environment is process-global and
    a build that leaked its model name into it would change what the *next*
    build produced -- which is the failure mode this op removes.

    `owned` is every variable the engine reads. Those NOT in `values` are
    **removed** for the duration rather than left alone, which is the other
    half of the same guarantee: on an engine that still reads the environment,
    omitting a knob is how a caller asks for the builder's own default, and
    leaving an inherited `W_DUAL_CHAN=0` in place would silently answer with
    something else. It would not be recorded either -- `fingerprint` hashes
    `env()`, which by construction does not contain what the caller never set.

    Only matters for the PLE engine today; the shared one resolves those four
    from `_MODELS` and would ignore them either way. Applied uniformly because
    "which engine reads what" is already the engine's business, not this
    function's.
    """
    keys = set(values) | set(owned)
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        if k in values:
            os.environ[k] = values[k]
        else:
            os.environ.pop(k, None)
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


def _load_builder(fused_decode_dir, engine=DEFAULT_ENGINE):
    """mlir-air's builder module for `engine`, freshly executed.

    A fresh module object each time, deliberately: the builder resolves its
    geometry into module-level constants while it executes, so re-executing is
    what makes a second configuration in the same process read its own values
    rather than the first one's. Verified across model, layer count, vocab
    chunking and context length.
    """
    import importlib.util
    import sys

    path = os.path.join(fused_decode_dir, ENGINES[engine].module)
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
        fused_decode_dir: the engine's directory under mlir-air's
            `programming_examples` -- `fused_decode`, or `fused_decode_ple`
            when `config.engine` says so. The two must agree: the directory
            says where to look and the engine says what to look for, and a
            mismatch is refused rather than searched around.
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

    with _BUILD_LOCK, _environment(config.env(), ENGINES[config.engine].env):
        builder = _load_builder(fused_decode_dir, config.engine)
        if builder.MODEL_NAME != config.model:
            raise DecodeConfigError(
                f"asked for {config.model!r} but mlir-air's builder resolved "
                f"{builder.MODEL_NAME!r}. It is one of "
                f"{sorted(getattr(builder, '_MODELS', {}))}."
            )
        _check_model_table(builder, config)
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
