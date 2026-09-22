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
            * `"kv_arrays"` -- Qwen3-4B and Gemma3-4B. `generate()` has no
              handoff parameter at all: it calls its own `_prefill_npu` and
              passes the `(K, V, first)` straight to the decode loop. So that
              one function is replaced with ours, which is the same trick the
              npz path uses on `run_prefill` and keeps us on their public
              `generate()`.
    """

    name: str
    decode_env: dict
    model_type: str
    air_package: str
    air_inference: str
    tokenizer_fallback: str
    driver_api: str = "npz"
    #: For `driver_api="prefiller"`: the decoder class that driver exposes.
    #: Named per model (`FusedDecode3B`, `FusedDecode8B`, and plain
    #: `FusedDecoder` for the kv_arrays ones), so it is recorded rather than
    #: derived from the model name.
    decoder_class: str = ""
    #: Whether `AMD_TRITON_NPU_RUNTIME=hsa` works for this model. The HSA
    #: adapter subclasses `air.FusedDecoder`, which only the npz-API drivers
    #: define -- the prefiller ones name theirs `FusedDecode3B` -- so asking
    #: for HSA elsewhere raises AttributeError from inside hsa_decode.py.
    #: Generalizing it belongs with the HSA scratchpad work, not with adding
    #: models.
    supports_hsa: bool = False
    #: Host memory the prefill needs to hold this model's dequantized bf16
    #: weights, in GiB. Checked before anything is allocated: an undersized
    #: host is SIGKILLed partway through the load, and a process cannot catch
    #: that to report the skip itself. 0 means unchecked.
    min_host_gib: float = 0.0
    #: llms packages to put on sys.path, beyond `air_package`.
    extra_packages: tuple = field(default_factory=tuple)
    #: Environment variable this model's mlir-air driver reads, at import time,
    #: to find the decode artifacts -- empty when it resolves `fused_decode/`
    #: itself. `decode_build.py` writes every model's artifacts there, while
    #: mlir-air's Makefiles leave each model's in its own directory, so a driver
    #: that defaults to the latter has to be pointed back.
    decode_dir_env: str = ""

    def decode_config(self, context_length):
        """This spec as a `tl.extra.npu.DecodeConfig`.

        The mapping lives here, once, rather than at each call site: every knob
        is transcribed from `decode_env`, and transcribing it by hand per caller
        is the same class of error the whole file exists to prevent -- a wrong
        value builds cleanly and decodes to garbage.

        `decode_env` keeps its string values because that is what mlir-air's
        Makefiles hold and what makes them diffable against upstream; the op
        takes them as they are. A knob absent from `decode_env` stays absent
        here too, leaving the builder its own default -- which is deliberate for
        `UNIFIED`, `DECODE_WGROUP`, `DECODE_STACK` and `PROJ_RC_CACHE`, each
        documented above where it is or is not set.
        """
        from triton.language.extra.npu import DecodeConfig

        e = self.decode_env
        return DecodeConfig(
            model=e["DECODE_MODEL"],
            model_type=self.model_type,
            context_length=context_length,
            vocab_chunk=e["VOCAB_CHUNK_I2"],
            layers_per_dispatch=e["NLAYERS"],
            unified=e.get("UNIFIED"),
            lm_head=e["LM_HEAD"],
            golden=e["DECODE_GOLDEN"],
            dual_channel=e["W_DUAL_CHAN"],
            weight_group=e.get("DECODE_WGROUP"),
            stack_size=e.get("DECODE_STACK"),
            proj_rc_cache=e.get("PROJ_RC_CACHE"),
        )


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
    supports_hsa=True,
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
    # Measured, not estimated: peak RSS of `--prefill-only` is 23.9 GiB, set
    # here with a margin. It was 30.7 before the padded weights stopped being
    # held alongside the unpadded ones they were built from.
    min_host_gib=26.0,
    extra_packages=("llama32_3b", "llama32_1b_q4nx"),
    driver_api="prefiller",
    decoder_class="FusedDecode8B",
)


#: Qwen3-4B. The first Q4NX model here that is not Llama-shaped, and the first
#: to need its own forward: between the QKV projection and RoPE each head's 128
#: lanes are RMS-normalized by a per-layer weight
#: (`llm_q4nx/qwen3_prefill.py`). Also the
#: first with DQ != D -- 32 heads of 128 against a 2560 model dim -- so o_proj
#: contracts 4096 -> 2560 instead of being square.
#:
#: Its environment comes from the same two places as the 3B's: `DECODE_ENV`
#: (llms/qwen3_4b_q4nx/Makefile) for most of it, and a bare `export` for
#: `W_DUAL_CHAN=1`. `VOCAB_CHUNK_I2=30` is this model's own value -- the
#: divisibility constraint depends on its NCX/NCY/PAIR_ROWS geometry, and
#: Qwen3-8B's 8 does not transfer despite the shared reader.
#:
#: `DECODE_WGROUP` is absent because its Makefile defaults it to 0 (disabled):
#: the split exists for models past the 4 GiB one-BO shim-BD-offset limit, and
#: 4B's ~2.1 GiB of decode weight is under it. The Makefile keeps it as an
#: override, and a model that needs it must set it here *and* match it on the
#: host, which is a second reason not to pin a value we do not need.
QWEN3_4B = ModelSpec(
    name="qwen3-4b",
    decode_env=dict(
        DECODE_MODEL="qwen3-4b",
        VOCAB_CHUNK_I2="30",
        UNIFIED="1",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        W_DUAL_CHAN="1",
    ),
    model_type="QWEN3_4B",
    air_package="qwen3_4b_q4nx",
    air_inference="qwen3_4b_q4nx_inference.py",
    # Qwen3's driver detokenizes straight from the weight repo rather than
    # exporting a tokenizer path, so there is no attribute for `tokenizer_dir`
    # to find. Qwen3 ships one ungated checkpoint, base and instruct alike.
    tokenizer_fallback="Qwen/Qwen3-4B",
    driver_api="kv_arrays",
    decoder_class="FusedDecoder",
    decode_dir_env="Q4NX_QWEN3_4B_DECODE_DIR",
    # Measured on this box, as the 8B's was: peak RSS of `--prefill-only` is
    # 12.4 GiB, down from 20.3. Most of that drop is `make_npu_resident` --
    # this model's weights pad from 6.8 GiB to 10.6, and both used to be
    # resident at once.
    min_host_gib=14.0,
    extra_packages=("qwen3_8b_q4nx", "qwen3_4b"),
    # Its driver *does* name its decoder `FusedDecoder`, so the adapter would
    # import -- but nothing here has run it on hardware, and the ELF route it
    # wants is still blocked. False records "untested", not "cannot".
    supports_hsa=False,
)


#: Gemma3-4B (text). The furthest from the Llama block of anything here: a
#: four-norm sandwich, dual-theta RoPE with a 1024-token sliding window on five
#: layers in six, GELU-tanh in the GLU, and Qwen3's per-head qk-norm and
#: decoupled q dim as well (`llm_q4nx/gemma3_prefill.py`).
#:
#: Its environment is the smallest of the four and still comes from two places:
#: `DECODE_ENV` in `llms/gemma3_4b_q4nx/Makefile`, plus `W_DUAL_CHAN=1` by bare
#: `export`. `VOCAB_CHUNK_I2=5` is its own -- the lowest here, and unsurprising
#: at a 262208 vocabulary.
#:
#: Neither `DECODE_STACK` nor `DECODE_WGROUP` appears, because its Makefile sets
#: neither; the builder defaults stand, and `decode_build` derives the stack
#: from the builder rather than repeating a number.
GEMMA3_4B = ModelSpec(
    name="gemma3-4b",
    decode_env=dict(
        DECODE_MODEL="gemma3-4b",
        VOCAB_CHUNK_I2="5",
        UNIFIED="1",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        W_DUAL_CHAN="1",
    ),
    model_type="GEMMA3_4B",
    air_package="gemma3_4b_q4nx",
    air_inference="gemma3_4b_q4nx_inference.py",
    # Like Qwen3's, its driver detokenizes from the weight repo and exports no
    # tokenizer path. Gemma is gated on the Hub, so the fallback is the bundle
    # repo rather than google/gemma-3-4b-it.
    tokenizer_fallback="FastFlowLM/Gemma3-4B-NPU2",
    driver_api="kv_arrays",
    decoder_class="FusedDecoder",
    decode_dir_env="Q4NX_GEMMA_DECODE_DIR",
    # Measured: peak RSS of `--prefill-only` is 15.1 GiB, down from 21.3. It
    # is now the highest of the three 4B-class models rather than the lowest,
    # because its 262208-row LM head is a separate tensor where Qwen3-4B's is
    # tied.
    min_host_gib=17.0,
    supports_hsa=False,
)


#: Qwen3-8B. Qwen3-4B's block at 4096 -- the same per-head qk-norm, the same
#: single-theta RoPE, the same 36 layers -- so it runs `qwen3_prefill.py`
#: unchanged and adds no forward of its own. Two differences that matter, and
#: neither is in the forward:
#:
#: * **The LM head is untied**, where Qwen3-4B's is tied. The bundle carries a
#:   separate Q4NX `lm_head.weight`; the 4B's bundle does not carry one at all.
#: * **K is 4096, not 2560**, which is what moves the two decode knobs below.
#:
#: Its environment comes from the same two places as every other model's:
#: `DECODE_ENV` in `llms/qwen3_8b_q4nx/Makefile`, plus `W_DUAL_CHAN=1` by bare
#: `export`. Three values are this model's own:
#:
#: * `VOCAB_CHUNK_I2=8` -- Qwen3-4B's 30 does not transfer despite the shared
#:   151936 vocabulary and the shared reader, because the divisibility
#:   constraint is on NCX/NCY/PAIR_ROWS rather than on the vocabulary.
#: * `DECODE_STACK=6144` -- a *third* distinct value (the 1B, 3B and Qwen3-4B
#:   take the builder's 10240 default, Llama-3.1-8B 8064). At K=4096 the seven
#:   K-wide L1 activation buffers leave under 8 KiB of the 64 KiB core memory,
#:   so the default overflows. As with the 8B, `decode_build` reads
#:   `fused_decode.STACK_SIZE` back from the builder rather than repeating it.
#: * `DECODE_WGROUP=9` -- 36 layers at K=4096 is 4.04 GiB, and a shim BD's byte
#:   offset is a uint32, so one BO only reaches 4 GiB. Four groups of nine keep
#:   each at ~1 GiB. The host must slice the weights the same way, which
#:   mlir-air's driver does from its own `DECODE_WGROUP = 9`.
QWEN3_8B = ModelSpec(
    name="qwen3-8b",
    decode_env=dict(
        DECODE_MODEL="qwen3-8b",
        VOCAB_CHUNK_I2="8",
        UNIFIED="1",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        DECODE_STACK="6144",
        DECODE_WGROUP="9",
        W_DUAL_CHAN="1",
    ),
    model_type="QWEN3_8B",
    air_package="qwen3_8b_q4nx",
    air_inference="qwen3_8b_q4nx_inference.py",
    # Qwen3 ships one ungated checkpoint, base and instruct alike, and this
    # driver detokenizes from the weight repo rather than exporting a tokenizer
    # path -- same as Qwen3-4B's.
    tokenizer_fallback="Qwen/Qwen3-8B",
    driver_api="kv_arrays",
    decoder_class="FusedDecoder",
    decode_dir_env="Q4NX_QWEN3_8B_DECODE_DIR",
    # Measured, not estimated: peak RSS of `--prefill-only` is 24.2 GiB, set
    # here with a margin. Within a rounding of Llama-3.1-8B's 23.9, which is
    # what a second 8B-parameter model dequantized to bf16 should cost.
    min_host_gib=26.0,
    # `qwen3_8b_q4nx_weights` puts `qwen3_4b` on sys.path itself (its
    # `LlamaConfig` lives there), but naming it here keeps the path set the
    # same whichever module gets imported first.
    extra_packages=("qwen3_4b",),
    supports_hsa=False,
)


#: Qwen2.5-7B. Llama-shaped in every respect the operator routing cares about
#: -- one norm pair, fused QKV, half-split RoPE, SwiGLU, GQA (7 q heads per kv
#: head), single theta, causal with no window -- and different in one place: q,
#: k and v each carry a **bias**, added to the raw projection output before
#: RoPE. Nothing else here has a bias on any projection, which is why it needs
#: its own forward (`llm_q4nx/qwen25_prefill.py`). It emphatically does *not*
#: have Qwen3's qk-norm.
#:
#: The first model here whose weights are **not a Q4NX bundle**. FastFlowLM
#: publishes no Qwen2.5-7B NPU2 bundle -- their Qwen2.5 line stops at 3B, and
#: that converter's output uses a nibble interleave the Llama/Qwen3 bundles do
#: not -- so mlir-air quantizes an ungated upstream HF checkpoint on load
#: instead, through the same quantizer the decode's cascade cache uses. Both
#: sides therefore see bit-identical weights, and `config.load_q4nx` goes
#: through mlir-air's `open_weight_source`, which picks the bundle reader only
#: when the source really is a `model.q4nx`.
#:
#: `DECODE_ENV` plus the usual bare `export` for `W_DUAL_CHAN=1`. Its own
#: values, all from its Makefile:
#:
#: * `VOCAB_CHUNK_I2=7`, on a 152064 vocabulary -- a fourth distinct value.
#: * `DECODE_STACK=6144`, the same as Qwen3-8B's and for the same reason at a
#:   different K: at K=3584 the seven K-wide L1 activation buffers leave too
#:   little of the 64 KiB core memory for the builder's 10240 default.
#: * `DECODE_WGROUP=7` -- 28 layers is 3.80 GiB and the lm-head adds 0.32, over
#:   the 4 GiB a shim BD's uint32 byte offset can address in one BO.
QWEN2_5_7B = ModelSpec(
    name="qwen2.5-7b",
    decode_env=dict(
        DECODE_MODEL="qwen2.5-7b",
        VOCAB_CHUNK_I2="7",
        UNIFIED="1",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        DECODE_STACK="6144",
        DECODE_WGROUP="7",
        W_DUAL_CHAN="1",
    ),
    model_type="QWEN2_5_7B",
    air_package="qwen25_7b_q4nx",
    air_inference="qwen25_7b_q4nx_inference.py",
    # The weight source *is* the tokenizer's checkpoint here, which is not true
    # of the bundle-backed models: mlir-air's default `Q4NX_MODEL_SOURCE` for
    # this one is the upstream repo itself. Ungated.
    tokenizer_fallback="Qwen/Qwen2.5-7B-Instruct",
    driver_api="kv_arrays",
    decoder_class="FusedDecoder",
    decode_dir_env="Q4NX_QWEN25_7B_DECODE_DIR",
    # Measured, not estimated: peak RSS of `--prefill-only` is 33.8 GiB, set
    # here with a margin. The largest of any model here, and not because it is
    # the largest model -- Llama-3.1-8B has more parameters and peaks at 23.9.
    # The difference is that this one quantizes on load: the mapped fp16
    # checkpoint and the Q4NX-rounded result are both resident while the
    # quantizer runs. Expect this to exceed a CI runner and skip.
    min_host_gib=36.0,
    # `qwen25_3b` holds the `LlamaConfig` this model's dims are read back
    # against; mlir-air's own weights module inserts it at import.
    extra_packages=("qwen25_3b",),
    supports_hsa=False,
)


#: Phi-4-mini. Llama-shaped but for the rotation: `partial_rotary_factor=0.75`
#: means RoPE covers 96 of each head's 128 lanes and the trailing 32 pass
#: through, and the frequencies come from a LongRoPE factor table shipped in
#: the bundle rather than from a closed form. That is a `_rope` override
#: (`llm_q4nx/phi4_prefill.py`); the block itself is Llama's. Tied LM head, as
#: on the 1B and 3B.
#:
#: **`model_type` is `PHI4_4B`, not `PHI4_MINI`.** The kernels are compiled
#: against that name and the decoder class defaults to it; deriving a
#: `-DMODEL_TYPE` from this spec's `name` would produce a header that does not
#: exist. Taken from its Makefile, like every other value here.
#:
#: The smallest `decode_env` of any model here, and the absences are facts
#: rather than omissions: no `DECODE_STACK`, no `DECODE_WGROUP` -- its
#: Makefile's `DECODE_FLAGS` carries only `W_DUAL_CHAN`, which again arrives by
#: a bare `export`. `VOCAB_CHUNK_I2=18` is emphatically not free: its Makefile
#: records that `(K/PAYLOAD)=6` must divide `VOCAB_I2*PAIR_ROWS`, leaving
#: {3,6,9,18} legal, and that the 1B's default is not even a divisor and would
#: deadlock this model's vocab wave.
PHI4_MINI = ModelSpec(
    name="phi4-mini",
    decode_env=dict(
        DECODE_MODEL="phi4-mini",
        VOCAB_CHUNK_I2="18",
        UNIFIED="1",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        W_DUAL_CHAN="1",
    ),
    model_type="PHI4_4B",
    air_package="phi4_mini_q4nx",
    air_inference="phi4_mini_q4nx_inference.py",
    # The Q4NX bundle carries no chat template, so its driver takes the
    # tokenizer from the HF checkpoint -- which is also the bf16 reference its
    # own verify gate compares against. Ungated.
    tokenizer_fallback="microsoft/Phi-4-mini-instruct",
    driver_api="prefiller",
    # Its own name for the class, as `FusedDecode3B`/`FusedDecode8B` are
    # theirs. It is imported into the inference module's namespace, which is
    # where the harness looks.
    decoder_class="FusedDecodePhi4",
    # Its driver resolves the templates from `DECODE_TEMPLATES`, defaulting to
    # its own directory; ours are written to the shared `fused_decode/` one.
    decode_dir_env="DECODE_TEMPLATES",
    # Measured, not estimated: peak RSS of `--prefill-only` is 12.1 GiB, set
    # here with a margin. Under the 4B-class models despite a 200064-row
    # embedding, because that embedding is also the LM head -- tied, so one
    # allocation serves both.
    min_host_gib=14.0,
    extra_packages=("llama32_1b_q4nx", "llama32_3b"),
    supports_hsa=False,
)


#: Qwen2.5-3B. Qwen2.5-7B's block at 2048, so it runs `qwen25_prefill.py`
#: unchanged -- the same q/k/v projection bias, no qk-norm. GQA 8 (16 q heads
#: over 2 kv heads), the widest ratio here.
#:
#: Its mlir-air directory is `llms/qwen25_3b_q4`, not `_q4nx`, and that names
#: the *weight* codec rather than the decode: the decode is the same Q4NX fused
#: engine every model here drives (`DECODE_MODEL=qwen2.5-3b`). Its default
#: source is a FastFlowLM `model.q4nx` bundle; `open_weight_source` falls back
#: to Q4_0-quantizing an HF checkpoint, which is what the name refers to.
#:
#: **`W_DUAL_CHAN=0`, alone among the models here**, and its Makefile is
#: emphatic about why: the dual feed wedges every decode dispatch on a Krackan
#: NPU -- 0 of 13 complete at every context from 1024 to 32768, with no prefill
#: and no weights involved -- while another model runs at 28 tok/s on the same
#: part minutes apart. Off completes 10 of 10 and costs ~16% decode throughput
#: on Strix. It is exported there rather than `?=`-assigned because the builder
#: reads it from the environment; a value that reaches only the stamp is
#: cosmetic.
#:
#: `VOCAB_CHUNK_I2=12` is likewise not free: it must pair with the model
#: entry's `UNI_LM=25`, and its Makefile records that 20 and 15 satisfy every
#: divisibility rule and still deadlock the vocab wave on device.
QWEN2_5_3B = ModelSpec(
    name="qwen2.5-3b",
    decode_env=dict(
        DECODE_MODEL="qwen2.5-3b",
        VOCAB_CHUNK_I2="12",
        UNIFIED="1",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        W_DUAL_CHAN="0",
    ),
    model_type="QWEN2_5_3B",
    air_package="qwen25_3b_q4",
    air_inference="qwen25_3b_q4_inference.py",
    tokenizer_fallback="Qwen/Qwen2.5-3B-Instruct",
    driver_api="kv_arrays",
    decoder_class="FusedDecoder",
    decode_dir_env="Q4NX_QWEN25_3B_DECODE_DIR",
    # Measured, not estimated: peak RSS of `--prefill-only` is 10.3 GiB, set
    # here with a margin. The smallest of the non-1B models, and comfortably
    # inside CI's runner -- unlike its 7B sibling, which declines there.
    min_host_gib=12.0,
    extra_packages=("qwen25_3b",),
    supports_hsa=False,
)


SPECS = {
    s.name: s
    for s in (
        LLAMA_3_2_1B,
        LLAMA_3_2_3B,
        LLAMA_3_1_8B,
        QWEN3_4B,
        QWEN3_8B,
        QWEN2_5_7B,
        QWEN2_5_3B,
        PHI4_MINI,
        GEMMA3_4B,
    )
}

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
