# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Llama-3.2-1B dims, the q4nx weight loader, and the RoPE table.

The dims and the loader are mlir-air's -- there is no reason to re-derive
either, and a second copy of the RoPE frequency scaling is exactly the drift
worth avoiding. This module only locates mlir-air's LLM
packages on sys.path and re-exports what the prefill needs.
"""

import os
import sys
from pathlib import Path

# Llama-3.2-1B. Mirrors mlir-air's llama32_1b_q4nx_weights; kept here so the
# kernels can be read without chasing an import.
D = 2048  # model dim
DQ = 2048  # q proj out  (32 heads * 64)
DK = 512  # k proj out  (8 kv heads * 64)
DV = 512
DH = 64  # head dim
N_Q_HEADS = 32
N_KV_HEADS = 8
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 4
INTER = 8192  # mlp intermediate
N_LAYERS = 16
VOCAB = 128256

# The deployed fused decode's epsilon (fused_decode/models/all_models.h:54).
# HF's config says 1e-5; matching the decode matters more than matching HF,
# and at D=2048 the two differ far below bf16 resolution.
RMS_EPS = 1e-6

# The Paris gate, from llama32_1b_q4nx_prefill.py.
PROMPT = [128000, 791, 6864, 315, 9822, 374]  # "The capital of France is"
EXPECT_FIRST = 12366  # " Paris"

MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Llama-3.2-1B-NPU2")


#: Where mlir-air's sources may be, in priority order. The first is a
#: developer's own working clone; the second is the sparse checkout
#: `utils/fetch_mlir_air_src.py` makes at the pinned commit. Checking the
#: working clone first means someone editing mlir-air sees their edits.
_AIR_CHECKOUTS = ("mlir-air-local", "third_party/mlir-air-src")


def _air_llms_root():
    """The mlir-air programming_examples/llms directory.

    AIR_LLMS_ROOT overrides everything. Otherwise walk up looking for either
    checkout; if neither exists, say how to get one rather than failing later
    on an import of `fused_decode`.
    """
    env = os.environ.get("AIR_LLMS_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        for rel in _AIR_CHECKOUTS:
            cand = parent / rel / "programming_examples" / "llms"
            if cand.is_dir():
                return cand
    raise RuntimeError(
        "cannot find mlir-air's sources (programming_examples/llms).\n"
        "  Fetch them at the pinned commit:\n"
        "      python3 utils/fetch_mlir_air_src.py\n"
        "  or point AIR_LLMS_ROOT at your own checkout."
    )


class DecodeArtifactError(RuntimeError):
    """Raised when the decode shape asked for is not one this example drives."""


def select_decode_artifact(env=None):
    """Keep mlir-air's decoder off *its own* full-ELF dispatch.

    ``DECODE_ELF`` (``fused_decode/decode_elf.py``) does not select a decode
    shape so much as select *who dispatches it*: set, ``FusedDecoder`` loads a
    full ELF and runs it through pyxrt itself. That is the one thing this
    example never wants, on either runtime:

    * on XRT, because ``decode_build.py`` builds the xclbin templates and the
      decode runs from those;
    * on HSA, because the full ELF **is** what we run -- but we load and
      dispatch it ourselves, through ``HsaElfProgram`` and a scratchpad (see
      ``hsa_decode.py``). Letting mlir-air do it too would do it twice.

    And in both cases it would not get that far: mlir-air's ELF path brings up
    a second LLVM and re-registers an option Triton has already registered
    ("Option 'print-inst-addrs' registered more than once!"), which aborts the
    process rather than raising. That is a bug to fix, not a shape rejected on
    taste.

    So the variable is written here rather than left to its default -- it is
    the only channel ``FusedDecoder`` offers, its ``__init__`` taking no such
    argument -- and an explicit request for it is refused with the reason,
    which beats aborting later inside mlir-air.

    Note this says nothing about whether *we* use a full ELF. On HSA we
    normally do; ``AMD_TRITON_NPU_HSA_DECODE`` selects that, not this.

    Returns the value written, so a caller (and a test) can check it.
    """
    env = os.environ if env is None else env
    asked = env.get("DECODE_ELF")
    if asked is not None and asked != "0":
        raise DecodeArtifactError(
            f"DECODE_ELF={asked!r} hands the decode to mlir-air's own full-ELF "
            "dispatch, which this example never uses: on XRT it runs the "
            "xclbin templates, and on HSA it loads and dispatches the ELF "
            "itself. mlir-air's route also aborts in-process on a duplicate "
            "LLVM option registration. Unset DECODE_ELF; to choose the HSA "
            "decode's shape use AMD_TRITON_NPU_HSA_DECODE=elf|insts."
        )
    env["DECODE_ELF"] = "0"
    return env["DECODE_ELF"]


def _add_air_paths():
    llms = _air_llms_root()
    for p in (
        str(llms),
        str(llms / "llama32_1b"),
        str(llms / "llama32_1b_q4nx"),
        str(llms.parent),  # programming_examples, for `shared.*`
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_q4nx(model=None):
    """Host-dequantized Llama-3.2-1B weights from a model.q4nx bundle.

    Dequantizing on the host is deliberate: prefill is compute-bound, so W4A16
    would cut memory traffic that is not the bottleneck and add unpack work to
    every GEMM tile.

    Returns a dict with
        layers: list of 16 dicts, each with
            attn_norm, ffn_norm : float32 [D]
            q, k, v, o, gate, up, down : bfloat16 [K, out]  (y = x @ W)
        embed, final_norm, lm_head : float32
    """
    _add_air_paths()
    from llama32_1b_q4nx_weights import Q4nxModel

    qm = Q4nxModel(model or MODEL_DEFAULT)
    layers = []
    for k in range(N_LAYERS):
        w = qm.layer_weights(k)
        attn_norm, ffn_norm = qm.layer_rms(k)
        layers.append(dict(attn_norm=attn_norm, ffn_norm=ffn_norm, **w))
    embed, final_norm, lm_head = qm.embed_norm_lmhead()
    return dict(
        layers=layers,
        embed=embed,
        final_norm=final_norm,
        lm_head=lm_head,
        fingerprint=qm.fingerprint(),
    )


def rope_lut(seq_len, dtype=None):
    """[seq_len, 64] = [cos_0..cos_31, sin_0..sin_31] per position.

    mlir-air's generate_rope_lut, unmodified. It carries the llama3 frequency
    scaling (factor 32, old_ctx 8192) that the fused decode also applies; a
    reimplementation that drops it produces K rotated on the wrong frequencies
    and long prompts silently degrade.
    """
    _add_air_paths()
    from llama32_1b_weights import LlamaConfig, generate_rope_lut

    if dtype is None:
        from ml_dtypes import bfloat16

        dtype = bfloat16
    return generate_rope_lut(LlamaConfig(n_layers=N_LAYERS), seq_len, dtype)
