# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Locating mlir-air's LLM sources, and choosing who dispatches the decode.

Model-agnostic: every example under `examples/*_q4nx/` needs mlir-air's
`programming_examples/` on `sys.path` (the wheel ships no `programming_examples`)
and needs mlir-air's decoder kept off its own dispatch path. Neither depends on
which model is being run, so both live here rather than in each model's config.
"""

import os
import sys
from pathlib import Path

#: Where mlir-air's sources may be, in priority order. The first is a
#: developer's own working clone; the second is the sparse checkout
#: `utils/fetch_mlir_air_src.py` makes at the pinned commit. Checking the
#: working clone first means someone editing mlir-air sees their edits.
_AIR_CHECKOUTS = ("mlir-air-local", "third_party/mlir-air-src")


def air_llms_root():
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


def fused_decode_dir(engine="fused_decode"):
    """Where the decode's builder, kernels and built artifacts live.

    `engine` is a key of `tl.extra.npu.ENGINES` -- there are two builders, the
    shared one and the per-layer-embedding fork, and they are separate
    directories under `programming_examples`. Resolved through that table
    rather than by name here so the directory and the module that has to be in
    it cannot drift apart.

    The kernel *sources* stay in the shared engine's directory either way: the
    fork ships only its own `ple.cc` and compiles the other six out of its
    parent's `kernels/`, which is `decode_kernels`' business, not this one's.
    """
    from triton.language.extra.npu import ENGINES

    if engine not in ENGINES:
        raise RuntimeError(
            f"unknown decode engine {engine!r}; known: {sorted(ENGINES)}"
        )
    return str(air_llms_root().parent / ENGINES[engine].directory)


def add_air_paths(*packages):
    """Put mlir-air's llms packages on sys.path.

    `packages` are subdirectories of `llms/` this model needs -- its own q4nx
    package and whatever base package it borrows its config and RoPE table
    from. `llms/` itself and `programming_examples/` (for `shared.*`) are always
    added.
    """
    llms = air_llms_root()
    for p in (
        str(llms),
        *(str(llms / pkg) for pkg in packages),
        str(llms.parent),  # programming_examples, for `shared.*`
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


def decode_attn_maxl(engine="fused_decode", want=None, artifact_dir=None):
    """The calibrated ATTN_MAXL a decode of reach `want` will resolve to.

    For a prefill that wants to write mlir-air's device KV layout directly: the
    slab's geometry is a function of ATTN_MAXL, so it has to be known before
    the prefill runs -- but the decoder that fixes it is built afterwards, and
    upstream picks the *smallest* window covering `P + n_tokens`, which the
    prefill cannot know. A mismatch would not raise; it would place every row
    at the wrong offset and decode to fluent nonsense.

    So the two sides agree on a number that does not depend on the generation
    length: `want` is the session's declared context bound (`--max-seq`), and
    the decoder is then pinned to the window this returns rather than to the
    one it would have chosen. A window LARGER than the prompt needs is always
    correct -- a template built at ATTN_MAXL serves every L in [1, ATTN_MAXL] --
    so the only cost of overshooting is speed, which is why `want` is honoured
    rather than always taking the largest.

    Falls back to the largest calibrated window when `want` exceeds every one
    of them: that build simply cannot serve a context that long, and saying so
    belongs at the prompt that asks for it, not here.

    Resolved through mlir-air's own `DecodeInstsGen` rather than by globbing
    for `decode_L*.xclbin`, so "calibrated" means what it means there (a pair
    of same-ATTN_MAXL builds whose instruction streams differ by a constant
    slope), not what a filename suggests.

    Returns None when no calibrated template exists -- an ordinary state, not
    an error: `--decode gpu` and `--prefill-only` never build one.
    """
    import sys as _sys

    d = artifact_dir or fused_decode_dir(engine)
    try:
        if d not in _sys.path:
            _sys.path.insert(0, str(air_llms_root().parent / "fused_decode"))
        from decode_insts_gen import DecodeInstsGen

        gen = DecodeInstsGen(str(d), None)
        if want is not None:
            try:
                return int(gen.select(int(want)))
            except KeyError:
                pass  # nothing covers `want`; the largest is the best on offer
        return int(gen.attn_maxl)
    except Exception:  # noqa: BLE001 -- see the docstring
        return None


class DecodeArtifactError(RuntimeError):
    """Raised when the decode shape asked for is not one this example drives."""


def select_decode_artifact(env=None):
    """Keep mlir-air's decoder off *its own* full-ELF dispatch.

    ``DECODE_ELF`` (``fused_decode/decode_elf.py``) does not select a decode
    shape so much as select *who dispatches it*: set, ``FusedDecoder`` loads a
    full ELF and runs it through pyxrt itself. These examples never want that.
    They build the xclbin templates (``decode_build.py``) and run the decode
    from those, and on HSA they drive the dispatch themselves.

    It would not get that far in any case: mlir-air's ELF path brings up a
    second LLVM and re-registers an option Triton has already registered
    ("Option 'print-inst-addrs' registered more than once!"), which aborts the
    process rather than raising. That is a bug to fix, not a shape rejected on
    taste.

    So the variable is written here rather than left to its default -- it is
    the only channel ``FusedDecoder`` offers, its ``__init__`` taking no such
    argument -- and an explicit request for it is refused with the reason,
    which beats aborting later inside mlir-air.

    Returns the value written, so a caller (and a test) can check it.
    """
    env = os.environ if env is None else env
    asked = env.get("DECODE_ELF")
    if asked is not None and asked != "0":
        raise DecodeArtifactError(
            f"DECODE_ELF={asked!r} hands the decode to mlir-air's own full-ELF "
            "dispatch, which these examples never use: they run the xclbin "
            "templates built by decode_build.py, and on HSA they dispatch the "
            "decode themselves. mlir-air's route also aborts in-process on a "
            "duplicate LLVM option registration. Unset DECODE_ELF."
        )
    env["DECODE_ELF"] = "0"
    return env["DECODE_ELF"]
