# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared code-generation helpers for the NPU launchers.

These primitives are used by both the XRT launchers (``driver.py``) and the
HSA launcher (``hsa_launcher.py``). They live in their own module so the two
drivers depend on a common surface instead of reaching into each other's
internals.
"""


def ty_to_cpp(ty: str) -> str:
    """Map a Triton signature type to the C++ type used in the launcher."""
    if ty[0] == "*":
        return "void*"
    if ty == "constexpr":
        return "PyObject*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "bfloat16",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def extracted_type(ty: str) -> str:
    """C++ type used to receive an argument from ``PyArg_ParseTuple``."""
    if ty[0] == "*" or ty == "constexpr":
        return "PyObject*"
    return ty_to_cpp(ty)


def format_of(ty: str) -> str:
    """``PyArg_ParseTuple`` format character for a C++ type from ``extracted_type``."""
    return {
        "PyObject*": "O",
        "constexpr": "O",
        "float": "f",
        "double": "d",
        "long": "l",
        "int8_t": "b",
        "int16_t": "h",
        "int32_t": "i",
        "int64_t": "l",
        "uint8_t": "B",
        "uint16_t": "H",
        "uint32_t": "I",
        "uint64_t": "K",
    }[ty]


def extract_signature_and_constants(src) -> tuple[dict, dict]:
    """Return ``(constants, signature)`` keyed by positional arg index."""
    constants = src.constants if hasattr(src, "constants") else dict()
    cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
    constants = {cst_key(key): value for key, value in constants.items()}
    signature = {cst_key(key): value for key, value in src.signature.items()}
    return constants, signature


def extract_actual_sizes(src) -> "str | None":
    """Extract actual (non-padded) problem sizes from constexpr args.

    When the kernel has constexpr args named "M" and "N", their values are the
    actual problem dimensions. These are passed to air-wrap-func-with-parallel
    as actual-sizes to enable DMA padding via air-split-launch-for-padding on
    boundary tiles. Only set actual-sizes when dimensions are NOT tile-aligned
    (M % BLOCK_SIZE_M != 0 or N % BLOCK_SIZE_N != 0), to avoid triggering the
    padding split path when it's not needed.

    Returns a "M,N,1" string, or None.
    """
    if not (hasattr(src, "fn") and hasattr(src.fn, "arg_names")):
        return None
    arg_names = src.fn.arg_names
    raw_constants = src.constants if hasattr(src, "constants") else {}

    def _get_constexpr(name):
        """Look up a constexpr value by arg name, trying multiple key forms."""
        if name not in arg_names:
            return None
        idx = arg_names.index(name)
        # src.constants uses tuple keys (idx,) per ASTSource.__init__,
        # but check multiple forms for robustness across versions.
        for key in [(idx,), idx, name]:
            if key in raw_constants:
                return raw_constants[key]
        return None

    m_val = _get_constexpr("M")
    n_val = _get_constexpr("N")
    if m_val is not None and n_val is not None:
        bsm = _get_constexpr("BLOCK_SIZE_M")
        bsn = _get_constexpr("BLOCK_SIZE_N")
        needs_padding = True
        if bsm is not None and bsn is not None:
            needs_padding = (m_val % bsm != 0) or (n_val % bsn != 0)
        if needs_padding:
            return f"{m_val},{n_val},1"
    return None
