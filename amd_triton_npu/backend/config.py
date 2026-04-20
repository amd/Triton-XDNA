# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Process-global configuration for the Triton-XDNA NPU backend.

Provides a Python API to control settings that were previously only
available via environment variables. Programmatic values always take
priority over environment variables, which in turn take priority over
built-in defaults.

Usage::

    from triton.backends.amd_triton_npu.config import npu_config, set_config

    # Direct attribute assignment
    npu_config.bf16_emulation = True
    npu_config.compile_only = True

    # Or dict-style
    set_config(bf16_emulation=True, compile_only=False)

    # Temporary overrides via context manager
    from triton.backends.amd_triton_npu.config import config_context
    with config_context(compile_only=True):
        kernel[grid](a, b, c)

    # Environment variables still work as fallback
    # AMD_TRITON_NPU_BF16_EMULATION=1 python my_script.py
"""

import contextlib
import logging
import os
from pathlib import Path

_UNSET = object()


class _NPUConfig:
    """Process-global configuration for the NPU backend.

    Each property falls back to its corresponding environment variable
    when no programmatic value has been set. Call ``reset()`` to clear
    all programmatic overrides.
    """

    def __init__(self):
        self._compile_only = _UNSET
        self._transform_tiling_script = _UNSET
        self._bf16_emulation = _UNSET
        self._output_format = _UNSET
        self._air_project_path = _UNSET
        self._debug = _UNSET

    # ---- compile_only ----

    @property
    def compile_only(self) -> bool:
        """If True, compile kernels but skip XRT launch.

        Env var fallback: ``AMD_TRITON_NPU_COMPILE_ONLY`` (``"1"`` to enable).
        """
        if self._compile_only is not _UNSET:
            return self._compile_only
        return os.getenv("AMD_TRITON_NPU_COMPILE_ONLY", "0") == "1"

    @compile_only.setter
    def compile_only(self, value: bool):
        self._compile_only = bool(value)

    # ---- transform_tiling_script ----

    @property
    def transform_tiling_script(self):
        """Path to a custom MLIR Transform dialect tiling script.

        When set, the backend loads this file instead of the built-in
        default tiling strategy. Set to ``None`` to use the default.

        Env var fallback: ``AIR_TRANSFORM_TILING_SCRIPT``.
        """
        if self._transform_tiling_script is not _UNSET:
            return self._transform_tiling_script
        return os.getenv("AIR_TRANSFORM_TILING_SCRIPT")

    @transform_tiling_script.setter
    def transform_tiling_script(self, value):
        self._transform_tiling_script = value

    # ---- bf16_emulation ----

    @property
    def bf16_emulation(self) -> bool:
        """If True, pass ``--bf16-emulation`` to aircc.

        This enables hardware truncation of f32 to bf16 before multiply,
        with f32 accumulation.

        Env var fallback: ``AMD_TRITON_NPU_BF16_EMULATION`` (``"1"`` to enable).
        """
        if self._bf16_emulation is not _UNSET:
            return self._bf16_emulation
        return os.getenv("AMD_TRITON_NPU_BF16_EMULATION", "0") == "1"

    @bf16_emulation.setter
    def bf16_emulation(self, value: bool):
        self._bf16_emulation = bool(value)

    # ---- output_format ----

    @property
    def output_format(self):
        """Force the output format to ``"elf"`` or ``"xclbin"``.

        Set to ``None`` for auto-detection (ELF on npu2, xclbin on npu1).
        ELF format is only supported on npu2 (AIE2P) devices.

        Env var fallback: ``AMD_TRITON_NPU_OUTPUT_FORMAT``.
        """
        if self._output_format is not _UNSET:
            return self._output_format
        v = os.getenv("AMD_TRITON_NPU_OUTPUT_FORMAT", "").lower()
        return v if v in ("elf", "xclbin") else None

    @output_format.setter
    def output_format(self, value):
        if value is not None and value not in ("elf", "xclbin"):
            raise ValueError(
                f"output_format must be 'elf', 'xclbin', or None; got {value!r}"
            )
        self._output_format = value

    # ---- air_project_path ----

    @property
    def air_project_path(self) -> Path:
        """Directory where intermediate IR files and compiled artifacts are written.

        Defaults to ``./air_project/`` relative to the current working directory.

        Env var fallback: ``AMD_TRITON_NPU_AIR_PROJECT_PATH``.
        """
        if self._air_project_path is not _UNSET:
            return Path(self._air_project_path)
        custom = os.getenv("AMD_TRITON_NPU_AIR_PROJECT_PATH")
        if custom:
            return Path(custom)
        return Path(os.getcwd()) / "air_project"

    @air_project_path.setter
    def air_project_path(self, value):
        if value is None:
            self._air_project_path = _UNSET
            return
        self._air_project_path = value

    # ---- debug ----

    @property
    def debug(self) -> bool:
        """If True, enable verbose logging from subprocesses and the C++ launcher.

        Env var fallback: ``AMD_TRITON_NPU_DEBUG`` (``"1"`` to enable).
        """
        if self._debug is not _UNSET:
            return self._debug
        return os.getenv("AMD_TRITON_NPU_DEBUG", "0") == "1"

    @debug.setter
    def debug(self, value: bool):
        self._debug = bool(value)
        # Keep the driver logger level in sync so logger.debug() calls
        # are enabled/suppressed when the flag is toggled programmatically.
        _drv = logging.getLogger("triton.backends.amd_triton_npu.driver")
        _drv.setLevel(logging.DEBUG if self._debug else logging.CRITICAL)

    # ---- utilities ----

    def reset(self):
        """Clear all programmatic overrides, reverting to env var / default values."""
        self._compile_only = _UNSET
        self._transform_tiling_script = _UNSET
        self._bf16_emulation = _UNSET
        self._output_format = _UNSET
        self._air_project_path = _UNSET
        self._debug = _UNSET


# Module-level singleton
npu_config = _NPUConfig()


def set_config(**kwargs):
    """Set multiple configuration options at once.

    Example::

        set_config(compile_only=True, bf16_emulation=True)

    Raises ``ValueError`` for unknown keys.
    """
    valid_keys = {
        "compile_only",
        "debug",
        "transform_tiling_script",
        "bf16_emulation",
        "output_format",
        "air_project_path",
    }
    for key, value in kwargs.items():
        if key not in valid_keys:
            raise ValueError(
                f"Unknown config key: {key!r}. Valid keys: {sorted(valid_keys)}"
            )
        setattr(npu_config, key, value)


@contextlib.contextmanager
def config_context(**kwargs):
    """Temporarily override configuration settings, restoring on exit.

    Example::

        with config_context(compile_only=True):
            kernel[grid](a, b, c)
    """
    saved = npu_config.__dict__.copy()
    try:
        set_config(**kwargs)
        yield npu_config
    finally:
        npu_config.__dict__.clear()
        npu_config.__dict__.update(saved)
