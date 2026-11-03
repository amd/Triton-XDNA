#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E4B: an image turn, its vision tower, prefill and decode each on the
NPU or the iGPU. See `../llm_q4nx/gemma4_flash.py`.

    python gemma4_e4b_flash_inference.py --image photo.jpg --text "What is this?"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

sys.path.insert(0, config._SHARED)

import gemma4_flash  # noqa: E402

if __name__ == "__main__":
    sys.exit(gemma4_flash.run(config, __doc__))
