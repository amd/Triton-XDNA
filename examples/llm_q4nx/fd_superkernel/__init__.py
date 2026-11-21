# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The decode of the models air-FLM's shared `fused_decode` builder serves, as
one Triton superkernel dispatch a token. A model's module (`qwen3_4b`,
`gemma3_4b`) is its entry point: its `SuperkernelDecoder` selects the model
(`spec`) before the sizes (`geom`) and the kernel (`wave`) are read.
"""
