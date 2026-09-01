# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The NPU add chain both examples in this directory dispatch.

``shared_buffer_test.py`` and ``zero_copy_benchmark.py`` want the same thing --
a compiled NPU chain over N f32 elements whose first and last operands can be
bound to shared buffers -- so it is built once here rather than twice.

The combined-arg layout is fixed by the chain and is what callers index with::

    0 IN_A   caller-supplied, per dispatch
    1 IN_B   caller-supplied
    2 TMP    device intermediate, never leaves the NPU
    3 ADDEND static
    4 OUT    result

which computes ``OUT = (IN_A + IN_B) + ADDEND``.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from triton.backends.amd_triton_npu.multilaunch import NPUChain

BLOCK_SIZE = 1024

#: How the add is tiled. Named here rather than built inside build() because
#: the HSA half of zero_copy_benchmark.py dispatches this same kernel through
#: the ordinary Triton path, and the two runtimes are only comparable if they
#: lower it the same way. aie2p only, like the rest of the gpt2 scripts.
TRANSFORM_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gpt2",
    "transform_add_f32_aie2p.mlir",
)

#: Combined-arg positions; see the module docstring.
I_A, I_B, I_TMP, I_ADDEND, I_OUT = range(5)

#: Fixed by the chain's wiring, so both callers pass these unchanged.
INTERMEDIATE = {I_TMP}
OUTPUT = {I_OUT}

# Which of I_A / I_B / I_ADDEND are static is the caller's business, not the
# chain's: the benchmark holds two of them fixed across iterations, the test
# rewrites both inputs every dispatch. So `static_indices` stays a run() arg.


@triton.jit
def add_kernel(A, B, C, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """C = A + B, one block per program.

    No masking: the caller sizes the grid to cover n_elements exactly, so the
    element count must be a multiple of BLOCK_SIZE.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(A + offsets[:])
    b = tl.load(B + offsets[:])
    tl.store(C + offsets[:], a + b)


def build(name: str, n: int) -> NPUChain:
    """Compile the two-op add chain for ``n`` f32 elements.

    Two chained adds rather than one, deliberately. A chain holding a *single*
    op returns correct data on its first dispatch and corrupt data on every
    dispatch after that -- a pre-existing bug in the multi-launch path,
    reproducible on an unmodified checkout with plain host staging and no
    shared buffers involved. Two or more ops are unaffected, which is why the
    fused MLP in examples/gpt2 (four ops) never hit it. Do not "simplify" this
    back to one op: both callers dispatch repeatedly, so they would report
    wrong results for a reason that has nothing to do with what they measure.
    """
    zeros = torch.zeros(n, dtype=torch.float32)
    chain = NPUChain(name)
    for arg_map in ({0: I_A, 1: I_B, 2: I_TMP}, {0: I_TMP, 1: I_ADDEND, 2: I_OUT}):
        chain.add(
            add_kernel,
            grid=(n // BLOCK_SIZE,),
            arg_map=arg_map,
            args=(zeros, zeros, zeros, n),
            constexprs={"BLOCK_SIZE": BLOCK_SIZE},
            transform_script=TRANSFORM_SCRIPT,
        )
    return chain
