# Llama-3.2-1B Q4NX: a Triton prefill feeding mlir-air's fused decode

End-to-end Llama-3.2-1B on the AMD XDNA NPU. The prefill is Triton — this
repo's kernels, compiled through the `amd_triton_npu` backend. The decode is
mlir-air's Q4NX fused decode, run unmodified.

```
$ make chat
[session] ready: Triton prefill + AIR decode resident (ATTN_MAXL=2048, ...)

You: What is the capital of France?
Assistant: The capital of France is Paris.
Generated 6 tokens in 0.13s (47.24 tok/s)
```

```
$ python llama32_1b_q4nx_inference.py --backend npu --max-tokens 12 --greedy
[triton-prefill] backend=npu ops={'matmul', 'rms_norm', 'swiglu'} P=6 ... first=12366
[triton-prefill] first token 12366 (expect 12366) -- PASS
Generated 12 tokens in 0.22s (55.76 tok/s)
[e2e] '<|begin_of_text|>The capital of France is'
   ->  ' Paris. The capital of Germany is Berlin. The capital of Japan'
```

## Why this split

Prefill and decode want opposite things. Decode is bandwidth-bound — it
streams every weight once per token, so 4-bit weights are the whole point, and
mlir-air already has a fused Q4NX decode that unpacks nibbles on-device.
Prefill is compute-bound: W4A16 would cut traffic that is not the bottleneck
while adding unpack work to every GEMM tile. So the prefill dequantizes to
bf16 **on the host** at load and runs plain bf16 GEMMs, which is also what
mlir-air's own prefill does.

That leaves prefill as the half worth writing in Triton, and it is the half
this example owns.

## The interface between them is one file

```python
np.savez(path, k=K, v=V, first=first, prompt=prompt)   # K, V: [16, P, 512] f32
```

No shared weight format, no buffer plumbing, no ELF coupling. Any prefill that
emits that npz drops into mlir-air's `generate()`. `save_kv_npz` in `model.py`
states the layout — head order, the half-split RoPE convention, and the llama3
frequency scaling that a reimplementation silently gets wrong.

`llama32_1b_q4nx_inference.py` writes the npz and then neutralizes the one
step in mlir-air's `generate()` that would have produced it. Everything after
that — `seed_kv`, the dispatch loop, sampling — is mlir-air's code untouched.

## The decode is lowered by this backend

The decode superkernel does not fit the Triton compiler path: one dispatch
spans every decoder layer, weights are resident across tokens, and the
per-token work is an instruction patch rather than a launch. So it enters from
the other end. `decode_build.py` takes mlir-air's `fused_decode.build_module()`
for the AIR IR and hands it to `FusedDecodeOp`
(`amd_triton_npu/backend/fused_decode_op.py`), which calls the same
`_aircc_compile` every other kernel in this backend uses — same aircc
invocation, same stack size and extra-args plumbing, same artifact handling.
From there it is an ordinary AIR design.

    make compile-decode      # from nothing: AIE kernels + xclbins, ~22 s
    make recompile-decode    # just relower the xclbins, ~9 s
    make decode-kernels      # just rebuild the AIE kernels, ~13 s

Verified against mlir-air's own Makefile build: `insts.bin` byte-identical, the
xclbin differing only in its UUID and timestamps, and generation producing the
same token ids.

## What runs where

| Op | Device | Kernel |
|---|---|---|
| all 7 projections per layer (112 GEMMs) | **NPU** | `kernels.triton_matmul` |
| RMSNorm (33) | **NPU** | `kernels.triton_rms_norm` |
| SwiGLU (16) | **NPU** | `kernels.triton_swiglu` |
| RoPE | CPU | no transform script — see below |
| causal GQA attention | CPU | no transform script |
| LM head | CPU | one GEMV, off the hot path |
| decode (all 16 layers, Q4NX) | **NPU** | mlir-air, unmodified |

Two ops stay on the CPU for the same reason `examples/gpt2 --backend npu`
leaves attention there: no transform script yet, not a Triton limitation.

**RoPE** is the more interesting of the two. Its two half-row stores lower to
two `linalg.generic`s, and every elementwise sequence in mlir-air's transform
library matches a *single* payload op, so the script aborts with "expected a
single payload op" and the half-lowered IR then fails legalization. Folding
the halves into one generic needs either a host-side rotate-half — the data
movement the kernel exists to avoid — or a transform sequence written for a
two-output elementwise op.

**Attention** is the one with real headroom, and it is a bigger job: online
softmax carries a running max and accumulator across blocks, so
`air-label-scf-for-to-ping-pong` declines it and the schedule has to be
hand-written.

## RMSNorm needed its own transform script

`examples/weighted_rms_norm/transform_aie2p.mlir` casts the row reduction to
bf16. That is fine at the N=256 it is exercised at; at Llama's N=2048 a bf16
sum of 2048 positive terms undercounts systematically — the accumulator
outgrows the tail — and `rstd` came out ~9% high on every one of the model's
33 norms.

`transform_rms_norm_aie2p.mlir` is that script with the reduction left in f32.
The multiplies are still cast to bf16, so this is bf16 products with an f32
accumulator — exactly what mlir-air's own `rms_residual.cc` does
(`aie::mac_square` into an `accfloat` accumulator, then one 16-lane f32
`aie::reduce_add`). Relative error: 9.5% → 0.4%.

## Setup

mlir-air's *sources* are needed too — `fused_decode.build_module()` for the
AIR IR and `kernels/*.cc` for the AIE kernels — and the wheel does not ship
`programming_examples/`. `make compile-decode` fetches them at the commit
`utils/mlir-air-hash.txt` already pins, the same one the wheel version is built
from, so the toolchain and the sources that generated it cannot disagree. It is
a sparse blobless checkout of two directories (~1.7 MB of sources) into
`third_party/mlir-air-src/`, and it is skipped if you have your own clone at
`mlir-air-local/` or point `AIR_LLMS_ROOT` at one.

**A virtualenv must be active.** `utils/env_setup.sh` pip-installs into
whatever Python is current, and a system Python is externally managed
(PEP 668), so it fails there -- and then `make` runs against a Python with no
`triton` and dies in an import. From the repo root:

```bash
source sandbox/bin/activate                  # whatever your venv is named
source /opt/xilinx/xrt/setup.sh
source utils/env_setup.sh                    # installs mlir_air[aie] + llvm-aie
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers ml_dtypes           # transformers only for --text
```

`make preflight` checks all of the above and says which piece is missing.

One-time, and only needed for `chat`/`run` (not `prefill`):

```bash
make compile-decode      # ~22 s; the decode's AIE kernels, then its xclbins
```

Iterating on the AIR builder or the lowering afterwards only needs
`make recompile-decode` (~9 s) — the AIE kernels are C++ and rarely change.

Both steps are owned here. `decode_kernels.py` compiles the six AIE kernels
with Peano, byte-identically to mlir-air's Makefile and cached on the source
and flags; `decode_build.py` lowers the xclbins through `FusedDecodeOp`. What
is needed from mlir-air is now *source* — `fused_decode.py` and `kernels/` —
not its build system. That is also why this went from ~15 min to ~22 s: the
delegated target spent almost all of it building templates we relowered
anyway.

Weights (`model.q4nx`, 1.3 GB) are fetched from the Hub on first use;
`Q4NX_MODEL_SOURCE` overrides the repo or points at a local dir.

## Usage

```bash
make chat                       # interactive REPL  <-- start here
make run PROMPT="..." N_TOKENS=50
make prefill                    # prefill only + the Paris gate
make profile                    # per-operator prefill timing
make compile-decode             # one-time, only needed for chat/run
```

`make chat` loads the weights once and keeps both the Triton prefill and the
AIR decode resident, so every turn is prefill compute only. `/clear` resets the
conversation, `/exit` quits.

Underneath:

```bash
# Prefill only -- no decode build required. Gates on first token == 12366.
python llama32_1b_q4nx_inference.py --prefill-only

# End to end
python llama32_1b_q4nx_inference.py --backend npu --max-tokens 20 --greedy
python llama32_1b_q4nx_inference.py --text "The theory of relativity was developed by"

# The prefill on its own, with per-op routing and a CPU cross-check
python prefill.py --backend npu --ops all
python prefill.py --backend npu --ops matmul --compare-cpu
python prefill.py --backend cpu --kv-out /tmp/kv.npz
```

### Running on the HSA runtime

```bash
make compile-decode RUNTIME=hsa   # the decode as one full ELF, not xclbin
make chat RUNTIME=hsa
```

Both halves then dispatch through `HsaRuntime` rather than XRT, and the
decode's logits are bit-identical to the XRT path.

It is not as fast. Measured on Strix, 60 tokens, runtimes alternated:

| | tok/s |
|---|---|
| XRT | 54.5 |
| HSA | 46.9 – 47.3 |

The token ids are identical between the two, so this is a throughput gap and
not a correctness one.

The interesting part is how the context length gets to the device. It is a
**scratchpad parameter**: two scalars in device memory that the design reads in
its dispatch preamble, so one full ELF serves every context length and nothing
rewrites the instruction stream per token. That replaced a calibration — two
builds at adjacent L, a diff to find the 248 L-dependent words, and a linear
extrapolation patched in on every dispatch.

It needs a ROCR that can resolve the device address of an application's buffer
(`hsa_amd_aie_agent_device_address`). A full-ELF design reaches its scratchpad
through an address patched into its control code, and that address is the one
the NPU sees, not the host address the allocation is known by. Without it
`triton_npu_hsa_prepare_elf` refuses the design rather than hanging on it.

About 3 ms/token of the gap to XRT is unattributed. One measured candidate: the
full-ELF control code is 354764 bytes against the older path's 158032-byte
instruction stream, and ROCR walks `insts_size` with CLFLUSH on every dispatch
even though the control code only changes when an argument address does — which,
for a decode, is never after the first token.

`--ops` takes `all` or a comma list of `matmul,rms_norm,swiglu`, so a
numerical regression can be bisected to a single kernel against the same CPU
reference.

## Files

```
llama32_1b_q4nx_inference.py  # end to end: our prefill -> mlir-air's decode
prefill.py                    # prefill alone, with the Paris gate
model.py                      # the 16-layer forward pass and its operators
kernels.py                    # the Triton kernels + NPU dispatch plumbing
config.py                     # dims; re-exports mlir-air's loader and RoPE table
transform_rms_norm_aie2p.mlir # f32 row reduction (see above)
```

Matmul and SwiGLU reuse the transform scripts from `examples/gpt2` and
`examples/swiglu` unchanged.

Q, K and V share one normalized hidden state, and gate and up share another,
so each pair is issued as a single GEMM over concatenated weights. An NPU
dispatch costs ~24 ms of fixed overhead whatever its size, so this removes 48
launches per prefill for identical arithmetic.

## Accuracy

The NPU path is bf16 with f32 accumulation, so it does not reproduce the CPU
reference bit for bit. Measured against it on the canonical prompt, mean
relative error in the emitted KV cache:

| layer | K | V |
|---|---|---|
| 0 | 0.4% | 1.7% |
| 7 | 3.6% | 7.0% |
| 15 | 2.5% | 5.1% |

Layer 0 is one GEMM's worth of bf16 rounding; the growth with depth is that
error compounding through the residual stream, not a kernel defect. mlir-air's
own bf16 prefill sits in the same place. `--compare-cpu` prints this table for
any run, and `--ops` narrows it to a single kernel.

## Numbers

P=6, NPU prefill with all three kernels enabled, on Strix (NPU2):

| | |
|---|---|
| host weight load (host dequant, 1.3 GB) | ~9.6 s |
| prefill | ~1.4 s |
| decode (XRT) | ~54 tok/s |
| decode (HSA) | ~51 tok/s |

The prefill figure is not a speed result. Every GEMM pads M to 128 for a
6-token prompt -- 128 is the floor the matmul transform script's herd tiling
assumes -- so ~95% of the matmul work is padding; the point here is that
the kernels are Triton and they run on the NPU. Real prefill throughput needs
an NPU attention kernel and M-padding that tracks the prompt.
