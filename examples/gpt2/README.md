# GPT-2 End-to-End Inference

End-to-end inference of GPT-2 using Triton kernels on AMD iGPU and NPU.
Loads pretrained weights from HuggingFace, runs forward passes with
KV-cached autoregressive generation, and validates output against the
HuggingFace reference.

Config-driven across all four GPT-2 sizes via `--model`. The variants share
architecture, vocab (50257), context length (1024), and head_dim (64) — only
depth and width differ — so the same kernels and transform scripts serve
every size with no changes.

| `--model` | Params | Layers | Heads | Hidden | MLP |
|-----------|--------|--------|-------|--------|------|
| `gpt2` (default) | 124M | 12 | 12 | 768 | 3072 |
| `gpt2-medium` | 355M | 24 | 16 | 1024 | 4096 |
| `gpt2-large` | 774M | 36 | 20 | 1280 | 5120 |
| `gpt2-xl` | 1.5B | 48 | 25 | 1600 | 6400 |

## Quick Start

```bash
# Prerequisites
pip install transformers

# Environment setup (required for NPU/hetero modes)
source /opt/xilinx/xrt/setup.sh
source utils/env_setup.sh

# Single forward pass (prefill only, compares logits to HuggingFace)
python gpt2_inference.py --backend gpu
python gpt2_inference.py --backend hetero
python gpt2_inference.py --backend hetero-fast

# Larger variants (same flags, just add --model)
python gpt2_inference.py --model gpt2-xl --backend gpu
python gpt2_inference.py --model gpt2-xl --backend hetero-fast --max-tokens 20

# Autoregressive generation with KV cache
python gpt2_inference.py --backend gpu --max-tokens 20
python gpt2_inference.py --backend hetero-fast --max-tokens 20

# Interactive mode
python gpt2_inference.py --backend hetero-fast --interactive --max-tokens 50

# Custom prompt
python gpt2_inference.py --backend gpu --max-tokens 20 --prompt "Once upon a time"

# Per-op timing profile
python gpt2_inference.py --backend hetero --max-tokens 10 --profile

# HuggingFace reference only (no Triton)
python gpt2_inference.py --backend reference
```

## Backend Modes

Four backends route operators across devices differently:

| Backend | Description | Best for |
|---------|-------------|----------|
| `gpu` | All ops on iGPU via ROCm/Triton | Baseline, lowest decode latency |
| `npu` | All ops on NPU via MLIR-AIR/AIE | NPU development and testing |
| `hetero` | Attention on GPU, LN/MLP/add on NPU | Consistent NPU utilization |
| `hetero-fast` | Same as hetero for prefill; all-GPU decode | Best hetero decode latency |

### Per-Op Device Routing

| Op | `gpu` | `npu` | `hetero` | `hetero-fast` prefill | `hetero-fast` decode |
|----|-------|-------|----------|----------------------|---------------------|
| LayerNorm (ln1) | GPU | NPU | NPU | NPU | **GPU** |
| QKV projection | GPU | NPU | GPU | GPU | GPU |
| Fused attention | GPU | NPU | GPU | GPU | GPU |
| Output projection | GPU | NPU | GPU | GPU | GPU |
| Residual add | GPU | NPU | NPU | NPU | **GPU** |
| LayerNorm (ln2) | GPU | NPU | NPU | NPU | **GPU** |
| MLP up-proj | GPU | NPU | NPU | NPU | **GPU** |
| GELU | GPU | NPU | NPU | NPU | **GPU** |
| MLP down-proj | GPU | NPU | NPU | NPU | **GPU** |
| Residual add | GPU | NPU | NPU | NPU | **GPU** |
| Final LayerNorm | GPU | NPU | NPU | NPU | **GPU** |
| LM head | GPU | GPU | GPU | GPU | GPU |

### Why `hetero-fast`?

During decode (sequence length = 1), tensors are tiny (768 or 3072 elements),
so per-dispatch NPU overhead dominates regardless of tensor size. Each launch
re-instantiates the XRT hardware context (buffer setup, DMA, sync, and AIE array
programming), which is expensive. With 8 NPU ops per layer x 12 layers + 1 ln_f
= 97 dispatches per token, that overhead is far larger than the ~6ms the iGPU
needs for the whole decode step (see the known issue under Expected Performance).
`hetero-fast` eliminates this by routing all decode ops to GPU while keeping
NPU for prefill, where larger tensors justify the dispatch cost.

## Expected Performance

Measured on AMD Ryzen AI MAX+ PRO 395 -- Radeon 8060S iGPU (gfx1151, RDNA 3.5)
+ NPU (Strix Halo, AIE2P / XDNA2). gpt2 (124M), prompt "The quick brown fox"
(4 tokens), generating 20 tokens, warm Triton cache.

### Decode Latency (Steady-State TPOT)

| Backend | Steady TPOT | Tokens/sec |
|---------|-------------|------------|
| `gpu` | ~6 ms | ~170 |
| `hetero-fast` | ~5 ms | ~205 |
| `hetero` | ~8.5 s | ~0.1 |
| `hetero` + `AMD_TRITON_NPU_FUSED_MLP=1` | ~86 ms | ~12 |

"Steady TPOT" excludes the first decode step (which includes JIT compilation).

`hetero` decode is slow by default: each NPU kernel launch re-instantiates the
XRT hardware context, and that per-launch context setup dominates decode latency.

Setting `AMD_TRITON_NPU_FUSED_MLP=1` fuses the MLP block
(`mlp_fc -> gelu -> mlp_proj`) plus the following residual add into a single
`load_pdi` multi-launch ELF that runs through **one** persistent `hw_context`,
reused across all layers (the mlir-air llama pattern: one ELF compiled once,
per-layer weights swapped in). This collapses four per-op NPU dispatches per
layer into one (the residual `add` rides the chain via DDR hand-off, with no
host round-trip) and removes the bulk of the context-rebuild cost, taking
`hetero` decode from ~8.5 s to ~86 ms (~100x). It is opt-in and applies to
`hetero`/`npu` modes (where the MLP runs on NPU). Decode is still dispatch-bound
-- at sequence length 1 the tensors are
tiny, so per-layer launch overhead dominates regardless. For the lowest decode
latency use `hetero-fast`, which routes decode to the iGPU and avoids per-step
NPU dispatch entirely.

### Prefill Latency (TTFT)

| Backend | TTFT (4 tokens, warm cache) |
|---------|-----------------------------|
| `gpu` | ~3.4 s |
| `hetero` | ~11 s |
| `hetero-fast` | ~11 s |

`gpu` TTFT is lower because prefill runs entirely on the iGPU. `hetero` and
`hetero-fast` run the prefill matmuls/elementwise ops on the NPU, which costs
more for a one-shot prefill but keeps the NPU utilized. Even with a warm Triton
cache, the first forward pass pays NPU compilation and XRT setup; these tables
reflect that warm-but-first-pass cost.

### Larger variants on NPU

All four sizes run end-to-end on the NPU. The single-block NPU matmul reduces
over the full K dimension on-device (the transform script tiles K across
L3/L2/L1), so the wider `gpt2-large`/`gpt2-xl` MLP matmuls (K padded to 8192)
compile and run as one launch each -- no host-side K-chunking and no PyTorch
fallback. For reference, `gpt2-large` (774M, 36 layers): `hetero-fast` decode
~15 ms TPOT; `hetero` + `AMD_TRITON_NPU_FUSED_MLP=1` ~540 ms TPOT (the fused MLP
shares one `hw_context` across all 36 layers, so deep models no longer exhaust
NPU hardware contexts).

### Accuracy

Single forward pass cosine similarity vs HuggingFace fp32 reference:

| Backend | Cosine sim | Top-1 match | Top-5 overlap |
|---------|-----------|-------------|---------------|
| `gpu` | 0.9999 | Yes | 5/5 |
| `hetero` | 0.9994 | Yes | 5/5 |
| `hetero-fast` | 0.9994 | Yes | 5/5 |

Remaining divergence in hetero modes is from bf16 matmul accumulation through
12 transformer layers. Both GPU and NPU kernels use identical GELU
(tanh approximation) and float32 layernorm precision.

## Architecture

### GPT-2 Small Parameters

| Parameter | Value |
|-----------|-------|
| Vocab size | 50,257 |
| Hidden dim | 768 |
| Layers | 12 |
| Attention heads | 12 |
| Head dim | 64 |
| MLP intermediate dim | 3,072 |
| Max sequence length | 1,024 |

### Forward Pass

Each of 12 transformer layers runs:

1. **LayerNorm** (ln1) -> **QKV projection** (matmul) -> **Multi-head attention** (fused Q@K, softmax, attn@V) -> **Output projection** (matmul) -> **Residual add**
2. **LayerNorm** (ln2) -> **MLP up-proj** (matmul, 768->3072) -> **GELU** -> **MLP down-proj** (matmul, 3072->768) -> **Residual add**

Followed by final LayerNorm and LM head (tied embedding weights).

Autoregressive generation uses pre-allocated KV caches with in-place writes
(no `torch.cat` per step).

### File Structure

```
gpt2_inference.py                  # Entry point: load, run, compare
model.py                           # GPT2Model: forward pass, weight placement, KV cache
kernels/
    __init__.py                    # Re-exports all kernel wrappers
    matmul.py                      # Linear layers (GPU + NPU)
    softmax.py                     # Row-wise softmax (GPU + NPU)
    layernorm.py                   # LayerNorm with gamma/beta (GPU + NPU)
    gelu.py                        # GELU activation (GPU + NPU)
    add.py                         # Elementwise addition (GPU + NPU)
    attention.py                   # Fused multi-head attention (GPU only)
    backend_utils.py               # CachedNPUKernel, npu_driver_scope
transform_matmul_aie2p.mlir        # NPU tiling recipe for matmul
transform_elementwise_aie2p.mlir   # NPU tiling recipe for GELU
transform_add_aie2p.mlir           # NPU tiling recipe for add
transform_softmax_aie2p.mlir       # NPU tiling recipe for softmax
transform_layernorm_aie2p.mlir     # NPU tiling recipe for layernorm
```

### Kernel Design

Each kernel file contains two `@triton.jit` kernels (GPU and NPU variants)
plus a Python wrapper handling dtype conversion, padding, device transfers,
and transform script selection. Every wrapper includes a PyTorch fallback
if compilation fails.

**GPU kernels** use standard Triton patterns with ROCm (`device="cuda"`).
**NPU kernels** follow MLIR-AIR conventions: power-of-2 block sizes, padding
to transform script tile boundaries, bf16 input with f32 accumulation.

NPU dispatch uses `CachedNPUKernel` for fast-path re-invocation (~0.1ms vs
~27ms for full Triton JIT dispatch).

## CLI Reference

```
python gpt2_inference.py [OPTIONS]

Options:
  --backend {gpu,npu,hetero,hetero-fast,reference}
                        Inference backend (default: gpu)
  --prompt TEXT         Input prompt (default: "The quick brown fox")
  --max-tokens N        Tokens to generate; 0 = single forward pass (default: 0)
  --interactive         Interactive REPL mode
  --profile             Enable per-op timing breakdown
  --verbose             Debug logging
```
