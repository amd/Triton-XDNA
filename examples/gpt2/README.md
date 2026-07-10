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

## Setup

**Prerequisite: a ROCm torch device is required for all backends except
`reference`.** Every mode — including `--backend npu` — runs the LM head on
the iGPU (see the routing table below), so a working ROCm/Triton GPU with a
ROCm build of PyTorch must be installed.

```bash
# Prerequisites
pip install transformers
# ROCm PyTorch (adjust the ROCm version to match your install)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# Environment setup (required for NPU/hetero modes)
source /opt/xilinx/xrt/setup.sh
source utils/env_setup.sh
```

## Usage

```bash
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

# HuggingFace reference only (no Triton)
python gpt2_inference.py --backend reference
```

## Backend Modes

Four backends route operators across devices differently:

| Backend | Description |
|---------|-------------|
| `gpu` | All ops on iGPU via ROCm/Triton |
| `npu` | All ops on NPU via MLIR-AIR/AIE, except the LM head (GPU) |
| `hetero` | Attention on GPU, LN/MLP/add on NPU |
| `hetero-fast` | Same as hetero for prefill; all-GPU decode |

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

## Architecture

### Forward Pass

Each transformer layer runs:

1. **LayerNorm** (ln1) -> **QKV projection** (matmul) -> **Multi-head attention** (fused Q@K, softmax, attn@V) -> **Output projection** (matmul) -> **Residual add**
2. **LayerNorm** (ln2) -> **MLP up-proj** (matmul) -> **GELU** -> **MLP down-proj** (matmul) -> **Residual add**

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

NPU dispatch uses `CachedNPUKernel` for fast-path re-invocation.

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
