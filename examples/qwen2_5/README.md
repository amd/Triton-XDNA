# Qwen2.5 End-to-End Inference

End-to-end inference of Qwen2.5-Instruct using Triton kernels on AMD iGPU and
NPU. Loads pretrained weights from HuggingFace, runs forward passes with
KV-cached autoregressive generation, and validates output against the
HuggingFace reference.

Config-driven across model sizes via `--model`; the same kernels and transform
scripts serve every size:

| `--model` | Params | Layers | Q heads | KV heads | Hidden | Head dim | MLP |
|-----------|--------|--------|---------|----------|--------|----------|-----|
| `qwen2.5-0.5b` (default) | 494M | 24 | 14 | 2 | 896 | 64 | 4,864 |
| `qwen2.5-1.5b` | 1.5B | 28 | 12 | 2 | 1,536 | 128 | 8,960 |

## Setup

**Prerequisite: a ROCm torch device is required for all backends except
`reference`.** Every mode — including `--backend npu` — runs attention (Q/K/V/O
projections, RoPE, the fused grouped-query attention kernel) and the LM head on
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
# Select model size with --model (default: qwen2.5-0.5b)
python qwen_inference.py --model qwen2.5-1.5b --backend gpu

# Single forward pass (prefill only, compares logits to HuggingFace)
python qwen_inference.py --backend gpu
python qwen_inference.py --backend hetero
python qwen_inference.py --backend hetero-fast

# Autoregressive generation with KV cache
python qwen_inference.py --backend gpu --max-tokens 20
python qwen_inference.py --backend hetero-fast --max-tokens 20

# Interactive mode
python qwen_inference.py --backend hetero-fast --interactive --max-tokens 50

# Custom prompt (chat template applied by default)
python qwen_inference.py --backend gpu --max-tokens 20 --prompt "What is the capital of France?"

# Raw prompt without the chat template
python qwen_inference.py --backend gpu --no-chat --prompt "Once upon a time"

# HuggingFace reference only (no Triton)
python qwen_inference.py --backend reference
```

## Backend Modes

Four backends route operators across devices differently:

| Backend | Description |
|---------|-------------|
| `gpu` | All ops on iGPU via ROCm/Triton |
| `npu` | RMSNorm/MLP/add on NPU; attention and LM head on iGPU |
| `hetero` | Attention on GPU, RMSNorm/MLP/add on NPU |
| `hetero-fast` | Same as hetero for prefill; all-GPU decode |

### Per-Op Device Routing

| Op | `gpu` | `npu` | `hetero` | `hetero-fast` prefill | `hetero-fast` decode |
|----|-------|-------|----------|----------------------|---------------------|
| RMSNorm (input) | GPU | NPU | NPU | NPU | **GPU** |
| Q/K/V projection | GPU | GPU | GPU | GPU | GPU |
| RoPE | GPU | GPU | GPU | GPU | GPU |
| Fused attention (GQA) | GPU | GPU | GPU | GPU | GPU |
| Output projection | GPU | GPU | GPU | GPU | GPU |
| Residual add | GPU | NPU | NPU | NPU | **GPU** |
| RMSNorm (post-attn) | GPU | NPU | NPU | NPU | **GPU** |
| MLP gate/up proj | GPU | NPU | NPU | NPU | **GPU** |
| SwiGLU (SiLU * up) | GPU | NPU | NPU | NPU | **GPU** |
| MLP down proj | GPU | NPU | NPU | NPU | **GPU** |
| Residual add | GPU | NPU | NPU | NPU | **GPU** |
| Final RMSNorm | GPU | NPU | NPU | NPU | **GPU** |
| LM head | GPU | GPU | GPU | GPU | GPU |

Attention (Q/K/V/O, RoPE, the fused attention kernel) always runs on the iGPU:
the fused FlashAttention-style kernel is GPU-only, and Qwen's grouped-query
attention is expanded to full heads before the kernel. The LM head also always
runs on the iGPU.

## Architecture

### Qwen2.5 Parameters

| Parameter | 0.5B | 1.5B |
|-----------|------|------|
| Vocab size | 151,936 | 151,936 |
| Hidden dim | 896 | 1,536 |
| Layers | 24 | 28 |
| Attention heads | 14 | 12 |
| KV heads (GQA) | 2 | 2 |
| Head dim | 64 | 128 |
| MLP intermediate dim | 4,864 | 8,960 |
| Activation | SwiGLU (SiLU) | SwiGLU (SiLU) |
| Normalization | RMSNorm (eps 1e-6) | RMSNorm (eps 1e-6) |
| Position encoding | RoPE (theta 1e6) | RoPE (theta 1e6) |
| Tied embeddings | Yes | Yes |

### How Qwen2 differs from GPT-2

| Aspect | GPT-2 | Qwen2.5 |
|--------|-------|---------|
| Normalization | LayerNorm (mean + var, bias) | RMSNorm (var only, no bias) |
| Position encoding | Learned absolute | RoPE rotary |
| Attention | Multi-head | Grouped-query (14 or 12 Q / 2 KV) |
| MLP | fc -> GELU -> proj | gate/up -> SiLU*up -> down (SwiGLU) |
| QKV bias | yes (combined c_attn) | yes (separate q/k/v) |
| Output/MLP bias | yes | no |
| Embeddings | tied | tied |

### Forward Pass

Each transformer layer (24 for 0.5B, 28 for 1.5B) runs:

1. **RMSNorm** -> **Q/K/V projection** -> **RoPE** -> **GQA fused attention** -> **Output projection** -> **Residual add**
2. **RMSNorm** -> **gate/up projection** -> **SwiGLU** -> **down projection** -> **Residual add**

Followed by a final RMSNorm and the LM head (tied embedding weights).

Autoregressive generation uses pre-allocated KV caches with in-place writes.

### File Structure

```
qwen_inference.py                   # Entry point: load, run, compare
model.py                            # Qwen2Model: forward pass, weight placement, KV cache, QWEN_CONFIGS
kernels/
    __init__.py                     # Re-exports all kernel wrappers
    matmul.py                       # Linear layers (GPU + NPU, single full-K launch)
    softmax.py                      # Row-wise softmax (GPU + NPU)
    rmsnorm.py                      # RMSNorm with weight (GPU + NPU)
    swiglu.py                       # SwiGLU: SiLU(gate)*up (GPU fused; NPU silu+mul)
    rope.py                         # Rotary position embeddings (torch, applied pre-attention)
    add.py                          # Elementwise addition (GPU + NPU)
    attention.py                    # Fused multi-head attention (GPU only)
    backend_utils.py                # CachedNPUKernel, npu_driver_scope
transform_matmul_aie2p.mlir         # NPU tiling recipe for matmul
transform_rmsnorm_aie2p.mlir        # NPU tiling recipe for RMSNorm (single reduction)
transform_elementwise_aie2p.mlir    # NPU tiling recipe for SiLU
transform_add_aie2p.mlir            # NPU tiling recipe for add / elementwise multiply
transform_softmax_aie2p.mlir        # NPU tiling recipe for softmax
```

### Kernel Design

Each kernel file has GPU and NPU `@triton.jit` variants plus a Python wrapper
handling dtype conversion, padding, device transfers, and transform-script
selection. Every wrapper includes a PyTorch fallback if compilation fails.

**GPU kernels** use standard Triton patterns with ROCm (`device="cuda"`).
**NPU kernels** follow MLIR-AIR conventions: power-of-2 block sizes, padding
to transform script tile boundaries, bf16 input with f32 accumulation.

The matmul NPU path keeps a 256x256 program block (required by the transform
script's multi-core herd tiling) and issues a single full-K launch: K is padded
to a power of two and the transform script's `k_reduction_loop` tiles the K
reduction across L3/L2/L1 on-device, so K is not chunked on the host. This lets
the large `down_proj` (K=4864 for 0.5B, K=8960 padded to 16384 for 1.5B) run on
NPU in one dispatch.

RoPE and grouped-query attention run on the GPU alongside the fused attention
kernel; KV heads are expanded to match the query heads before the kernel.

NPU dispatch uses `CachedNPUKernel` for fast-path re-invocation.

## CLI Reference

```
python qwen_inference.py [OPTIONS]

Options:
  --model {qwen2.5-0.5b,qwen2.5-1.5b}
  --backend {gpu,npu,hetero,hetero-fast,reference}
  --prompt TEXT         Input prompt (default: a short LLM intro request)
  --max-tokens N        Tokens to generate; 0 = single forward pass (default: 0)
  --interactive         Interactive REPL mode
  --profile             Enable per-op timing breakdown
  --no-chat             Disable the Qwen chat template (use the raw prompt)
  --verbose             Debug logging
```
