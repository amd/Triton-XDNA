# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# LLM Inference Building Blocks on AMD Ryzen AI NPU
# ==================================================
# Demonstrates offloading core LLM operations to the NPU via Triton-XDNA.
#
# What runs on the NPU:
#   - RMSNorm (pre-attention & pre-FFN normalization)
#   - SiLU activation (FFN gate activation in LLaMA/Mistral)
#   - SwiGLU (fused gate * up projection activation)
#   - GELU activation (used in GPT-2, BERT FFN)
#   - Residual addition (skip connections)
#   - Elementwise multiply (weight scaling)
#
# What stays on CPU (for now):
#   - Matrix multiplications (QKV projections, FFN linear layers)
#     → matmul works but current NPU tiling is fixed-size; using
#       torch.matmul on CPU is faster for variable-size inference
#   - Attention score computation & softmax
#   - Embedding lookup, sampling/argmax
#
# Each kernel is individually verified against a PyTorch reference.
# At the end, they're composed into a simplified LLaMA-style transformer
# block that runs a mix of NPU + CPU ops.

import torch
import triton
import triton.language as tl
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import benchmark

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-5

# ============================================================================
# NPU Kernel: SiLU activation  —  silu(x) = x * sigmoid(x)
# Used in: LLaMA/Mistral FFN gate pathway
# ============================================================================
@triton.jit
def silu_kernel(X, Y, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets[:])
    x_f32 = x.to(tl.float32)
    y = (x_f32 * tl.sigmoid(x_f32)).to(x.dtype)
    tl.store(Y + offsets[:], y)


# ============================================================================
# NPU Kernel: GELU activation  —  gelu(x) ≈ x * sigmoid(1.702 * x)
# Used in: GPT-2, BERT, Phi FFN
# ============================================================================
@triton.jit
def gelu_kernel(X, Y, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets[:])
    x_f32 = x.to(tl.float32)
    y = (x_f32 * tl.sigmoid(1.702 * x_f32)).to(x.dtype)
    tl.store(Y + offsets[:], y)


# ============================================================================
# NPU Kernel: SwiGLU  —  swiglu(gate, up) = silu(gate) * up
# Used in: LLaMA 2/3, Mistral FFN (fused gate + up projection activation)
# ============================================================================
@triton.jit
def swiglu_kernel(
    GATE, UP, OUT, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    gate = tl.load(GATE + offsets[:])
    up = tl.load(UP + offsets[:])
    gate_f32 = gate.to(tl.float32)
    silu_gate = (gate_f32 * tl.sigmoid(gate_f32)).to(gate.dtype)
    tl.store(OUT + offsets[:], silu_gate * up)


# ============================================================================
# NPU Kernel: Residual Add  —  out = x + residual
# Used in: Every transformer block skip connection
# ============================================================================
@triton.jit
def residual_add_kernel(
    X, RESIDUAL, OUT, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets[:])
    r = tl.load(RESIDUAL + offsets[:])
    tl.store(OUT + offsets[:], x + r)


# ============================================================================
# NPU Kernel: Elementwise Multiply  —  out = x * w
# Used in: RMSNorm weight scaling (x_normed * weight)
# ============================================================================
@triton.jit
def elementwise_mul_kernel(
    X, W, OUT, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets[:])
    w = tl.load(W + offsets[:])
    tl.store(OUT + offsets[:], x * w)


# ============================================================================
# Wrappers that handle grid launch + transform script selection
# ============================================================================
def set_transform(name):
    """Set the AIR transform tiling script for the NPU compilation."""
    script = os.path.join(SCRIPT_DIR, name)
    os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = script


def npu_silu(x, out=None):
    set_transform("transform_unary_aie2p.mlir")
    if out is None:
        out = torch.empty_like(x)
    N = x.numel()
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    silu_kernel[grid](x.contiguous().view(-1), out.view(-1), N, BLOCK_SIZE=1024)
    return out


def npu_gelu(x, out=None):
    set_transform("transform_unary_aie2p.mlir")
    if out is None:
        out = torch.empty_like(x)
    N = x.numel()
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    gelu_kernel[grid](x.contiguous().view(-1), out.view(-1), N, BLOCK_SIZE=1024)
    return out


def npu_swiglu(gate, up, out=None):
    set_transform("transform_binary_aie2p.mlir")
    if out is None:
        out = torch.empty_like(gate)
    N = gate.numel()
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    swiglu_kernel[grid](
        gate.contiguous().view(-1),
        up.contiguous().view(-1),
        out.view(-1),
        N,
        BLOCK_SIZE=1024,
    )
    return out


def npu_residual_add(x, residual, out=None):
    set_transform("transform_binary_aie2p.mlir")
    if out is None:
        out = torch.empty_like(x)
    N = x.numel()
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    residual_add_kernel[grid](
        x.contiguous().view(-1),
        residual.contiguous().view(-1),
        out.view(-1),
        N,
        BLOCK_SIZE=1024,
    )
    return out


def npu_elementwise_mul(x, w, out=None):
    set_transform("transform_binary_aie2p.mlir")
    if out is None:
        out = torch.empty_like(x)
    N = x.numel()
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    elementwise_mul_kernel[grid](
        x.contiguous().view(-1),
        w.contiguous().view(-1),
        out.view(-1),
        N,
        BLOCK_SIZE=1024,
    )
    return out


# ============================================================================
# Composite: RMSNorm  —  uses CPU for reduction, NPU for scaling
# rms_norm(x, w) = (x / sqrt(mean(x²) + eps)) * w
# ============================================================================
def npu_rms_norm(x, weight):
    """RMSNorm: reduction on CPU, element-wise scaling on NPU."""
    # CPU: compute the per-row RMS (reduction — small, fast on CPU)
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + EPS)
    x_normed = (x_f32 / rms).to(x.dtype)
    # NPU: element-wise multiply with learned weight (this is the big operation)
    return npu_elementwise_mul(x_normed.view(-1), weight.expand_as(x_normed).contiguous().view(-1)).\
        view_as(x)


# ============================================================================
# Individual kernel tests
# ============================================================================
def test_silu():
    print("Testing SiLU on NPU...", end=" ", flush=True)
    x = torch.randn(4096, dtype=torch.bfloat16)
    ref = torch.nn.functional.silu(x)
    out = npu_silu(x)
    torch.testing.assert_close(out, ref, atol=0.1, rtol=0.1)
    print("PASS")


def test_gelu():
    print("Testing GELU on NPU...", end=" ", flush=True)
    x = torch.randn(4096, dtype=torch.bfloat16)
    ref = x * torch.sigmoid(1.702 * x.float()).to(torch.bfloat16)
    out = npu_gelu(x)
    torch.testing.assert_close(out, ref, atol=0.1, rtol=0.1)
    print("PASS")


def test_swiglu():
    print("Testing SwiGLU on NPU...", end=" ", flush=True)
    gate = torch.randn(4096, dtype=torch.bfloat16)
    up = torch.randn(4096, dtype=torch.bfloat16)
    ref = torch.nn.functional.silu(gate) * up
    out = npu_swiglu(gate, up)
    torch.testing.assert_close(out, ref, atol=0.1, rtol=0.1)
    print("PASS")


def test_residual_add():
    print("Testing Residual Add on NPU...", end=" ", flush=True)
    x = torch.randn(4096, dtype=torch.bfloat16)
    r = torch.randn(4096, dtype=torch.bfloat16)
    ref = x + r
    out = npu_residual_add(x, r)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("PASS")


def test_rms_norm():
    print("Testing RMSNorm on NPU...", end=" ", flush=True)
    hidden = 1024
    x = torch.randn(1, hidden, dtype=torch.bfloat16)
    w = torch.randn(hidden, dtype=torch.bfloat16)
    # Reference
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + EPS)
    ref = ((x_f32 / rms) * w.float()).to(torch.bfloat16)
    out = npu_rms_norm(x, w)
    torch.testing.assert_close(out, ref, atol=0.5, rtol=0.1)
    print("PASS")


# ============================================================================
# Simulated LLaMA-style Transformer Block
# ============================================================================
class LLaMABlockNPU:
    """
    Simplified single-head LLaMA transformer block.
    NPU handles: RMSNorm, SwiGLU activation, residual connections.
    CPU handles: Linear projections (matmul), attention scores, softmax.

    This is intentionally simplified to demonstrate NPU offloading:
    - Single attention head (no multi-head splitting)
    - Fixed sequence length
    - No KV cache (single-pass)
    """

    def __init__(self, hidden_dim, ffn_dim):
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        # Normalization weights
        self.norm1_weight = torch.randn(hidden_dim, dtype=torch.bfloat16) * 0.02
        self.norm2_weight = torch.randn(hidden_dim, dtype=torch.bfloat16) * 0.02
        # Attention projection weights (CPU matmul)
        self.wq = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
        self.wk = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
        self.wv = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
        self.wo = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
        # FFN weights (CPU matmul)
        self.w_gate = torch.randn(hidden_dim, ffn_dim, dtype=torch.bfloat16) * 0.02
        self.w_up = torch.randn(hidden_dim, ffn_dim, dtype=torch.bfloat16) * 0.02
        self.w_down = torch.randn(ffn_dim, hidden_dim, dtype=torch.bfloat16) * 0.02

    def forward(self, x):
        """
        x: [batch=1, seq_len, hidden_dim] bf16

        Returns: [batch=1, seq_len, hidden_dim] bf16
        """
        seq_len = x.shape[1]

        # ---- Pre-Attention RMSNorm [NPU] ----
        normed = npu_rms_norm(x.view(-1, self.hidden_dim), self.norm1_weight)
        normed = normed.view(1, seq_len, self.hidden_dim)

        # ---- Self-Attention [CPU] ----
        # Linear projections (matmul on CPU — variable size, fast enough)
        q = normed @ self.wq    # [1, seq, hid] @ [hid, hid] → [1, seq, hid]
        k = normed @ self.wk
        v = normed @ self.wv

        # Scaled dot-product attention (CPU)
        scale = self.hidden_dim ** -0.5
        attn_scores = (q @ k.transpose(-2, -1)) * scale  # [1, seq, seq]
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(torch.bfloat16)
        attn_out = attn_weights @ v  # [1, seq, hid]

        # Output projection (CPU matmul)
        attn_out = attn_out @ self.wo

        # ---- Residual Add [NPU] ----
        h = npu_residual_add(
            attn_out.view(-1), x.view(-1)
        ).view(1, seq_len, self.hidden_dim)

        # ---- Pre-FFN RMSNorm [NPU] ----
        normed2 = npu_rms_norm(h.view(-1, self.hidden_dim), self.norm2_weight)
        normed2 = normed2.view(1, seq_len, self.hidden_dim)

        # ---- FFN with SwiGLU [NPU activation, CPU matmul] ----
        gate = normed2 @ self.w_gate   # [1, seq, ffn_dim]  (CPU)
        up = normed2 @ self.w_up       # [1, seq, ffn_dim]  (CPU)

        # SwiGLU activation [NPU] — this is the expensive elementwise part
        ffn_act = npu_swiglu(gate.view(-1), up.view(-1))
        ffn_act = ffn_act.view(1, seq_len, self.ffn_dim)

        # Down projection (CPU matmul)
        ffn_out = ffn_act @ self.w_down  # [1, seq, hid]

        # ---- Residual Add [NPU] ----
        out = npu_residual_add(
            ffn_out.view(-1), h.view(-1)
        ).view(1, seq_len, self.hidden_dim)

        return out


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    benchmark.select_npu_backend()

    print("=" * 60)
    print("LLM Inference Kernels on AMD Ryzen AI NPU")
    print("=" * 60)
    print()

    # ---- Test individual kernels ----
    print("--- Individual Kernel Tests ---")
    test_silu()
    test_gelu()
    test_swiglu()
    test_residual_add()
    test_rms_norm()
    print()

    # ---- Run a transformer block ----
    print("--- LLaMA-style Transformer Block (NPU + CPU) ---")
    hidden_dim = 1024
    ffn_dim = 2048  # Typically ~2.7x hidden for LLaMA, using 2x for simplicity
    seq_len = 8     # Short sequence for demo
    block = LLaMABlockNPU(hidden_dim, ffn_dim)

    x = torch.randn(1, seq_len, hidden_dim, dtype=torch.bfloat16)

    print(f"  Config: hidden={hidden_dim}, ffn={ffn_dim}, seq_len={seq_len}")
    print(f"  Input:  {list(x.shape)} bf16")

    # Warmup
    print("  Compiling NPU kernels (first pass)...", flush=True)
    t0 = time.perf_counter()
    out = block.forward(x)
    compile_time = time.perf_counter() - t0
    print(f"  First pass (compile+run): {compile_time:.1f}s")
    print(f"  Output: {list(out.shape)} bf16")
    print(f"  Output sample: {out[0, 0, :8].tolist()}")

    # Benchmark cached
    print("  Running 5 cached iterations...", flush=True)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = block.forward(x)
        times.append(time.perf_counter() - t0)

    print(f"  Mean: {sum(times)/len(times)*1000:.1f} ms")
    print(f"  Min:  {min(times)*1000:.1f} ms")
    print()

    print("NPU ops per block: 2x RMSNorm + 1x SwiGLU + 2x Residual Add = 5 dispatches")
    print("CPU ops per block: 7x matmul + 1x softmax + 1x attention")
    print()
    print("In production, the NPU shines for always-on inference where")
    print("these activations run continuously at low power, freeing the")
    print("CPU/GPU for other work.")
