#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Real LLM Inference with AMD Ryzen AI NPU Offloading
# ====================================================
# Loads HuggingFaceTB/SmolLM2-1.7B-Instruct and offloads SwiGLU activations
# (and optionally RMSNorm scaling) to the NPU via Triton-XDNA kernels.
#
# SmolLM2-1.7B is chosen because:
#   - hidden_size=2048, intermediate_size=8192 (both multiples of 1024,
#     matching our Triton BLOCK_SIZE without needing masking)
#   - LLaMA architecture: RMSNorm + SiLU/SwiGLU (our proven NPU kernels)
#   - Apache 2.0 license, freely downloadable
#   - 1.7B params in bf16 ≈ 3.4 GB RAM (trivial with 128GB)
#
# What runs on the NPU:
#   - SwiGLU activation in each MLP layer (fused SiLU + multiply)
#   - (optional) RMSNorm weight scaling
#
# What stays on CPU:
#   - Linear projections (matmul) — QKV, output, gate/up/down
#   - Attention: score computation, softmax, value aggregation
#   - Embedding lookup, final LM head, sampling

import torch
import triton
import triton.language as tl
import time
import os
import sys
import types

# Use all CPU cores for PyTorch matmul operations
torch.set_num_threads(os.cpu_count())

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import benchmark

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Direct NPU Dispatch (bypasses Triton JIT for cached kernels)
# =============================================================================

class FastNPUDispatch:
    """Cache loaded .pyd modules keyed by element count (N), calling them
    directly and completely bypassing the Triton JIT machinery, MLIR pipeline,
    hash computation, and Python cache lookups.

    Auto-capture: after each slow-path dispatch, the module is captured from
    driver._last_dispatched_module and stored for that N. Subsequent calls
    with the same N use the fast path. This means only the FIRST dispatch
    per unique N value pays the ~218ms Triton JIT overhead.
    """
    def __init__(self):
        self._by_n = {}  # n_elements -> (mod, grid, constexpr_vals)

    def auto_capture(self, n_elements, block_size=1024):
        """Capture _last_dispatched_module for this N value."""
        if n_elements in self._by_n:
            return
        from triton.backends.amd_triton_npu.driver import _last_dispatched_module
        if _last_dispatched_module is not None:
            grid = n_elements // block_size
            self._by_n[n_elements] = (
                _last_dispatched_module, grid, [n_elements, block_size]
            )

    def run(self, n_elements, *tensor_args):
        """Call the cached C extension directly with minimal Python overhead."""
        mod, grid, constexpr_vals = self._by_n[n_elements]
        mod.launch(
            grid, 1, 1,          # gridX, gridY, gridZ
            None, None,          # kernel_metadata, launch_metadata
            None, None,          # launch_enter_hook, launch_exit_hook
            *tensor_args,        # tensor arguments (pointers)
            *constexpr_vals      # constexpr values (n_elements, BLOCK_SIZE, etc.)
        )

# Global instance
_fast_dispatch = FastNPUDispatch()

# =============================================================================
# NPU Triton Kernels
# =============================================================================

@triton.jit
def swiglu_kernel(
    GATE, UP, OUT, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """SwiGLU(gate, up) = SiLU(gate) * up = gate * sigmoid(gate) * up"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    gate = tl.load(GATE + offsets[:])
    up = tl.load(UP + offsets[:])
    gate_f32 = gate.to(tl.float32)
    silu_gate = (gate_f32 * tl.sigmoid(gate_f32)).to(gate.dtype)
    tl.store(OUT + offsets[:], silu_gate * up)


@triton.jit
def elementwise_mul_kernel(
    X, W, OUT, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """out = x * w (element-wise)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets[:])
    w = tl.load(W + offsets[:])
    tl.store(OUT + offsets[:], x * w)


# =============================================================================
# NPU Dispatch Wrappers
# =============================================================================

def set_transform(name):
    os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = os.path.join(SCRIPT_DIR, name)

# Set transform once at import time — only needed during compilation
set_transform("transform_binary_aie2p.mlir")

def npu_swiglu(gate_flat, up_flat):
    """Dispatch SwiGLU on NPU. Auto-captures modules for each unique N value.
    First dispatch per N goes through slow Triton JIT path (~218ms).
    All subsequent dispatches with same N use fast direct C extension (~0.3ms)."""
    N = gate_flat.numel()
    if N in _fast_dispatch._by_n:
        # Fast path: direct C extension call
        out = torch.empty_like(gate_flat)
        _fast_dispatch.run(N, gate_flat, up_flat, out)
        return out
    # Slow path: full Triton JIT (first time for this N value)
    out = torch.empty_like(gate_flat)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    swiglu_kernel[grid](gate_flat, up_flat, out, N, BLOCK_SIZE=1024)
    # Auto-capture for future calls with same N
    _fast_dispatch.auto_capture(N, block_size=1024)
    return out


def npu_elementwise_mul(x_flat, w_flat):
    """Dispatch element-wise multiply on NPU. Auto-captures like npu_swiglu."""
    N = x_flat.numel()
    if N in _fast_dispatch._by_n:
        out = torch.empty_like(x_flat)
        _fast_dispatch.run(N, x_flat, w_flat, out)
        return out
    # Slow path: full Triton JIT
    out = torch.empty_like(x_flat)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    elementwise_mul_kernel[grid](x_flat, w_flat, out, N, BLOCK_SIZE=1024)
    _fast_dispatch.auto_capture(N, block_size=1024)
    return out


# =============================================================================
# Model Monkey-Patching
# =============================================================================

# Global counter for NPU dispatches
npu_dispatch_count = 0


def fuse_gate_up_weights(model):
    """Fuse gate_proj and up_proj into a single weight matrix for faster CPU matmul.
    Instead of two separate matmuls: gate = x @ W_gate, up = x @ W_up
    We do one fused matmul: gate_up = x @ [W_gate | W_up] then split.
    This halves the CPU matmul call overhead for the MLP."""
    import torch.nn as nn
    fused_count = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        gate_w = mlp.gate_proj.weight  # [8192, 2048]
        up_w = mlp.up_proj.weight      # [8192, 2048]
        # Fuse into [16384, 2048] — one matmul produces both gate and up
        fused_w = torch.cat([gate_w.data, up_w.data], dim=0)  # [16384, 2048]
        fused_proj = nn.Linear(gate_w.shape[1], gate_w.shape[0] * 2, bias=False,
                               dtype=gate_w.dtype, device=gate_w.device)
        fused_proj.weight = nn.Parameter(fused_w, requires_grad=False)
        mlp.gate_up_proj = fused_proj
        # Free the individual projections to save memory
        del mlp.gate_proj, mlp.up_proj
        fused_count += 1
    return fused_count


def patched_mlp_forward(self, hidden_states):
    """
    Replaces LlamaMLP.forward:
      Original: down_proj(act_fn(gate_proj(x)) * up_proj(x))
      Patched:  down_proj(npu_swiglu(fused_gate_up_proj(x)))

    Fused gate+up into single matmul, NPU handles SwiGLU activation.
    """
    global npu_dispatch_count
    # CPU: fused gate+up projection (single matmul instead of two)
    intermediate_size = self.down_proj.in_features  # 8192
    gate_up = self.gate_up_proj(hidden_states)
    gate = gate_up[..., :intermediate_size]
    up = gate_up[..., intermediate_size:]
    # NPU: fused SwiGLU activation
    shape = gate.shape
    activated = npu_swiglu(
        gate.contiguous().view(-1),
        up.contiguous().view(-1),
    ).view(shape)
    npu_dispatch_count += 1
    # CPU: down projection (matmul)
    return self.down_proj(activated)


def patched_rmsnorm_forward(self, hidden_states):
    """
    Replaces LlamaRMSNorm.forward:
      Original: weight * (x * rsqrt(mean(x²) + eps))
      Patched:  npu_mul(weight, x * rsqrt(mean(x²) + eps))

    CPU handles the reduction (mean/rsqrt), NPU handles the broadcast multiply.
    """
    global npu_dispatch_count
    # CPU: variance computation (reduction — very fast, not worth offloading)
    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    normed = (hidden_states * torch.rsqrt(variance + self.variance_epsilon)).to(input_dtype)
    # NPU: element-wise weight scaling
    shape = normed.shape
    w_expanded = self.weight.expand(shape).contiguous()
    out = npu_elementwise_mul(
        normed.contiguous().view(-1),
        w_expanded.view(-1),
    ).view(shape)
    npu_dispatch_count += 1
    return out


def patch_model(model, offload_swiglu=True, offload_rmsnorm=False):
    """Apply NPU patches to model layers. Returns estimated NPU dispatches per token."""
    dispatches_per_token = 0
    num_layers = model.config.num_hidden_layers

    if offload_swiglu:
        # Fuse gate+up into single matmul first
        n_fused = fuse_gate_up_weights(model)
        print(f"  Fused gate+up projections in {n_fused} layers (2 matmuls -> 1)")
        print(f"  Patching {num_layers} MLP layers -> NPU SwiGLU")
        for layer in model.model.layers:
            layer.mlp.forward = types.MethodType(patched_mlp_forward, layer.mlp)
        dispatches_per_token += num_layers

    if offload_rmsnorm:
        total_norms = num_layers * 2 + 1  # pre-attn + pre-ffn per layer + final
        print(f"  Patching {total_norms} RMSNorm layers -> NPU weight scaling")
        for layer in model.model.layers:
            layer.input_layernorm.forward = types.MethodType(
                patched_rmsnorm_forward, layer.input_layernorm
            )
            layer.post_attention_layernorm.forward = types.MethodType(
                patched_rmsnorm_forward, layer.post_attention_layernorm
            )
        model.model.norm.forward = types.MethodType(
            patched_rmsnorm_forward, model.model.norm
        )
        dispatches_per_token += total_norms

    return dispatches_per_token


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM inference with AMD Ryzen AI NPU offloading")
    parser.add_argument("--prompt", "-p", type=str, default="Explain what an NPU is in one sentence:",
                        help="Input prompt for text generation")
    parser.add_argument("--max-tokens", "-n", type=int, default=20,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--model", "-m", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--no-swiglu", action="store_true", help="Disable NPU SwiGLU offloading")
    parser.add_argument("--rmsnorm", action="store_true", help="Enable NPU RMSNorm offloading (adds dispatch overhead)")
    args = parser.parse_args()

    MODEL_ID = args.model
    PROMPT = args.prompt
    MAX_NEW_TOKENS = args.max_tokens
    OFFLOAD_SWIGLU = not args.no_swiglu
    OFFLOAD_RMSNORM = args.rmsnorm

    print("=" * 70)
    print("  Real LLM Inference with AMD Ryzen AI NPU Offloading")
    print("  Model: SmolLM2-1.7B-Instruct (1.7B params, bf16)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[1/4] Loading {MODEL_ID}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    load_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Parameters: {n_params:.2f}B")
    print(f"  Hidden: {model.config.hidden_size}")
    print(f"  FFN intermediate: {model.config.intermediate_size}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Vocab: {model.config.vocab_size}")

    # Verify dimensions are multiples of BLOCK_SIZE
    assert model.config.hidden_size % 1024 == 0, \
        f"hidden_size {model.config.hidden_size} not multiple of 1024"
    assert model.config.intermediate_size % 1024 == 0, \
        f"intermediate_size {model.config.intermediate_size} not multiple of 1024"

    # ------------------------------------------------------------------
    # 2. Activate NPU backend & patch model
    # ------------------------------------------------------------------
    print(f"\n[2/4] Configuring NPU backend...")
    benchmark.select_npu_backend()

    dispatches_per_token = patch_model(
        model,
        offload_swiglu=OFFLOAD_SWIGLU,
        offload_rmsnorm=OFFLOAD_RMSNORM,
    )
    print(f"  NPU dispatches per decode token: {dispatches_per_token}")

    # ------------------------------------------------------------------
    # 3. Warm up NPU kernels (compiles xclbin for both prefill & decode sizes)
    # ------------------------------------------------------------------
    print(f"\n[3/4] Warming up NPU kernels (JIT compile)...")
    t0 = time.perf_counter()
    # Warmup with the ACTUAL prompt to pre-compile both prefill and decode kernels
    messages = [{"role": "user", "content": PROMPT}]
    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    warmup_inputs = tokenizer(chat_input, return_tensors="pt")
    with torch.no_grad():
        _ = model.generate(warmup_inputs["input_ids"], max_new_tokens=2, do_sample=False)
    warmup_time = time.perf_counter() - t0
    print(f"  Warmup done in {warmup_time:.1f}s (compiled {npu_dispatch_count} NPU dispatches)")
    cached_sizes = sorted(_fast_dispatch._by_n.keys())
    print(f"  Auto-captured {len(cached_sizes)} kernel variants: N={cached_sizes}")
    print(f"  Fast dispatch enabled: bypassing Triton JIT for all cached N values")

    # ------------------------------------------------------------------
    # 4. Generate text
    # ------------------------------------------------------------------
    print(f"\n[4/4] Generating text...")
    print(f"  Prompt: \"{PROMPT}\"")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print()

    # Reuse the chat-formatted input from warmup
    inputs = warmup_inputs
    input_len = inputs["input_ids"].shape[1]

    npu_dispatch_count = 0  # reset counter
    token_times = []

    # Manual token-by-token generation for per-token timing
    generated_ids = inputs["input_ids"].clone()
    past_key_values = None

    with torch.no_grad():
        for i in range(MAX_NEW_TOKENS):
            t_tok = time.perf_counter()

            if past_key_values is None:
                # Prefill: process entire prompt
                outputs = model(
                    input_ids=generated_ids,
                    use_cache=True,
                )
            else:
                # Decode: process only the last token
                outputs = model(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            dt = time.perf_counter() - t_tok
            token_times.append(dt)

            # Decode and print token
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(f"  Token {i+1:2d}: \"{token_text}\"  ({dt:.2f}s)", flush=True)

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    full_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    n_tokens = len(token_times)
    total_time = sum(token_times)

    # Prefill is token 0, decode is token 1+
    prefill_time = token_times[0] if token_times else 0
    decode_times = token_times[1:] if len(token_times) > 1 else []
    avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0

    print(f"\n{'-' * 70}")
    print(f"  Generated: \"{full_text}\"")
    print(f"{'-' * 70}")
    print(f"  Tokens generated:     {n_tokens}")
    print(f"  Total time:           {total_time:.1f}s")
    print(f"  Prefill (token 1):    {prefill_time:.2f}s")
    if decode_times:
        print(f"  Avg decode time:      {avg_decode:.2f}s/token")
        print(f"  Decode throughput:    {1/avg_decode:.3f} tokens/s")
    print(f"  NPU dispatches:       {npu_dispatch_count}")
    print(f"  NPU dispatches/token: {npu_dispatch_count // max(n_tokens, 1)}")

    # NPU utilization analysis
    npu_time_per_token = dispatches_per_token * 0.00026  # ~260us per dispatch from C++ profiling
    npu_util = npu_time_per_token / avg_decode * 100 if avg_decode > 0 else 0
    print(f"\n  NPU Utilization Analysis:")
    print(f"    NPU compute/token:  {npu_time_per_token*1000:.1f}ms ({dispatches_per_token} x ~260us)")
    print(f"    Total time/token:   {avg_decode*1000:.0f}ms")
    print(f"    NPU utilization:    ~{npu_util:.0f}%")
    print(f"    CPU matmul time:    ~{(avg_decode - npu_time_per_token)*1000:.0f}ms (gate+up fused + down + attention)")
    print(f"\n  The NPU finishes in {npu_time_per_token*1000:.1f}ms but waits {(avg_decode-npu_time_per_token)*1000:.0f}ms")
    print(f"  for CPU matmuls. To increase NPU utilization:")
    print(f"    - INT8 quantized matmul on NPU (weights shrink 2x, NPU has native INT8)")
    print(f"    - Batched inference (multiple sequences = larger NPU workloads)")
    print(f"    - Speculative decoding with a smaller draft model")
