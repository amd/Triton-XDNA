# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
GPT-2 model wired with Triton kernels for GPU, NPU, and hetero inference.

Config-driven across all four GPT-2 sizes (small/medium/large/xl) via
GPT2_CONFIGS — the variants share architecture, vocab, context length, and
head_dim (64), so the same kernels and transform scripts serve every size.
Weights loaded from HuggingFace state_dict.

Two hetero modes route operators across iGPU and NPU:
  "hetero": consistent NPU/GPU split for both prefill and decode
  - iGPU: Attention (QKV proj, Q@K^T, softmax, attn@V, output proj)
  - NPU:  ln1, ln2, ln_f, MLP (up-proj, GELU, down-proj), residual add
  "hetero-fast": GPU-only decode for lower TPOT latency
  - Prefill: same split as "hetero"
  - Decode:  ALL ops on iGPU (NPU dispatch overhead dominates tiny tensors)

LayerNorm runs on NPU in hetero modes because the NPU kernel computes and
outputs in float32, while the GPU kernel truncates to bf16.  This float32
precision through layernorm is critical: bf16 layernorm output compounds
into significant logit drift over 12 transformer layers.
"""

import os
import logging
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
import math

from kernels import triton_linear, triton_bmm, triton_softmax, triton_layernorm, triton_gelu, triton_add, triton_fused_attention

logger = logging.getLogger(__name__)


class OpTimer:
    """Lightweight per-op wall-clock timer. Zero overhead when disabled."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.records = []  # list of (op_name, duration_ms)

    def reset(self):
        self.records.clear()

    @contextmanager
    def track(self, op_name):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.records.append((op_name, (time.perf_counter() - t0) * 1000))

    def summary(self):
        """Aggregate by op name: total_ms, count, avg_ms. Sorted by total descending."""
        agg = defaultdict(lambda: [0.0, 0])
        for op, ms in self.records:
            agg[op][0] += ms
            agg[op][1] += 1
        rows = []
        for op, (total, count) in sorted(agg.items(), key=lambda x: -x[1][0]):
            rows.append((op, total, count, total / count))
        return rows

    def total_ms(self):
        return sum(ms for _, ms in self.records)

# GPT-2 variant configs. All variants share the same architecture, vocab,
# context length, and head_dim (64) — only depth/width differ — so the same
# kernels and transform scripts serve every size.
GPT2_CONFIGS = {
    "gpt2":        {"hf_name": "gpt2",        "n_layer": 12, "n_head": 12, "n_embd": 768},
    "gpt2-medium": {"hf_name": "gpt2-medium", "n_layer": 24, "n_head": 16, "n_embd": 1024},
    "gpt2-large":  {"hf_name": "gpt2-large",  "n_layer": 36, "n_head": 20, "n_embd": 1280},
    "gpt2-xl":     {"hf_name": "gpt2-xl",     "n_layer": 48, "n_head": 25, "n_embd": 1600},
}

# Shared across all GPT-2 variants
VOCAB_SIZE = 50257
MAX_SEQ_LEN = 1024
LN_EPS = 1e-5
EOS_TOKEN = 50256  # <|endoftext|>

# Default config
GPT2_CONFIG = GPT2_CONFIGS["gpt2"]

# Default hetero routing policy: which device runs each operator
HETERO_ROUTING = {
    "layernorm":  "npu",
    "qkv_linear": "gpu",
    "attn_proj":  "gpu",
    "softmax":    "gpu",
    "mlp_fc":     "npu",
    "gelu":       "npu",
    "mlp_proj":   "npu",
    "add":        "npu",
}


class GPT2Model:
    """
    GPT-2 inference model using Triton kernels (config-driven; any GPT-2 size).

    Loads weights from a HuggingFace GPT-2 state_dict and implements
    the forward pass using Triton matmul, softmax, layernorm, GELU,
    and elementwise add kernels.

    Supports four backends:
      - "gpu":         All ops on iGPU via ROCm/Triton
      - "npu":         All ops on NPU via MLIR-AIR/AIE
      - "hetero":      Attention on iGPU, MLP/LN/add on NPU (both prefill & decode)
      - "hetero-fast": Same as hetero for prefill; all-GPU for decode (lower TPOT)
    """

    def __init__(self, state_dict, backend="gpu", config=None):
        """
        Args:
            state_dict: HuggingFace GPT-2 state_dict (from model.state_dict())
            backend: "gpu", "npu", or "hetero"
            config: variant config dict (see GPT2_CONFIGS); defaults to gpt2-small
        """
        self.cfg = config or GPT2_CONFIG
        self.n_layer = self.cfg["n_layer"]
        self.n_head = self.cfg["n_head"]
        self.n_embd = self.cfg["n_embd"]
        self.head_dim = self.n_embd // self.n_head
        self.mlp_dim = 4 * self.n_embd

        self.backend = backend
        self._is_hetero = backend in ("hetero", "hetero-fast")
        self.op_backend = HETERO_ROUTING if self._is_hetero else None
        self.timer = OpTimer(enabled=False)
        self._load_weights(state_dict)

        # Place weights on the device where they'll be consumed
        if backend == "gpu":
            self._move_weights_to_gpu()
        elif backend == "hetero-fast":
            self._place_weights_fast()
        elif backend == "hetero":
            self._place_weights()

        # Cache wte on GPU for the LM head matmul (768x50257).
        # On CPU this takes ~20ms; on GPU <1ms.
        if backend in ("npu", "hetero", "hetero-fast"):
            self._wte_lm_head = self.wte.to(device="cuda", dtype=torch.float32)
        else:
            self._wte_lm_head = None

        # Resolve transform scripts relative to this file's directory
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.matmul_script = os.path.join(self.script_dir, "transform_matmul_aie2p.mlir")
        self.elem_script = os.path.join(self.script_dir, "transform_elementwise_aie2p.mlir")
        self.add_script = os.path.join(self.script_dir, "transform_add_aie2p.mlir")
        self.softmax_script = os.path.join(self.script_dir, "transform_softmax_aie2p.mlir")
        self.layernorm_script = os.path.join(self.script_dir, "transform_layernorm_aie2p.mlir")

    def _load_weights(self, sd):
        """Map HuggingFace GPT-2 weight names to internal parameters."""
        dtype = torch.bfloat16

        # Embeddings
        self.wte = sd["transformer.wte.weight"].to(dtype)  # (50257, 768)
        self.wpe = sd["transformer.wpe.weight"].to(dtype)  # (1024, 768)

        # Per-layer weights
        self.layers = []
        for i in range(self.n_layer):
            prefix = f"transformer.h.{i}"
            layer = {
                # Pre-attention layernorm
                "ln1_weight": sd[f"{prefix}.ln_1.weight"].to(dtype),
                "ln1_bias": sd[f"{prefix}.ln_1.bias"].to(dtype),
                # QKV projection (combined)
                "qkv_weight": sd[f"{prefix}.attn.c_attn.weight"].to(dtype),  # (768, 2304)
                "qkv_bias": sd[f"{prefix}.attn.c_attn.bias"].to(dtype),  # (2304,)
                # Attention output projection
                "attn_proj_weight": sd[f"{prefix}.attn.c_proj.weight"].to(dtype),  # (768, 768)
                "attn_proj_bias": sd[f"{prefix}.attn.c_proj.bias"].to(dtype),  # (768,)
                # Pre-MLP layernorm
                "ln2_weight": sd[f"{prefix}.ln_2.weight"].to(dtype),
                "ln2_bias": sd[f"{prefix}.ln_2.bias"].to(dtype),
                # MLP
                "mlp_fc_weight": sd[f"{prefix}.mlp.c_fc.weight"].to(dtype),  # (768, 3072)
                "mlp_fc_bias": sd[f"{prefix}.mlp.c_fc.bias"].to(dtype),  # (3072,)
                "mlp_proj_weight": sd[f"{prefix}.mlp.c_proj.weight"].to(dtype),  # (3072, 768)
                "mlp_proj_bias": sd[f"{prefix}.mlp.c_proj.bias"].to(dtype),  # (768,)
            }
            self.layers.append(layer)

        # Final layernorm
        self.ln_f_weight = sd["transformer.ln_f.weight"].to(dtype)
        self.ln_f_bias = sd["transformer.ln_f.bias"].to(dtype)

    def _move_weights_to_gpu(self):
        """Move all model weights to GPU once to avoid per-kernel transfers."""
        device = "cuda"
        self.wte = self.wte.to(device)
        self.wpe = self.wpe.to(device)
        for layer in self.layers:
            for k, v in layer.items():
                layer[k] = v.to(device)
        self.ln_f_weight = self.ln_f_weight.to(device)
        self.ln_f_bias = self.ln_f_bias.to(device)

    def _place_weights(self):
        """Place weights for consistent hetero mode.

        Attention weights → GPU; LN/MLP/add weights stay on CPU for NPU.
        Same routing for both prefill and decode.
        """
        gpu_keys = {"qkv_weight", "qkv_bias", "attn_proj_weight", "attn_proj_bias"}
        for layer in self.layers:
            for k, v in layer.items():
                if k in gpu_keys:
                    layer[k] = v.to("cuda")
        # ln_f weights stay on CPU for NPU layernorm (float32 precision)

    def _place_weights_fast(self):
        """Place weights for hetero-fast mode: all on GPU, CPU copies for NPU prefill.

        During decode (S=1), all ops run on GPU to avoid NPU dispatch overhead
        (~0.5-1ms per launch × 72 dispatches = ~36ms wasted). All weights must
        be GPU-resident for this fast path.

        During prefill (S>1), MLP/ln2 ops run on NPU where larger tensors benefit
        from AIE parallelism. NPU kernels expect CPU-resident weights, so we keep
        CPU copies of the NPU-routed weights.
        """
        # CPU copies of NPU-routed weights (must be created before moving to GPU)
        npu_keys = {"ln1_weight", "ln1_bias",
                    "ln2_weight", "ln2_bias",
                    "mlp_fc_weight", "mlp_fc_bias",
                    "mlp_proj_weight", "mlp_proj_bias"}
        self._cpu_layers = []
        for layer in self.layers:
            cpu_layer = {}
            for k in npu_keys:
                cpu_layer[k] = layer[k].clone()  # already on CPU
            self._cpu_layers.append(cpu_layer)

        # CPU copies of ln_f for NPU prefill
        self._cpu_ln_f_weight = self.ln_f_weight.clone()
        self._cpu_ln_f_bias = self.ln_f_bias.clone()

        # All weights to GPU (for decode fast path + attention)
        for layer in self.layers:
            for k, v in layer.items():
                layer[k] = v.to("cuda")
        self.ln_f_weight = self.ln_f_weight.to("cuda")
        self.ln_f_bias = self.ln_f_bias.to("cuda")

    def _to_gpu(self, x):
        """Move tensor to CUDA if not already there."""
        if x.device.type != "cuda":
            return x.to("cuda")
        return x

    def _to_cpu(self, x):
        """Move tensor to CPU if not already there."""
        if x.device.type != "cpu":
            return x.cpu()
        return x

    def _linear(self, x, weight, bias=None, backend=None):
        """
        Linear layer: x @ weight^T + bias.
        Note: GPT-2 HF stores c_attn/c_fc weights as (in_features, out_features),
        which is transposed from nn.Linear convention. So we pass weight.t() which
        gives (out_features, in_features) — the shape triton_linear expects.
        """
        be = backend or self.backend
        try:
            return triton_linear(
                x, weight.t(), bias=bias,
                backend=be,
                transform_script=self.matmul_script if be == "npu" else None,
            )
        except Exception as e:
            logger.warning(f"Triton linear failed ({e}), falling back to PyTorch")
            x_f32 = x.to(torch.float32)
            w_f32 = weight.to(torch.float32)
            out = x_f32 @ w_f32
            if bias is not None:
                out = out + bias.to(torch.float32)
            return out.to(torch.bfloat16) if be == "gpu" else out

    def _layernorm(self, x, weight, bias, backend=None):
        """LayerNorm with learnable parameters."""
        be = backend or self.backend
        try:
            return triton_layernorm(
                x, weight, bias, eps=LN_EPS,
                backend=be,
                transform_script=self.layernorm_script if be == "npu" else None,
            )
        except Exception as e:
            logger.warning(f"Triton layernorm failed ({e}), falling back to PyTorch")
            out = torch.nn.functional.layer_norm(
                x.to(torch.float32), (x.shape[-1],),
                weight=weight.to(torch.float32),
                bias=bias.to(torch.float32),
                eps=LN_EPS,
            )
            return out.to(torch.bfloat16) if be == "gpu" else out

    def _gelu(self, x, backend=None):
        """GELU activation (tanh approximation on GPU, sigmoid approx on NPU)."""
        be = backend or self.backend
        try:
            return triton_gelu(
                x, backend=be,
                transform_script=self.elem_script if be == "npu" else None,
            )
        except Exception as e:
            logger.warning(f"Triton GELU failed ({e}), falling back to PyTorch")
            out = torch.nn.functional.gelu(x.to(torch.float32), approximate="tanh")
            return out.to(torch.bfloat16) if be == "gpu" else out

    def _add(self, a, b, backend=None):
        """Residual addition."""
        be = backend or self.backend
        try:
            return triton_add(
                a, b, backend=be,
                transform_script=self.add_script if be == "npu" else None,
            )
        except Exception as e:
            logger.warning(f"Triton add failed ({e}), falling back to PyTorch")
            out = a.to(torch.float32) + b.to(torch.float32)
            return out.to(torch.bfloat16) if be == "gpu" else out

    def _softmax(self, x, causal_mask=None, backend=None):
        """Softmax over last dimension with optional causal mask."""
        be = backend or self.backend
        try:
            return triton_softmax(
                x, causal_mask=causal_mask,
                backend=be,
                transform_script=self.softmax_script if be == "npu" else None,
            )
        except Exception as e:
            logger.warning(f"Triton softmax failed ({e}), falling back to PyTorch")
            if causal_mask is not None:
                x = x.masked_fill(~causal_mask, float("-inf"))
            out = torch.softmax(x.to(torch.float32), dim=-1)
            return out.to(torch.bfloat16) if be == "gpu" else out

    def _attention(self, x, layer, kv_cache=None, pos_offset=0):
        """
        Multi-head self-attention with optional KV cache.

        In hetero mode, x arrives on CUDA and all ops run on GPU.

        Args:
            x: (batch, seq_len, 768) — already normed
            layer: dict of layer weights
            kv_cache: tuple (cache_k, cache_v, seq_pos) with pre-allocated buffers
                      and current write position, or None for prefill
            pos_offset: position offset for causal mask (0 for prefill, past_len for decode)

        Returns:
            proj: (batch, seq_len, 768) — attention output (before residual)
            new_kv_cache: tuple (cache_k, cache_v, new_seq_pos)
        """
        B, S, D = x.shape

        # In hetero mode, attention ops are routed to GPU explicitly
        attn_be = self.op_backend["qkv_linear"] if self.op_backend else None
        softmax_be = self.op_backend["softmax"] if self.op_backend else None
        proj_be = self.op_backend["attn_proj"] if self.op_backend else None

        # QKV projection: (B, S, 768) -> (B, S, 2304)
        qkv = self._linear(x, layer["qkv_weight"], layer["qkv_bias"], backend=attn_be)

        # Split into Q, K, V: each (B, S, n_embd)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # Reshape to multi-head: (B, S, n_embd) -> (B, n_head, S, head_dim)
        q = q.reshape(B, S, self.n_head, self.head_dim).transpose(1, 2)
        k_new = k.reshape(B, S, self.n_head, self.head_dim).transpose(1, 2)
        v_new = v.reshape(B, S, self.n_head, self.head_dim).transpose(1, 2)

        # KV cache: write into pre-allocated buffer (no torch.cat)
        if kv_cache is not None:
            cache_k, cache_v, seq_pos = kv_cache
            cache_k[:, :, seq_pos:seq_pos + S, :] = k_new
            cache_v[:, :, seq_pos:seq_pos + S, :] = v_new
            total_len = seq_pos + S
            k_full = cache_k[:, :, :total_len, :]
            v_full = cache_v[:, :, :total_len, :]
            new_kv_cache = (cache_k, cache_v, total_len)
        else:
            k_full = k_new
            v_full = v_new
            total_len = S
            new_kv_cache = None  # Caller will build pre-allocated cache from this

        # Fused attention: Q@K^T, scaling, causal mask, softmax, attn@V in one kernel
        scale = 1.0 / math.sqrt(self.head_dim)
        if self.backend in ("gpu", "hetero", "hetero-fast"):
            q_3d = q.reshape(B * self.n_head, S, self.head_dim).contiguous()
            k_3d = k_full.reshape(B * self.n_head, total_len, self.head_dim).contiguous()
            v_3d = v_full.reshape(B * self.n_head, total_len, self.head_dim).contiguous()
            is_causal = S > 1  # Decode (S=1): every past position visible, no causal mask
            attn_output = triton_fused_attention(
                q_3d, k_3d, v_3d,
                scale=scale,
                causal=is_causal,
                pos_offset=pos_offset,
            ).reshape(B, self.n_head, S, self.head_dim)
        else:
            # NPU fallback: separate matmul + softmax path.
            # In npu backend _linear/_softmax return float32, but the KV cache is
            # stored bf16, so k_full/v_full come back bf16 once the cache is active.
            # Compute the reference attention in float32 to keep dtypes consistent.
            q = q.to(torch.float32)
            k_full = k_full.to(torch.float32)
            v_full = v_full.to(torch.float32)
            attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
            if S == 1:
                attn_flat = attn_scores.reshape(-1, total_len)
                attn_weights_flat = self._softmax(attn_flat, backend=softmax_be)
                attn_weights = attn_weights_flat.reshape(B, self.n_head, S, total_len)
            else:
                rows = torch.arange(S, device=x.device).unsqueeze(1) + pos_offset
                cols = torch.arange(total_len, device=x.device).unsqueeze(0)
                causal_mask = (cols <= rows).unsqueeze(0).unsqueeze(0)
                attn_flat = attn_scores.reshape(-1, total_len)
                causal_flat = causal_mask.expand(B, self.n_head, S, total_len).reshape(-1, total_len)
                attn_weights_flat = self._softmax(attn_flat, causal_mask=causal_flat, backend=softmax_be)
                attn_weights = attn_weights_flat.reshape(B, self.n_head, S, total_len)
            attn_output = torch.matmul(attn_weights, v_full)

        # Reshape back: (B, n_head, S, head_dim) -> (B, S, n_embd)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, self.n_embd)

        # Output projection: (B, S, 768) -> (B, S, 768)
        proj = self._linear(attn_output, layer["attn_proj_weight"], layer["attn_proj_bias"], backend=proj_be)

        return proj, new_kv_cache

    def forward(self, input_ids, kv_caches=None, pos_offset=0):
        """
        Forward pass with optional KV cache for autoregressive generation.

        Args:
            input_ids: (batch, seq_len) integer token IDs
            kv_caches: list of 12 (cache_k, cache_v, seq_pos) tuples, or None for prefill
            pos_offset: position offset for embeddings (0 for prefill, past_len for decode)

        Returns:
            logits: (batch, seq_len, vocab_size) float32
            new_kv_caches: list of 12 (cache_k, cache_v, seq_pos) tuples for next step
        """
        B, S = input_ids.shape
        hetero = self._is_hetero
        # All-GPU fast path for decode (hetero-fast only): when S=1, NPU dispatch
        # overhead (~0.5-1ms × 72 calls) dominates tiny-tensor compute. Route
        # everything to GPU using GPU-resident weights instead.
        decode_gpu = self.backend == "hetero-fast" and S == 1

        # Move input_ids to GPU for embedding lookup (weights already on GPU)
        if self.backend == "gpu" and input_ids.device.type != "cuda":
            input_ids = input_ids.to("cuda")

        # Token + position embeddings — stay in bf16 to avoid per-layer casts
        # In hetero mode, embeddings are on CPU (wte/wpe stay on CPU)
        positions = torch.arange(pos_offset, pos_offset + S, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(B, -1)
        x = self.wte[input_ids] + self.wpe[positions]  # bf16 + bf16 = bf16

        # Transformer blocks
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            logger.debug(f"Layer {i}/{self.n_layer}")

            if decode_gpu:
                # --- ALL-GPU DECODE PATH ---
                # x stays on CUDA for all 12 layers; zero device transfers.
                # Uses GPU-resident weights from self.layers[i] directly.
                if x.device.type != "cuda":
                    x = self._to_gpu(x)

                with self.timer.track("ln1"):
                    x_norm = self._layernorm(x, layer["ln1_weight"], layer["ln1_bias"], backend="gpu")

                layer_cache = kv_caches[i] if kv_caches else None
                with self.timer.track("attention"):
                    attn_out, new_cache = self._attention(x_norm, layer, kv_cache=layer_cache, pos_offset=pos_offset)
                new_kv_caches.append(new_cache)

                with self.timer.track("add1"):
                    x = self._add(x, attn_out, backend="gpu")
                with self.timer.track("ln2"):
                    x_norm = self._layernorm(x, layer["ln2_weight"], layer["ln2_bias"], backend="gpu")
                with self.timer.track("mlp_fc"):
                    h = self._linear(x_norm, layer["mlp_fc_weight"], layer["mlp_fc_bias"], backend="gpu")
                with self.timer.track("gelu"):
                    h = self._gelu(h, backend="gpu")
                with self.timer.track("mlp_proj"):
                    mlp_out = self._linear(h, layer["mlp_proj_weight"], layer["mlp_proj_bias"], backend="gpu")
                with self.timer.track("add2"):
                    x = self._add(x, mlp_out, backend="gpu")

            elif hetero:
                # --- HETERO PATH (prefill for hetero-fast, all steps for hetero) ---
                # GPU for attention; NPU for LN/MLP/add.
                # hetero-fast: weights are on GPU, use _cpu_layers for NPU ops
                # hetero: LN/MLP weights are already on CPU in layer dict
                npu_w = self._cpu_layers[i] if hasattr(self, '_cpu_layers') else layer
                ln1_be = self.op_backend["layernorm"]
                with self.timer.track("ln1"):
                    x_norm = self._layernorm(x, npu_w["ln1_weight"], npu_w["ln1_bias"], backend=ln1_be)
                with self.timer.track("to_gpu"):
                    x_norm = self._to_gpu(x_norm)

                layer_cache = kv_caches[i] if kv_caches else None
                with self.timer.track("attention"):
                    attn_out, new_cache = self._attention(x_norm, layer, kv_cache=layer_cache, pos_offset=pos_offset)
                new_kv_caches.append(new_cache)

                with self.timer.track("to_cpu"):
                    attn_out = self._to_cpu(attn_out)

                add_be = self.op_backend["add"]
                with self.timer.track("add1"):
                    x = self._add(x, attn_out, backend=add_be)

                # NPU ops: use CPU-resident weights
                npu_w = self._cpu_layers[i] if hasattr(self, '_cpu_layers') else layer
                ln2_be = self.op_backend["layernorm"]
                with self.timer.track("ln2"):
                    x_norm = self._layernorm(x, npu_w["ln2_weight"], npu_w["ln2_bias"], backend=ln2_be)

                mlp_fc_be = self.op_backend["mlp_fc"]
                gelu_be = self.op_backend["gelu"]
                mlp_proj_be = self.op_backend["mlp_proj"]
                with self.timer.track("mlp_fc"):
                    h = self._linear(x_norm, npu_w["mlp_fc_weight"], npu_w["mlp_fc_bias"], backend=mlp_fc_be)
                with self.timer.track("gelu"):
                    h = self._gelu(h, backend=gelu_be)
                with self.timer.track("mlp_proj"):
                    mlp_out = self._linear(h, npu_w["mlp_proj_weight"], npu_w["mlp_proj_bias"], backend=mlp_proj_be)

                with self.timer.track("add2"):
                    x = self._add(x, mlp_out, backend=add_be)

            else:
                # --- SINGLE-BACKEND PATH (gpu or npu) ---
                with self.timer.track("ln1"):
                    x_norm = self._layernorm(x, layer["ln1_weight"], layer["ln1_bias"])

                layer_cache = kv_caches[i] if kv_caches else None
                with self.timer.track("attention"):
                    attn_out, new_cache = self._attention(x_norm, layer, kv_cache=layer_cache, pos_offset=pos_offset)
                new_kv_caches.append(new_cache)

                with self.timer.track("add1"):
                    x = self._add(x, attn_out)
                with self.timer.track("ln2"):
                    x_norm = self._layernorm(x, layer["ln2_weight"], layer["ln2_bias"])
                with self.timer.track("mlp_fc"):
                    h = self._linear(x_norm, layer["mlp_fc_weight"], layer["mlp_fc_bias"])
                with self.timer.track("gelu"):
                    h = self._gelu(h)
                with self.timer.track("mlp_proj"):
                    mlp_out = self._linear(h, layer["mlp_proj_weight"], layer["mlp_proj_bias"])
                with self.timer.track("add2"):
                    x = self._add(x, mlp_out)

        # Final LayerNorm
        if hetero and not decode_gpu:
            with self.timer.track("ln_f"):
                ln_f_be = self.op_backend["layernorm"]
                ln_f_w = self._cpu_ln_f_weight if hasattr(self, '_cpu_ln_f_weight') else self.ln_f_weight
                ln_f_b = self._cpu_ln_f_bias if hasattr(self, '_cpu_ln_f_bias') else self.ln_f_bias
                x = self._layernorm(x, ln_f_w, ln_f_b, backend=ln_f_be)
        elif hetero:
            with self.timer.track("ln_f"):
                x = self._layernorm(x, self.ln_f_weight, self.ln_f_bias, backend="gpu")
        else:
            with self.timer.track("ln_f"):
                x = self._layernorm(x, self.ln_f_weight, self.ln_f_bias)

        # Language model head: x @ wte^T (tied weights)
        # (B, S, 768) @ (768, 50257) -> (B, S, 50257)
        with self.timer.track("lm_head"):
            if self._wte_lm_head is not None:
                # NPU/hetero: run on GPU (~0.5ms vs ~20ms on CPU)
                logits = (x.to(device="cuda", dtype=torch.float32) @ self._wte_lm_head.t()).cpu()
            else:
                # GPU: x and wte already on CUDA
                logits = x.to(torch.float32) @ self.wte.to(torch.float32).t()
                if logits.device.type == "cuda":
                    logits = logits.cpu()

        return logits, new_kv_caches

    def _allocate_kv_caches(self, batch_size, max_seq_len, device, dtype=torch.bfloat16):
        """Pre-allocate KV cache buffers for all layers."""
        caches = []
        for _ in range(self.n_layer):
            cache_k = torch.zeros(batch_size, self.n_head, max_seq_len, self.head_dim, dtype=dtype, device=device)
            cache_v = torch.zeros(batch_size, self.n_head, max_seq_len, self.head_dim, dtype=dtype, device=device)
            caches.append((cache_k, cache_v, 0))
        return caches

    def generate(self, input_ids, max_new_tokens=20, progress_callback=None):
        """
        Autoregressive generation with pre-allocated KV cache.

        Args:
            input_ids: (1, prompt_len) token IDs
            max_new_tokens: number of tokens to generate
            progress_callback: optional fn(tokens_done, total) called after each
                generated token, for progress reporting

        Returns:
            generated_ids: list of generated token IDs
            timing: dict with prefill_ms, decode_times_ms (list), total_ms
        """
        B, prompt_len = input_ids.shape
        total_seq_len = prompt_len + max_new_tokens
        device = input_ids.device
        if self.backend in ("gpu", "hetero", "hetero-fast"):
            device = "cuda"

        generated_ids = []
        timing = {"prefill_ms": 0, "decode_times_ms": []}

        # Pre-allocate KV caches for all layers
        kv_caches = self._allocate_kv_caches(B, total_seq_len, device)

        # Prefill: process full prompt
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, kv_caches = self.forward(input_ids, kv_caches=kv_caches)
        t1 = time.perf_counter()
        timing["prefill_ms"] = (t1 - t0) * 1000

        # Greedy: pick last token's argmax
        next_token = torch.argmax(logits[0, -1]).item()
        generated_ids.append(next_token)
        pos_offset = prompt_len
        if progress_callback is not None:
            progress_callback(len(generated_ids), max_new_tokens)

        # Decode loop
        for step in range(max_new_tokens - 1):
            next_input = torch.tensor([[next_token]], dtype=torch.long)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits, kv_caches = self.forward(next_input, kv_caches=kv_caches, pos_offset=pos_offset)
            t1 = time.perf_counter()
            timing["decode_times_ms"].append((t1 - t0) * 1000)

            next_token = torch.argmax(logits[0, -1]).item()
            generated_ids.append(next_token)
            pos_offset += 1
            if progress_callback is not None:
                progress_callback(len(generated_ids), max_new_tokens)

            # Stop on EOS (GPT-2 <|endoftext|> = 50256)
            if next_token == EOS_TOKEN:
                break

        return generated_ids, timing
