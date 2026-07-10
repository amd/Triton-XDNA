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
import numpy as np

from kernels import (
    triton_linear,
    triton_bmm,
    triton_softmax,
    triton_layernorm,
    triton_gelu,
    triton_add,
    triton_fused_attention,
)

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
    "gpt2": {"hf_name": "gpt2", "n_layer": 12, "n_head": 12, "n_embd": 768},
    "gpt2-medium": {
        "hf_name": "gpt2-medium",
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 1024,
    },
    "gpt2-large": {
        "hf_name": "gpt2-large",
        "n_layer": 36,
        "n_head": 20,
        "n_embd": 1280,
    },
    "gpt2-xl": {"hf_name": "gpt2-xl", "n_layer": 48, "n_head": 25, "n_embd": 1600},
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
    "layernorm": "npu",
    "qkv_linear": "gpu",
    "attn_proj": "gpu",
    "softmax": "gpu",
    "mlp_fc": "npu",
    "gelu": "npu",
    "mlp_proj": "npu",
    "add": "npu",
}


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


class _FusedMLP:
    """Fuse mlp_fc(+bias) -> gelu -> mlp_proj -> residual add into ONE load_pdi
    multi-launch ELF.

    Replaces four separate NPU dispatches (each paying the ~147ms hw_context
    rebuild) with a single fused ELF dispatched through one persistent
    hw_context + one xrt.run.

    Per layer (lazily on first use) builds an NPUChain with pre-prepped weights:
      op0 mlp_fc:   C0  = [x|1] @ [W_fc ; b_fc]  (bias folded via augmented-K)
      op1 gelu:     G   = gelu(C0)               (f32 -> bf16)
      op2 mlp_proj: C2  = G @ W_proj             (bf16 -> f32)
      op3 add:      out = C2 + residual          (f32; the block's post-MLP
                                                  residual add, DDR hand-off)
    mlp_proj's bias is added on host post-readback (its A is gelu's output, so
    the augmented-K fold can't apply; it folds into the residual add: the device
    computes C2 + residual and the host adds b_proj, giving x + mlp_out). The
    ln2 that feeds this chain stays a separate NPU dispatch -- folding it on
    device needs a custom normalize kernel + transform script for marginal gain
    (see the deferred Stage B note).

    Shapes (HF stores c_fc/c_proj as (in,out), used as x@W directly):
      W_fc=(D, H), b_fc=(H,)  ; W_proj=(H, D), b_proj=(D,).  D=n_embd, H=mlp_dim.
    K0_pad = next_pow2(D+1); HID_pad = next_pow2(H); both matmuls single-block.
    M is padded to 256 (BLOCK_M). Only valid rows/cols are read back.
    """

    def __init__(
        self, n_embd, mlp_dim, matmul_script, gelu_f32in_script, add_f32_script
    ):
        self.D = n_embd
        self.H = mlp_dim
        self.matmul_script = matmul_script
        self.gelu_script = gelu_f32in_script
        self.add_script = add_f32_script
        self.BM = 128
        self.BN = 256
        self.K0_pad = _next_pow2(self.D + 1)  # +1 bias row
        self.HID_pad = _next_pow2(self.H)
        # One shared chain (one stitched ELF + one hw_context) serves every
        # layer; per-layer weights are swapped in via bo_key at run time. A
        # chain-per-layer would allocate one hw_context per layer and exhaust the
        # NPU (DRM_IOCTL_AMDXDNA_CREATE_HWCTX) on deep models. Mirrors the
        # upstream mlir-air llama pattern (ELF compiled once, per-layer BOs).
        self._chain = None  # shared NPUChain
        self._weights = {}  # layer_idx -> (B0, B2, b_proj_np)
        self._mm = None
        self._gelu = None
        self._build_kernels()
        # Persistent host staging buffers (shape fixed across all calls/layers).
        # Reused every run() to avoid re-allocating/zeroing ~8MB per dispatch.
        # Rows beyond M_real hold stale data but are discarded on readback (the
        # matmul is row-independent and output is sliced to [:M_real]). K-padding
        # columns (D+1:) stay zero (written once here, never touched again).
        from ml_dtypes import bfloat16 as _bf16

        self._A_aug = np.zeros((self.BM, self.K0_pad), dtype=_bf16)
        self._C0 = np.zeros((self.BM, self.HID_pad), dtype=np.float32)
        self._G = np.zeros(self.BM * self.HID_pad, dtype=_bf16)
        self._C2 = np.zeros((self.BM, self.D), dtype=np.float32)
        self._R = np.zeros((self.BM, self.D), dtype=np.float32)
        self._OUT = np.zeros((self.BM, self.D), dtype=np.float32)

    def _build_kernels(self):
        import triton
        import triton.language as tl

        @triton.jit
        def _mm_kernel(
            A,
            B,
            C,
            M: tl.constexpr,
            N: tl.constexpr,
            K: tl.constexpr,
            sam: tl.constexpr,
            sak: tl.constexpr,
            sbk: tl.constexpr,
            sbn: tl.constexpr,
            scm: tl.constexpr,
            scn: tl.constexpr,
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a = tl.load(A + offs_m[:, None] * sam + offs_k[None, :] * sak)
            b = tl.load(B + offs_k[:, None] * sbk + offs_n[None, :] * sbn)
            c = tl.dot(a, b)
            tl.store(C + offs_m[:, None] * scm + offs_n[None, :] * scn, c)

        @triton.jit
        def _gelu_f32in(X, Y, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            xf = tl.load(X + offsets[:])
            k: tl.constexpr = 0.7978845608028654
            z = k * (xf + 0.044715 * xf * xf * xf)
            y = (xf * tl.sigmoid(2.0 * z)).to(tl.bfloat16)
            tl.store(Y + offsets[:], y)

        @triton.jit
        def _add_kernel(A, B, C, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            a = tl.load(A + offsets[:])
            b = tl.load(B + offsets[:])
            tl.store(C + offsets[:], a + b)

        self._mm = _mm_kernel
        self._gelu = _gelu_f32in
        self._add = _add_kernel

    def _prep_weights(self, w_fc, b_fc, w_proj, b_proj):
        """Build padded, bias-folded static weight arrays (numpy bf16)."""
        from ml_dtypes import bfloat16

        D, H, K0_pad, HID_pad = self.D, self.H, self.K0_pad, self.HID_pad
        w_fc = w_fc.to(torch.float32).cpu().numpy()  # (D, H)
        b_fc = b_fc.to(torch.float32).cpu().numpy()  # (H,)
        w_proj = w_proj.to(torch.float32).cpu().numpy()  # (H, D)
        # B0 = [W_fc ; b_fc] padded to (K0_pad, HID_pad)
        B0 = np.zeros((K0_pad, HID_pad), dtype=bfloat16)
        B0[:D, :H] = w_fc.astype(bfloat16)
        B0[D, :H] = b_fc.astype(bfloat16)
        # B2 = W_proj padded to (HID_pad, D)
        B2 = np.zeros((HID_pad, D), dtype=bfloat16)
        B2[:H, :D] = w_proj.astype(bfloat16)
        return B0, B2

    def _get_chain(self):
        """Build the shared MLP chain once (every layer has the same MLP shape).

        A single stitched ELF + one hw_context serves all layers; per-layer
        weights are supplied at run time via bo_key. Warmup tensors are
        shape-representative placeholders (layer-independent).
        """
        if self._chain is not None:
            return self._chain
        from triton.backends.amd_triton_npu.multilaunch import NPUChain

        D, H, K0_pad, HID_pad, BM, BN = (
            self.D,
            self.H,
            self.K0_pad,
            self.HID_pad,
            self.BM,
            self.BN,
        )
        M_pad = self.BM  # single M-block; valid rows sliced on readback

        tAf = torch.zeros((M_pad, K0_pad), dtype=torch.bfloat16)
        tB0 = torch.zeros((K0_pad, HID_pad), dtype=torch.bfloat16)
        tC0 = torch.zeros((M_pad, HID_pad), dtype=torch.float32)
        tXg = torch.zeros(M_pad * HID_pad, dtype=torch.float32)
        tG = torch.zeros(M_pad * HID_pad, dtype=torch.bfloat16)
        tGm = torch.zeros((M_pad, HID_pad), dtype=torch.bfloat16)
        tB2 = torch.zeros((HID_pad, D), dtype=torch.bfloat16)
        tC2 = torch.zeros((M_pad, D), dtype=torch.float32)
        tC2f = torch.zeros(M_pad * D, dtype=torch.float32)
        tR = torch.zeros(M_pad * D, dtype=torch.float32)
        tOut = torch.zeros(M_pad * D, dtype=torch.float32)

        chain = NPUChain("gpt2_mlp")
        chain.add(
            self._mm,
            grid=(M_pad // BM, HID_pad // BN),
            arg_map={0: 0, 1: 1, 2: 2},
            args=(
                tAf,
                tB0,
                tC0,
                M_pad,
                HID_pad,
                K0_pad,
                K0_pad,
                1,
                HID_pad,
                1,
                HID_pad,
                1,
            ),
            constexprs={"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": K0_pad},
            transform_script=self.matmul_script,
        )
        chain.add(
            self._gelu,
            grid=((M_pad * HID_pad) // 1024,),
            arg_map={0: 2, 1: 3},
            args=(tXg, tG, M_pad * HID_pad),
            constexprs={"BLOCK_SIZE": 1024},
            transform_script=self.gelu_script,
        )
        chain.add(
            self._mm,
            grid=(M_pad // BM, D // BN),
            arg_map={0: 3, 1: 4, 2: 5},
            args=(tGm, tB2, tC2, M_pad, D, HID_pad, HID_pad, 1, D, 1, D, 1),
            constexprs={
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": HID_pad,
            },
            transform_script=self.matmul_script,
        )
        # op3 add: out = C2 + residual (fold the post-MLP residual add into the
        # chain so it shares the one hw_context; mlp_proj bias is still applied
        # host-side on readback). C2 (combined idx 5) becomes an intermediate.
        chain.add(
            self._add,
            grid=((M_pad * D) // 1024,),
            arg_map={0: 5, 1: 6, 2: 7},
            args=(tC2f, tR, tOut, M_pad * D),
            constexprs={"BLOCK_SIZE": 1024},
            transform_script=self.add_script,
        )
        self._chain = chain
        return chain

    def close(self):
        """Release the shared chain's XRT context/BOs in order (see
        NPUChain.close / MultiLaunchRunner.unload)."""
        if self._chain is not None:
            self._chain.close()
            self._chain = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _weights_for(self, layer_idx, w_fc, b_fc, w_proj, b_proj):
        """Per-layer prepped weights (cached). Shape is identical across layers;
        only the values differ, so they ride the shared chain as per-layer BOs."""
        if layer_idx not in self._weights:
            B0, B2 = self._prep_weights(w_fc, b_fc, w_proj, b_proj)
            b_proj_np = b_proj.to(torch.float32).cpu().numpy()
            self._weights[layer_idx] = (B0, B2, b_proj_np)
        return self._weights[layer_idx]

    def run(self, layer_idx, x_norm, residual, w_fc, b_fc, w_proj, b_proj):
        """Run the fused MLP + residual add for one layer.

        x_norm: ln2(x) (B,S,D) tensor (CPU); residual: x (B,S,D) the block
        hidden state added back after the MLP. Returns x + mlp(x_norm)
        (B,S,D) torch f32 tensor -- i.e. the post-residual-add result, matching
        `x = add(x, mlp_out)` in the unfused path.
        """
        from ml_dtypes import bfloat16

        D, H, K0_pad, HID_pad = self.D, self.H, self.K0_pad, self.HID_pad
        chain = self._get_chain()
        B0, B2, b_proj_np = self._weights_for(layer_idx, w_fc, b_fc, w_proj, b_proj)

        orig_shape = x_norm.shape
        x2d = x_norm.reshape(-1, D).to(torch.float32).cpu().numpy()  # (M_real, D)
        res2d = residual.reshape(-1, D).to(torch.float32).cpu().numpy()
        M_real = x2d.shape[0]
        M_pad = self.BM
        if M_real > M_pad:
            # Multiple M-blocks not handled by this single-block chain; caller
            # should keep prefill on the unfused path for long sequences.
            raise ValueError(
                f"_FusedMLP only supports M<={M_pad} (got {M_real}); "
                "use the unfused MLP path for longer sequences."
            )
        # A_aug = [x | 1 | 0...]; reuse persistent buffers (only [:M_real] read
        # back, so stale rows beyond M_real are harmless).
        A_aug, C0, G, C2, R, OUT = (
            self._A_aug,
            self._C0,
            self._G,
            self._C2,
            self._R,
            self._OUT,
        )
        A_aug[:M_real, :D] = x2d.astype(bfloat16)
        A_aug[:M_real, D] = bfloat16(1.0)
        R[:M_real, :D] = res2d

        # On-device: OUT = C2 + R (residual). Host: + b_proj broadcast over M,
        # giving x + (mlp_proj_out + b_proj) = x + mlp_out.
        out = chain.run(
            [A_aug, B0, C0, G, B2, C2, R, OUT],
            bo_key=f"gpt2_mlp_L{layer_idx}",
            static_indices={1, 4},
            intermediate_indices={2, 3, 5},
            output_indices={7},
        )
        res = out[7].astype(np.float32)[:M_real, :D] + b_proj_np  # host bias
        return torch.from_numpy(res.copy()).reshape(orig_shape)


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
        self.matmul_script = os.path.join(
            self.script_dir, "transform_matmul_aie2p.mlir"
        )
        self.elem_script = os.path.join(
            self.script_dir, "transform_elementwise_aie2p.mlir"
        )
        self.add_script = os.path.join(self.script_dir, "transform_add_aie2p.mlir")
        self.add_f32_script = os.path.join(
            self.script_dir, "transform_add_f32_aie2p.mlir"
        )
        self.softmax_script = os.path.join(
            self.script_dir, "transform_softmax_aie2p.mlir"
        )
        self.layernorm_script = os.path.join(
            self.script_dir, "transform_layernorm_aie2p.mlir"
        )
        self.gelu_f32in_script = os.path.join(
            self.script_dir, "transform_gelu_f32in_aie2p.mlir"
        )

        # Fuse mlp_fc->gelu->mlp_proj into one load_pdi ELF on NPU. On by default
        # whenever the MLP runs on NPU (hetero/hetero-fast or npu mode); the
        # unfused per-op path is ~10x slower, so it's only kept as an escape
        # hatch via AMD_TRITON_NPU_FUSED_MLP=0.
        self._fused_mlp = None
        if (self._is_hetero or backend == "npu") and os.getenv(
            "AMD_TRITON_NPU_FUSED_MLP", "1"
        ) == "1":
            self._fused_mlp = _FusedMLP(
                self.n_embd,
                self.mlp_dim,
                self.matmul_script,
                self.gelu_f32in_script,
                self.add_f32_script,
            )

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
                "qkv_weight": sd[f"{prefix}.attn.c_attn.weight"].to(
                    dtype
                ),  # (768, 2304)
                "qkv_bias": sd[f"{prefix}.attn.c_attn.bias"].to(dtype),  # (2304,)
                # Attention output projection
                "attn_proj_weight": sd[f"{prefix}.attn.c_proj.weight"].to(
                    dtype
                ),  # (768, 768)
                "attn_proj_bias": sd[f"{prefix}.attn.c_proj.bias"].to(dtype),  # (768,)
                # Pre-MLP layernorm
                "ln2_weight": sd[f"{prefix}.ln_2.weight"].to(dtype),
                "ln2_bias": sd[f"{prefix}.ln_2.bias"].to(dtype),
                # MLP
                "mlp_fc_weight": sd[f"{prefix}.mlp.c_fc.weight"].to(
                    dtype
                ),  # (768, 3072)
                "mlp_fc_bias": sd[f"{prefix}.mlp.c_fc.bias"].to(dtype),  # (3072,)
                "mlp_proj_weight": sd[f"{prefix}.mlp.c_proj.weight"].to(
                    dtype
                ),  # (3072, 768)
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
        npu_keys = {
            "ln1_weight",
            "ln1_bias",
            "ln2_weight",
            "ln2_bias",
            "mlp_fc_weight",
            "mlp_fc_bias",
            "mlp_proj_weight",
            "mlp_proj_bias",
        }
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
                x,
                weight.t(),
                bias=bias,
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
                x,
                weight,
                bias,
                eps=LN_EPS,
                backend=be,
                transform_script=self.layernorm_script if be == "npu" else None,
            )
        except Exception as e:
            logger.warning(f"Triton layernorm failed ({e}), falling back to PyTorch")
            out = torch.nn.functional.layer_norm(
                x.to(torch.float32),
                (x.shape[-1],),
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
                x,
                backend=be,
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
                a,
                b,
                backend=be,
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
                x,
                causal_mask=causal_mask,
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
            cache_k[:, :, seq_pos : seq_pos + S, :] = k_new
            cache_v[:, :, seq_pos : seq_pos + S, :] = v_new
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
            k_3d = k_full.reshape(
                B * self.n_head, total_len, self.head_dim
            ).contiguous()
            v_3d = v_full.reshape(
                B * self.n_head, total_len, self.head_dim
            ).contiguous()
            is_causal = (
                S > 1
            )  # Decode (S=1): every past position visible, no causal mask
            attn_output = triton_fused_attention(
                q_3d,
                k_3d,
                v_3d,
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
                causal_flat = causal_mask.expand(B, self.n_head, S, total_len).reshape(
                    -1, total_len
                )
                attn_weights_flat = self._softmax(
                    attn_flat, causal_mask=causal_flat, backend=softmax_be
                )
                attn_weights = attn_weights_flat.reshape(B, self.n_head, S, total_len)
            attn_output = torch.matmul(attn_weights, v_full)

        # Reshape back: (B, n_head, S, head_dim) -> (B, S, n_embd)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, self.n_embd)

        # Output projection: (B, S, 768) -> (B, S, 768)
        proj = self._linear(
            attn_output,
            layer["attn_proj_weight"],
            layer["attn_proj_bias"],
            backend=proj_be,
        )

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
        positions = torch.arange(
            pos_offset, pos_offset + S, dtype=torch.long, device=input_ids.device
        )
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
                    x_norm = self._layernorm(
                        x, layer["ln1_weight"], layer["ln1_bias"], backend="gpu"
                    )

                layer_cache = kv_caches[i] if kv_caches else None
                with self.timer.track("attention"):
                    attn_out, new_cache = self._attention(
                        x_norm, layer, kv_cache=layer_cache, pos_offset=pos_offset
                    )
                new_kv_caches.append(new_cache)

                with self.timer.track("add1"):
                    x = self._add(x, attn_out, backend="gpu")
                with self.timer.track("ln2"):
                    x_norm = self._layernorm(
                        x, layer["ln2_weight"], layer["ln2_bias"], backend="gpu"
                    )
                with self.timer.track("mlp_fc"):
                    h = self._linear(
                        x_norm,
                        layer["mlp_fc_weight"],
                        layer["mlp_fc_bias"],
                        backend="gpu",
                    )
                with self.timer.track("gelu"):
                    h = self._gelu(h, backend="gpu")
                with self.timer.track("mlp_proj"):
                    mlp_out = self._linear(
                        h,
                        layer["mlp_proj_weight"],
                        layer["mlp_proj_bias"],
                        backend="gpu",
                    )
                with self.timer.track("add2"):
                    x = self._add(x, mlp_out, backend="gpu")

            elif hetero:
                # --- HETERO PATH (prefill for hetero-fast, all steps for hetero) ---
                # GPU for attention; NPU for LN/MLP/add.
                # hetero-fast: weights are on GPU, use _cpu_layers for NPU ops
                # hetero: LN/MLP weights are already on CPU in layer dict
                npu_w = self._cpu_layers[i] if hasattr(self, "_cpu_layers") else layer
                ln1_be = self.op_backend["layernorm"]
                # LayerNorm on tiny (<=4x768) CPU tensors: NPU dispatch overhead
                # (~1.5-4ms) dwarfs the compute, so normalize on host.
                with self.timer.track("ln1"):
                    x_norm = torch.nn.functional.layer_norm(
                        x.to(torch.float32),
                        (self.n_embd,),
                        npu_w["ln1_weight"].to(torch.float32),
                        npu_w["ln1_bias"].to(torch.float32),
                        eps=LN_EPS,
                    )
                with self.timer.track("to_gpu"):
                    x_norm = self._to_gpu(x_norm)

                layer_cache = kv_caches[i] if kv_caches else None
                with self.timer.track("attention"):
                    attn_out, new_cache = self._attention(
                        x_norm, layer, kv_cache=layer_cache, pos_offset=pos_offset
                    )
                new_kv_caches.append(new_cache)

                with self.timer.track("to_cpu"):
                    attn_out = self._to_cpu(attn_out)

                add_be = self.op_backend["add"]
                # add1 is a trivial elementwise residual add on tiny (<=4x768)
                # CPU tensors; NPU dispatch overhead (~2.8ms) dwarfs the compute,
                # so do it on host. No transfer (x and attn_out already on CPU).
                with self.timer.track("add1"):
                    x = x.to(torch.float32) + attn_out.to(torch.float32)

                # NPU ops: use CPU-resident weights
                npu_w = self._cpu_layers[i] if hasattr(self, "_cpu_layers") else layer
                ln2_be = self.op_backend["layernorm"]
                with self.timer.track("ln2"):
                    x_norm = torch.nn.functional.layer_norm(
                        x.to(torch.float32),
                        (self.n_embd,),
                        npu_w["ln2_weight"].to(torch.float32),
                        npu_w["ln2_bias"].to(torch.float32),
                        eps=LN_EPS,
                    )

                mlp_fc_be = self.op_backend["mlp_fc"]
                gelu_be = self.op_backend["gelu"]
                mlp_proj_be = self.op_backend["mlp_proj"]
                # Fused MLP fast path: one load_pdi ELF for fc->gelu->proj. Only
                # when all three route to NPU and the (flattened) row count fits a
                # single M-block (<=256); otherwise fall through to the unfused
                # path. mlp_proj bias is applied inside the fused helper.
                _fused_ok = (
                    self._fused_mlp is not None
                    and mlp_fc_be == "npu"
                    and gelu_be == "npu"
                    and mlp_proj_be == "npu"
                    and x_norm.reshape(-1, self.n_embd).shape[0] <= self._fused_mlp.BM
                )
                if _fused_ok:
                    # The fused chain folds the post-MLP residual add (op3), so
                    # run() returns x + mlp(x_norm) directly.
                    with self.timer.track("mlp_fused"):
                        x = self._fused_mlp.run(
                            i,
                            x_norm,
                            x,
                            npu_w["mlp_fc_weight"],
                            npu_w["mlp_fc_bias"],
                            npu_w["mlp_proj_weight"],
                            npu_w["mlp_proj_bias"],
                        )
                else:
                    with self.timer.track("mlp_fc"):
                        h = self._linear(
                            x_norm,
                            npu_w["mlp_fc_weight"],
                            npu_w["mlp_fc_bias"],
                            backend=mlp_fc_be,
                        )
                    with self.timer.track("gelu"):
                        h = self._gelu(h, backend=gelu_be)
                    with self.timer.track("mlp_proj"):
                        mlp_out = self._linear(
                            h,
                            npu_w["mlp_proj_weight"],
                            npu_w["mlp_proj_bias"],
                            backend=mlp_proj_be,
                        )
                    with self.timer.track("add2"):
                        x = self._add(x, mlp_out, backend=add_be)

            else:
                # --- SINGLE-BACKEND PATH (gpu or npu) ---
                with self.timer.track("ln1"):
                    x_norm = self._layernorm(x, layer["ln1_weight"], layer["ln1_bias"])

                layer_cache = kv_caches[i] if kv_caches else None
                with self.timer.track("attention"):
                    attn_out, new_cache = self._attention(
                        x_norm, layer, kv_cache=layer_cache, pos_offset=pos_offset
                    )
                new_kv_caches.append(new_cache)

                with self.timer.track("add1"):
                    x = self._add(x, attn_out)
                with self.timer.track("ln2"):
                    x_norm = self._layernorm(x, layer["ln2_weight"], layer["ln2_bias"])
                # Fused MLP fast path (npu mode): one load_pdi ELF for
                # fc->gelu->proj, when the flattened row count fits a single
                # M-block (<=256); else the unfused 3-dispatch path. Not used on
                # gpu (the fused helper builds NPU ELFs). mlp_proj bias is applied
                # inside the fused helper.
                _fused_ok = (
                    self._fused_mlp is not None
                    and self.backend == "npu"
                    and x_norm.reshape(-1, self.n_embd).shape[0] <= self._fused_mlp.BM
                )
                if _fused_ok:
                    # run() folds the post-MLP residual add (op3), returning
                    # x + mlp(x_norm) directly.
                    with self.timer.track("mlp_fused"):
                        x = self._fused_mlp.run(
                            i,
                            x_norm,
                            x,
                            layer["mlp_fc_weight"],
                            layer["mlp_fc_bias"],
                            layer["mlp_proj_weight"],
                            layer["mlp_proj_bias"],
                        )
                else:
                    with self.timer.track("mlp_fc"):
                        h = self._linear(
                            x_norm, layer["mlp_fc_weight"], layer["mlp_fc_bias"]
                        )
                    with self.timer.track("gelu"):
                        h = self._gelu(h)
                    with self.timer.track("mlp_proj"):
                        mlp_out = self._linear(
                            h, layer["mlp_proj_weight"], layer["mlp_proj_bias"]
                        )
                    with self.timer.track("add2"):
                        x = self._add(x, mlp_out)

        # Final LayerNorm
        if hetero and not decode_gpu:
            # Final LN on CPU: tiny tensor, NPU dispatch overhead dominates.
            with self.timer.track("ln_f"):
                ln_f_w = (
                    self._cpu_ln_f_weight
                    if hasattr(self, "_cpu_ln_f_weight")
                    else self.ln_f_weight
                )
                ln_f_b = (
                    self._cpu_ln_f_bias
                    if hasattr(self, "_cpu_ln_f_bias")
                    else self.ln_f_bias
                )
                x = torch.nn.functional.layer_norm(
                    x.to(torch.float32),
                    (self.n_embd,),
                    ln_f_w.to(torch.float32),
                    ln_f_b.to(torch.float32),
                    eps=LN_EPS,
                )
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
                logits = (
                    x.to(device="cuda", dtype=torch.float32) @ self._wte_lm_head.t()
                ).cpu()
            else:
                # GPU: x and wte already on CUDA
                logits = x.to(torch.float32) @ self.wte.to(torch.float32).t()
                if logits.device.type == "cuda":
                    logits = logits.cpu()

        return logits, new_kv_caches

    def _allocate_kv_caches(
        self, batch_size, max_seq_len, device, dtype=torch.bfloat16
    ):
        """Pre-allocate KV cache buffers for all layers."""
        caches = []
        for _ in range(self.n_layer):
            cache_k = torch.zeros(
                batch_size,
                self.n_head,
                max_seq_len,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
            cache_v = torch.zeros(
                batch_size,
                self.n_head,
                max_seq_len,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
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
                logits, kv_caches = self.forward(
                    next_input, kv_caches=kv_caches, pos_offset=pos_offset
                )
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
