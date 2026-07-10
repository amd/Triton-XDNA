# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Qwen2.5 model wired with Triton kernels for GPU, NPU, and hetero inference.

Architecture (Qwen2 decoder), differs from GPT-2 in several ways:
  - RMSNorm (no mean subtraction, no bias) instead of LayerNorm
  - Rotary position embeddings (RoPE) instead of learned position embeddings
  - Grouped-query attention (GQA): fewer KV heads than Q heads
  - SwiGLU MLP: down_proj(SiLU(gate_proj(x)) * up_proj(x))
  - q/k/v projections carry biases; o_proj and the MLP have none
  - Tied embeddings (lm_head shares embed_tokens weights)

This module is config-driven via QWEN_CONFIGS so the same code serves multiple
model sizes (0.5B, 1.5B); select one with the --model CLI flag.

Four backends route operators across iGPU and NPU:
  "gpu":         all ops on iGPU via ROCm/Triton
  "npu":         all ops on NPU via MLIR-AIR/AIE (attention falls back to GPU
                 fused kernel, matching the gpt2 example)
  "hetero":      attention on iGPU; RMSNorm/MLP/add on NPU (prefill & decode)
  "hetero-fast": same as hetero for prefill; all-GPU decode (lower TPOT)

RMSNorm runs on NPU in hetero modes because the NPU kernel computes in float32,
matching the gpt2 example's rationale that float32 norm precision avoids logit
drift across many transformer layers.
"""

import os
import logging
import time
import math
from collections import defaultdict
from contextlib import contextmanager

import torch
import numpy as np

from kernels import (
    triton_linear,
    triton_softmax,
    triton_rmsnorm,
    triton_swiglu,
    triton_add,
    triton_fused_attention,
    precompute_rope_cache,
    apply_rope,
)

logger = logging.getLogger(__name__)


# Config-driven across Qwen2.5 sizes. All variants share vocab, context length,
# KV-head count, RoPE theta, and RMSNorm epsilon — only depth/width differ — so
# the same kernels and transform scripts serve every size.
QWEN_CONFIGS = {
    "qwen2.5-0.5b": {
        "hf_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "n_layer": 24,
        "n_head": 14,
        "n_embd": 896,
        "head_dim": 64,  # n_embd // n_head
        "intermediate": 4864,
    },
    "qwen2.5-1.5b": {
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "n_layer": 28,
        "n_head": 12,
        "n_embd": 1536,
        "head_dim": 128,  # n_embd // n_head
        "intermediate": 8960,
    },
}

# Shared across all Qwen2.5 variants
N_KV_HEAD = 2
VOCAB_SIZE = 151936
MAX_SEQ_LEN = 32768
ROPE_THETA = 1000000.0
RMS_EPS = 1e-6

# Default config
QWEN_CONFIG = {
    **QWEN_CONFIGS["qwen2.5-0.5b"],
    "n_kv_head": N_KV_HEAD,
    "vocab_size": VOCAB_SIZE,
    "max_seq_len": MAX_SEQ_LEN,
    "rope_theta": ROPE_THETA,
    "rms_eps": RMS_EPS,
}


class OpTimer:
    """Lightweight per-op wall-clock timer. Zero overhead when disabled."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.records = []

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


# Default hetero routing policy: which device runs each operator
HETERO_ROUTING = {
    "rmsnorm": "npu",
    "qkv_linear": "gpu",
    "attn_proj": "gpu",
    "softmax": "gpu",
    "mlp_gate": "npu",
    "mlp_up": "npu",
    "swiglu": "npu",
    "mlp_down": "npu",
    "add": "npu",
}


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


class _FusedMLP:
    """Fuse the Qwen SwiGLU MLP (gate, up, silu*mul, down) + residual add into
    ONE load_pdi multi-launch ELF.

    Replaces five separate NPU dispatches (each paying the ~147ms hw_context
    rebuild) with a single fused ELF dispatched through one persistent
    hw_context + one xrt.run. See docs/load_pdi_multilaunch_design.md and the
    gpt2 example's _FusedMLP (linear fc->gelu->proj chain).

    Qwen's MLP is a diamond, not a line: gate_proj and up_proj both read the
    same ln(x), then merge via silu(gate)*up before down_proj:

      op0 gate_mm: Cg = x_norm @ W_gate           (bf16 -> f32)
      op1 up_mm:   Cu = x_norm @ W_up             (bf16 -> f32; same A as op0)
      op2 swiglu:  H  = silu(Cg) * Cu             (f32,f32 -> bf16)
      op3 down_mm: Cd = H @ W_down                (bf16 -> f32)
      op4 add:     out = Cd + residual            (f32; folded residual add)

    Qwen2.5 MLP has NO biases (gate/up/down bias=False), so unlike gpt2 there is
    no augmented-K bias fold and no host-side bias add -- run() returns
    x + mlp(x_norm) directly.

    Shapes (HF stores proj weights as (out, in), used as x@W^T):
      W_gate=(H,D), W_up=(H,D), W_down=(D,H).  D=n_embd, H=intermediate_size.
    K_pad = next_pow2(D); HID_pad = next_pow2(H); both matmuls single-block.
    M is padded to 256 (BLOCK_M). Only valid rows/cols are read back.
    """

    def __init__(
        self,
        n_embd,
        intermediate_size,
        matmul_script,
        swiglu_f32in_script,
        add_f32_script,
        rmsnorm_script,
    ):
        self.D = n_embd
        self.H = intermediate_size
        self.matmul_script = matmul_script
        self.swiglu_script = swiglu_f32in_script
        self.add_script = add_f32_script
        self.rmsnorm_script = rmsnorm_script
        self.eps = RMS_EPS
        # BM=128 is the transform's minimum M-tile. At decode (M_real=1) this
        # halves the wasted matmul rows and all M-dimension host buffers vs the
        # old 256; prefill (39 rows) still fits one M-block. BN stays 256.
        self.BM = 128
        self.BN = 256
        self.K_pad = _next_pow2(self.D)
        self.HID_pad = _next_pow2(self.H)
        # The down matmul's N (and the residual/output) is D, which is not a
        # multiple of BN for Qwen (896, 1536), so it must be padded to a block
        # multiple or the grid truncates and drops output columns. next_pow2(D)
        # is always a multiple of 256 for D>=256, so reuse it (== K_pad).
        self.D_pad = self.K_pad
        # One shared chain (one stitched ELF + one hw_context) serves every
        # layer; per-layer weights are swapped in via bo_key at run time. A
        # chain-per-layer would allocate one hw_context per layer and exhaust the
        # NPU on deep models (24/28 layers). Mirrors gpt2's _FusedMLP and the
        # upstream mlir-air llama pattern (ELF compiled once, per-layer BOs).
        self._chain = None
        self._weights = {}  # layer_idx -> (Bg, Bu, Bd)
        self._mm = None
        self._swiglu = None
        self._add = None
        self._build_kernels()

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
        def _swiglu_f32in(G, U, Y, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            g = tl.load(G + offsets[:])
            u = tl.load(U + offsets[:])
            y = (g * tl.sigmoid(g) * u).to(tl.bfloat16)
            tl.store(Y + offsets[:], y)

        @triton.jit
        def _add_kernel(A, B, C, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            a = tl.load(A + offsets[:])
            b = tl.load(B + offsets[:])
            tl.store(C + offsets[:], a + b)

        @triton.jit
        def _rmsnorm_bare(
            X,
            Y,
            sxr: tl.constexpr,
            sxc: tl.constexpr,
            syr: tl.constexpr,
            syc: tl.constexpr,
            N_pad: tl.constexpr,
            N_real: tl.constexpr,
            eps: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            # Bare RMSNorm (no gamma; gamma is folded into the gate/up weights)
            # for the fused chain. Reads f32, stores bf16 so the output feeds the
            # gate/up matmuls directly. Mirrors kernels/rmsnorm.rmsnorm_kernel_npu
            # so it reuses transform_rmsnorm_aie2p.mlir. Padded cols (>=N_real)
            # are zero, so summing over N_pad and dividing by N_real is correct.
            pid_row = tl.program_id(0)
            offs_row = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            offs_col = tl.arange(0, N_pad)
            a = tl.load(X + offs_row[:, None] * sxr + offs_col[None, :] * sxc)
            sum_sq = tl.sum(a * a, axis=1, keep_dims=True)
            inv_rms = 1.0 / tl.sqrt(sum_sq / N_real + eps)
            out = a * inv_rms
            tl.store(Y + offs_row[:, None] * syr + offs_col[None, :] * syc, out)

        self._mm = _mm_kernel
        self._swiglu = _swiglu_f32in
        self._add = _add_kernel
        self._rmsnorm = _rmsnorm_bare

    def _prep_weights(self, w_gate, w_up, w_down, gamma):
        """Build padded bf16 weight arrays (numpy). HF stores each as (out, in);
        transpose to (in, out) for x@W, then pad to the block dims.

        The post_ln gamma is folded into the gate/up weights by row-scaling:
        gate = (bare_norm * gamma) @ w_gate = bare_norm @ (gamma[:,None] * w_gate).
        This lets the in-chain rmsnorm stay *bare* (no gamma), so it reuses the
        standalone rmsnorm transform. Down is unchanged.
        """
        from ml_dtypes import bfloat16

        D, H, K_pad, HID_pad, D_pad = (
            self.D,
            self.H,
            self.K_pad,
            self.HID_pad,
            self.D_pad,
        )
        gamma = gamma.to(torch.float32).cpu().numpy()  # (D,)
        w_gate = w_gate.t().to(torch.float32).cpu().numpy()  # (D, H)
        w_up = w_up.t().to(torch.float32).cpu().numpy()  # (D, H)
        w_down = w_down.t().to(torch.float32).cpu().numpy()  # (H, D)
        w_gate = w_gate * gamma[:, None]  # fold gamma (rows)
        w_up = w_up * gamma[:, None]
        # Bg, Bu: (K_pad, HID_pad); Bd: (HID_pad, D_pad)
        Bg = np.zeros((K_pad, HID_pad), dtype=bfloat16)
        Bg[:D, :H] = w_gate.astype(bfloat16)
        Bu = np.zeros((K_pad, HID_pad), dtype=bfloat16)
        Bu[:D, :H] = w_up.astype(bfloat16)
        Bd = np.zeros((HID_pad, D_pad), dtype=bfloat16)
        Bd[:H, :D] = w_down.astype(bfloat16)
        return Bg, Bu, Bd

    def _get_chain(self):
        """Build the shared SwiGLU chain once (every layer has the same MLP
        shape). One stitched ELF + one hw_context serves all layers; per-layer
        weights are supplied at run time via bo_key. Warmup tensors are
        shape-representative placeholders."""
        if self._chain is not None:
            return self._chain
        from triton.backends.amd_triton_npu.multilaunch import NPUChain

        D, H, K_pad, HID_pad, D_pad, BM, BN = (
            self.D,
            self.H,
            self.K_pad,
            self.HID_pad,
            self.D_pad,
            self.BM,
            self.BN,
        )
        M_pad = self.BM  # single M-block; valid rows sliced on readback

        # Buffer layout (folds add1 + rmsnorm(post_ln) + residual add2 into the
        # chain, so the whole per-layer NPU tail is ONE load_pdi dispatch):
        #   0 Xpre  f32 (M_pad,D_pad)  pre-attn hidden (add1 lhs)
        #   1 Aout  f32 (M_pad,D_pad)  attn output from GPU (add1 rhs)
        #   2 X1    f32 (M_pad,D_pad)  x1 = Xpre+Aout (add2 residual; rmsnorm in)
        #   3 A     bf16(M_pad,K_pad)  rmsnorm(X1) [gamma folded into Bg/Bu]
        #   4 Bg static  5 Cg inter   6 Bu static  7 Cu inter
        #   8 H inter (swiglu)        9 Bd static  10 Cd inter   11 OUT output
        #   12 Xn  f32 (M_pad,D_pad)  bare rmsnorm(OUT) for NEXT layer's attention
        tOutm = torch.zeros((M_pad, D_pad), dtype=torch.float32)
        tXn = torch.zeros((M_pad, D_pad), dtype=torch.float32)
        tXpre = torch.zeros((M_pad, D_pad), dtype=torch.float32)
        tAout = torch.zeros((M_pad, D_pad), dtype=torch.float32)
        tX1 = torch.zeros((M_pad, D_pad), dtype=torch.float32)
        tX1f = torch.zeros(M_pad * D_pad, dtype=torch.float32)
        tXpref = torch.zeros(M_pad * D_pad, dtype=torch.float32)
        tAoutf = torch.zeros(M_pad * D_pad, dtype=torch.float32)
        tA = torch.zeros((M_pad, K_pad), dtype=torch.bfloat16)
        tBg = torch.zeros((K_pad, HID_pad), dtype=torch.bfloat16)
        tCg = torch.zeros((M_pad, HID_pad), dtype=torch.float32)
        tBu = torch.zeros((K_pad, HID_pad), dtype=torch.bfloat16)
        tCu = torch.zeros((M_pad, HID_pad), dtype=torch.float32)
        tCgf = torch.zeros(M_pad * HID_pad, dtype=torch.float32)
        tCuf = torch.zeros(M_pad * HID_pad, dtype=torch.float32)
        tH = torch.zeros(M_pad * HID_pad, dtype=torch.bfloat16)
        tHm = torch.zeros((M_pad, HID_pad), dtype=torch.bfloat16)
        tBd = torch.zeros((HID_pad, D_pad), dtype=torch.bfloat16)
        tCd = torch.zeros((M_pad, D_pad), dtype=torch.float32)
        tCdf = torch.zeros(M_pad * D_pad, dtype=torch.float32)
        tOut = torch.zeros(M_pad * D_pad, dtype=torch.float32)

        RMS_BM = 4  # rows per rmsnorm program (matches standalone kernel)
        chain = NPUChain("qwen_mlp")
        # op0 add1: X1(2) = Xpre(0) + Aout(1) -- post-attention residual add.
        chain.add(
            self._add,
            grid=((M_pad * D_pad) // 1024,),
            arg_map={0: 0, 1: 1, 2: 2},
            args=(tXpref, tAoutf, tX1f, M_pad * D_pad),
            constexprs={"BLOCK_SIZE": 1024},
            transform_script=self.add_script,
        )
        # op1 rmsnorm: A(3) = bare_rmsnorm(X1(2)); gamma folded into Bg/Bu.
        chain.add(
            self._rmsnorm,
            grid=(M_pad // RMS_BM, 1),
            arg_map={0: 2, 1: 3},
            args=(tX1, tA, D_pad, 1, K_pad, 1, D_pad, self.D, self.eps),
            constexprs={"BLOCK_SIZE": RMS_BM},
            transform_script=self.rmsnorm_script,
        )
        # op2 gate_mm: Cg(5) = A(3) @ Bg(4)
        chain.add(
            self._mm,
            grid=(M_pad // BM, HID_pad // BN),
            arg_map={0: 3, 1: 4, 2: 5},
            args=(
                tA,
                tBg,
                tCg,
                M_pad,
                HID_pad,
                K_pad,
                K_pad,
                1,
                HID_pad,
                1,
                HID_pad,
                1,
            ),
            constexprs={"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": K_pad},
            transform_script=self.matmul_script,
        )
        # op3 up_mm: Cu(7) = A(3) @ Bu(6)  -- shares input buffer 3 with op2
        chain.add(
            self._mm,
            grid=(M_pad // BM, HID_pad // BN),
            arg_map={0: 3, 1: 6, 2: 7},
            args=(
                tA,
                tBu,
                tCu,
                M_pad,
                HID_pad,
                K_pad,
                K_pad,
                1,
                HID_pad,
                1,
                HID_pad,
                1,
            ),
            constexprs={"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": K_pad},
            transform_script=self.matmul_script,
        )
        # op4 swiglu: H(8) = silu(Cg(5)) * Cu(7)
        chain.add(
            self._swiglu,
            grid=((M_pad * HID_pad) // 1024,),
            arg_map={0: 5, 1: 7, 2: 8},
            args=(tCgf, tCuf, tH, M_pad * HID_pad),
            constexprs={"BLOCK_SIZE": 1024},
            transform_script=self.swiglu_script,
        )
        # op5 down_mm: Cd(10) = H(8) @ Bd(9).
        chain.add(
            self._mm,
            grid=(M_pad // BM, D_pad // BN),
            arg_map={0: 8, 1: 9, 2: 10},
            args=(tHm, tBd, tCd, M_pad, D_pad, HID_pad, HID_pad, 1, D_pad, 1, D_pad, 1),
            constexprs={
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": HID_pad,
            },
            transform_script=self.matmul_script,
        )
        # op6 add2: OUT(11) = Cd(10) + X1(2) -- residual add on the pre-norm x1.
        chain.add(
            self._add,
            grid=((M_pad * D_pad) // 1024,),
            arg_map={0: 10, 1: 2, 2: 11},
            args=(tCdf, tX1f, tOut, M_pad * D_pad),
            constexprs={"BLOCK_SIZE": 1024},
            transform_script=self.add_script,
        )
        # op7 rmsnorm_in for the NEXT layer: Xn(12) = bare_rmsnorm(OUT(11)).
        # The block output x_out is the next layer's residual; its input_layernorm
        # is bare here (that gamma is applied host-side / folded into the next
        # qkv) so this reuses the standalone rmsnorm transform. Folding it here
        # removes the standalone per-layer rmsnorm_in dispatch (2->1 NPU/layer).
        chain.add(
            self._rmsnorm,
            grid=(M_pad // RMS_BM, 1),
            arg_map={0: 11, 1: 12},
            args=(tOutm, tXn, D_pad, 1, D_pad, 1, D_pad, self.D, self.eps),
            constexprs={"BLOCK_SIZE": RMS_BM},
            transform_script=self.rmsnorm_script,
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

    def _weights_for(self, layer_idx, w_gate, w_up, w_down, gamma):
        """Per-layer prepped weights (cached). Shape is identical across layers;
        only the values differ, so they ride the shared chain as per-layer BOs."""
        if layer_idx not in self._weights:
            self._weights[layer_idx] = self._prep_weights(w_gate, w_up, w_down, gamma)
        return self._weights[layer_idx]

    def run(self, layer_idx, x_pre, attn_out, w_gate, w_up, w_down, gamma):
        """Run the fused per-layer NPU tail for one layer in ONE dispatch:
        add1 (x1 = x_pre + attn_out) -> rmsnorm(post_ln) -> SwiGLU MLP ->
        residual add2 (x1 + mlp). x_pre is the pre-attention hidden, attn_out is
        the GPU attention output. gamma (post_ln) is folded into the gate/up
        weights so the in-chain rmsnorm is bare. Returns x1 + mlp(rmsnorm(x1))
        (B,S,D) torch f32 -- the full block output after the MLP residual add.
        """
        from ml_dtypes import bfloat16

        D, K_pad, HID_pad, D_pad = self.D, self.K_pad, self.HID_pad, self.D_pad
        chain = self._get_chain()
        Bg, Bu, Bd = self._weights_for(layer_idx, w_gate, w_up, w_down, gamma)

        orig_shape = x_pre.shape
        xp2d = x_pre.reshape(-1, D).to(torch.float32).cpu().numpy()  # (M_real, D)
        ao2d = attn_out.reshape(-1, D).to(torch.float32).cpu().numpy()
        M_real = xp2d.shape[0]
        M_pad = self.BM
        if M_real > M_pad:
            # Multiple M-blocks not handled by this single-block chain; caller
            # keeps long prefill on the unfused path.
            raise ValueError(
                f"_FusedMLP only supports M<={M_pad} (got {M_real}); "
                "use the unfused MLP path for longer sequences."
            )
        # Xpre/Aout need zero padding rows/cols: add1 and the in-chain rmsnorm
        # (reduces over D_pad cols) require padding to be 0. X1/A/Cg/Cu/Hbuf/Cd
        # are chain intermediates fully overwritten by their producing kernel, so
        # np.empty avoids the per-call memset.
        Xpre = np.zeros((M_pad, D_pad), dtype=np.float32)
        Xpre[:M_real, :D] = xp2d
        Aout = np.zeros((M_pad, D_pad), dtype=np.float32)
        Aout[:M_real, :D] = ao2d
        X1 = np.empty((M_pad, D_pad), dtype=np.float32)
        A = np.empty((M_pad, K_pad), dtype=bfloat16)
        Cg = np.empty((M_pad, HID_pad), dtype=np.float32)
        Cu = np.empty((M_pad, HID_pad), dtype=np.float32)
        Hbuf = np.empty(M_pad * HID_pad, dtype=bfloat16)
        Cd = np.empty((M_pad, D_pad), dtype=np.float32)
        OUT = np.empty((M_pad, D_pad), dtype=np.float32)
        Xn = np.empty((M_pad, D_pad), dtype=np.float32)

        out = chain.run(
            [Xpre, Aout, X1, A, Bg, Cg, Bu, Cu, Hbuf, Bd, Cd, OUT, Xn],
            bo_key=f"qwen_mlp_L{layer_idx}",
            static_indices={4, 6, 9},
            intermediate_indices={2, 3, 5, 7, 8, 10},
            output_indices={11, 12},
        )
        res = out[11].astype(np.float32)[:M_real, :D]
        # x_norm_bare = rmsnorm(x_out) for the next layer's attention (bare; the
        # next input_layernorm gamma is applied by the caller). Returned as a 2D
        # (M_real, D) numpy array to avoid an extra torch copy on the hot path.
        x_norm_bare = out[12].astype(np.float32)[:M_real, :D]
        return torch.from_numpy(res.copy()).reshape(orig_shape), x_norm_bare


class Qwen2Model:
    """Qwen2.5 inference model using Triton kernels."""

    def __init__(self, state_dict, backend="gpu", config=None):
        # Merge variant-specific config over shared defaults so per-variant
        # registry entries only need to carry the dims that differ.
        self.cfg = {
            "n_kv_head": N_KV_HEAD,
            "vocab_size": VOCAB_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "rope_theta": ROPE_THETA,
            "rms_eps": RMS_EPS,
            **(config or QWEN_CONFIG),
        }
        self.backend = backend
        self._is_hetero = backend in ("hetero", "hetero-fast")
        self.op_backend = HETERO_ROUTING if self._is_hetero else None
        self.timer = OpTimer(enabled=False)

        self.n_layer = self.cfg["n_layer"]
        self.n_head = self.cfg["n_head"]
        self.n_kv_head = self.cfg["n_kv_head"]
        self.n_embd = self.cfg["n_embd"]
        self.head_dim = self.cfg["head_dim"]
        self.kv_dim = self.n_kv_head * self.head_dim
        self.n_rep = self.n_head // self.n_kv_head
        self.intermediate = self.cfg["intermediate"]
        self.rms_eps = self.cfg["rms_eps"]

        self._load_weights(state_dict)

        if backend == "gpu":
            self._move_weights_to_gpu()
        elif backend == "hetero-fast":
            self._place_weights_fast()
        elif backend == "hetero":
            self._place_weights()

        # LM head matmul (tied to embeddings). Cache on GPU for speed.
        if backend in ("npu", "hetero", "hetero-fast"):
            self._lm_head = self.embed_tokens.to(device="cuda", dtype=torch.float32)
        else:
            self._lm_head = None

        # RoPE tables (float32 on CPU; sliced/moved per call)
        self.rope_cos, self.rope_sin = precompute_rope_cache(
            self.head_dim, self.cfg["max_seq_len"], base=self.cfg["rope_theta"]
        )

        # Transform scripts
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
        self.rmsnorm_script = os.path.join(
            self.script_dir, "transform_rmsnorm_aie2p.mlir"
        )
        self.swiglu_f32in_script = os.path.join(
            self.script_dir, "transform_swiglu_f32in_aie2p.mlir"
        )

        # Optional: fuse the SwiGLU MLP (gate->up->silu*mul->down->residual add)
        # into one load_pdi ELF on NPU. Opt-in via AMD_TRITON_NPU_FUSED_MLP=1;
        # active whenever the MLP runs on NPU -- i.e. hetero/hetero-fast (MLP
        # routed to NPU) or npu mode. See docs/load_pdi_multilaunch_design.md.
        self._fused_mlp = None
        if (self._is_hetero or backend == "npu") and os.getenv(
            "AMD_TRITON_NPU_FUSED_MLP", "1"
        ) == "1":
            self._fused_mlp = _FusedMLP(
                self.n_embd,
                self.intermediate,
                self.matmul_script,
                self.swiglu_f32in_script,
                self.add_f32_script,
                self.rmsnorm_script,
            )

    def _load_weights(self, sd):
        dtype = torch.bfloat16

        self.embed_tokens = sd["model.embed_tokens.weight"].to(dtype)  # (vocab, n_embd)

        self.layers = []
        for i in range(self.n_layer):
            prefix = f"model.layers.{i}"
            layer = {
                "input_ln": sd[f"{prefix}.input_layernorm.weight"].to(dtype),
                "post_ln": sd[f"{prefix}.post_attention_layernorm.weight"].to(dtype),
                "q_weight": sd[f"{prefix}.self_attn.q_proj.weight"].to(dtype),
                "q_bias": sd[f"{prefix}.self_attn.q_proj.bias"].to(dtype),
                "k_weight": sd[f"{prefix}.self_attn.k_proj.weight"].to(dtype),
                "k_bias": sd[f"{prefix}.self_attn.k_proj.bias"].to(dtype),
                "v_weight": sd[f"{prefix}.self_attn.v_proj.weight"].to(dtype),
                "v_bias": sd[f"{prefix}.self_attn.v_proj.bias"].to(dtype),
                "o_weight": sd[f"{prefix}.self_attn.o_proj.weight"].to(dtype),
                "gate_weight": sd[f"{prefix}.mlp.gate_proj.weight"].to(dtype),
                "up_weight": sd[f"{prefix}.mlp.up_proj.weight"].to(dtype),
                "down_weight": sd[f"{prefix}.mlp.down_proj.weight"].to(dtype),
            }
            self.layers.append(layer)

        self.norm_weight = sd["model.norm.weight"].to(dtype)

    def _move_weights_to_gpu(self):
        device = "cuda"
        self.embed_tokens = self.embed_tokens.to(device)
        for layer in self.layers:
            for k, v in layer.items():
                layer[k] = v.to(device)
        self.norm_weight = self.norm_weight.to(device)

    def _place_weights(self):
        """hetero: attention weights -> GPU; norm/MLP weights stay on CPU for NPU."""
        gpu_keys = {
            "q_weight",
            "q_bias",
            "k_weight",
            "k_bias",
            "v_weight",
            "v_bias",
            "o_weight",
        }
        for layer in self.layers:
            for k, v in layer.items():
                if k in gpu_keys:
                    layer[k] = v.to("cuda")

    def _place_weights_fast(self):
        """hetero-fast: all weights on GPU + CPU copies of NPU-routed weights."""
        npu_keys = {"input_ln", "post_ln", "gate_weight", "up_weight", "down_weight"}
        self._cpu_layers = []
        for layer in self.layers:
            cpu_layer = {k: layer[k].clone() for k in npu_keys}
            self._cpu_layers.append(cpu_layer)

        self._cpu_norm_weight = self.norm_weight.clone()

        for layer in self.layers:
            for k, v in layer.items():
                layer[k] = v.to("cuda")
        self.norm_weight = self.norm_weight.to("cuda")

    def _to_gpu(self, x):
        return x.to("cuda") if x.device.type != "cuda" else x

    def _to_cpu(self, x):
        return x.cpu() if x.device.type != "cpu" else x

    # ----- op helpers (mirror gpt2 example, with PyTorch fallbacks) -----
    def _linear(self, x, weight, bias=None, backend=None):
        be = backend or self.backend
        try:
            return triton_linear(
                x,
                weight,
                bias=bias,
                backend=be,
                transform_script=self.matmul_script if be == "npu" else None,
            )
        except Exception as e:
            logger.warning(f"Triton linear failed ({e}), falling back to PyTorch")
            x_f32 = x.to(torch.float32)
            w_f32 = weight.to(torch.float32)
            out = x_f32 @ w_f32.t()
            if bias is not None:
                out = out + bias.to(torch.float32)
            return out.to(torch.bfloat16) if be == "gpu" else out

    def _rmsnorm(self, x, weight, backend=None):
        be = backend or self.backend
        try:
            return triton_rmsnorm(
                x,
                weight,
                eps=self.rms_eps,
                backend=be,
                transform_script=self.rmsnorm_script if be == "npu" else None,
            )
        except Exception as e:
            logger.warning(f"Triton rmsnorm failed ({e}), falling back to PyTorch")
            x_f32 = x.to(torch.float32)
            var = x_f32.pow(2).mean(-1, keepdim=True)
            out = x_f32 * torch.rsqrt(var + self.rms_eps) * weight.to(torch.float32)
            return out.to(torch.bfloat16) if be == "gpu" else out

    def _swiglu(self, gate, up, backend=None):
        be = backend or self.backend
        try:
            return triton_swiglu(
                gate,
                up,
                backend=be,
                silu_script=self.elem_script if be == "npu" else None,
                mul_script=self.add_script if be == "npu" else None,
            )
        except Exception as e:
            logger.warning(f"Triton swiglu failed ({e}), falling back to PyTorch")
            g = gate.to(torch.float32)
            u = up.to(torch.float32)
            out = (g * torch.sigmoid(g)) * u
            return out.to(torch.bfloat16) if be == "gpu" else out

    def _add(self, a, b, backend=None):
        be = backend or self.backend
        try:
            if be == "npu":
                # Residual adds stay in f32 on-device: bf16-native accumulation
                # compounds across the residual stream over all layers.
                return triton_add(
                    a,
                    b,
                    backend=be,
                    transform_script=self.add_f32_script,
                    f32=True,
                )
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
        """Multi-head GQA self-attention with RoPE and optional KV cache.

        x arrives on CUDA in every backend; all attention ops run on GPU.
        """
        B, S, D = x.shape

        # Attention runs on GPU in every backend (the fused kernel is GPU-only and
        # there is no NPU attention path), so the qkv/o_proj matmuls must too
        attn_be = self.op_backend["qkv_linear"] if self.op_backend else "gpu"
        proj_be = self.op_backend["attn_proj"] if self.op_backend else "gpu"

        # Fused QKV projection: q/k/v share input x, so concat their weights and
        # biases into one (n_head+2*n_kv_head)*hd linear -> one GPU launch instead
        # of three, then split. The fused weight is built once per layer (cached
        # on the layer dict; the matmul weight cache keys on tensor id).
        if "qkv_weight" not in layer:
            layer["qkv_weight"] = torch.cat(
                [layer["q_weight"], layer["k_weight"], layer["v_weight"]], dim=0
            )
            layer["qkv_bias"] = torch.cat(
                [layer["q_bias"], layer["k_bias"], layer["v_bias"]], dim=0
            )
        qkv = self._linear(x, layer["qkv_weight"], layer["qkv_bias"], backend=attn_be)
        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

        # (B, S, n_head*hd) -> (B, n_head, S, hd)
        q = q.reshape(B, S, self.n_head, self.head_dim).transpose(1, 2)
        k_new = k.reshape(B, S, self.n_kv_head, self.head_dim).transpose(1, 2)
        v_new = v.reshape(B, S, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE on q and k (on GPU)
        q, k_new = apply_rope(
            q, k_new, self.rope_cos, self.rope_sin, pos_offset=pos_offset
        )

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
            new_kv_cache = None

        # Expand KV heads to match Q heads (GQA -> MHA layout for the kernel)
        if self.n_rep > 1:
            k_exp = k_full.repeat_interleave(self.n_rep, dim=1)
            v_exp = v_full.repeat_interleave(self.n_rep, dim=1)
        else:
            k_exp = k_full
            v_exp = v_full

        scale = 1.0 / math.sqrt(self.head_dim)
        q_3d = q.reshape(B * self.n_head, S, self.head_dim).contiguous()
        k_3d = k_exp.reshape(B * self.n_head, total_len, self.head_dim).contiguous()
        v_3d = v_exp.reshape(B * self.n_head, total_len, self.head_dim).contiguous()
        is_causal = S > 1
        attn_output = triton_fused_attention(
            q_3d,
            k_3d,
            v_3d,
            scale=scale,
            causal=is_causal,
            pos_offset=pos_offset,
        ).reshape(B, self.n_head, S, self.head_dim)

        attn_output = attn_output.transpose(1, 2).reshape(B, S, self.n_embd)

        proj = self._linear(attn_output, layer["o_weight"], None, backend=proj_be)
        return proj, new_kv_cache

    def forward(self, input_ids, kv_caches=None, pos_offset=0):
        B, S = input_ids.shape
        hetero = self._is_hetero
        decode_gpu = self.backend == "hetero-fast" and S == 1

        if self.backend == "gpu" and input_ids.device.type != "cuda":
            input_ids = input_ids.to("cuda")

        x = self.embed_tokens[input_ids]  # (B, S, n_embd) bf16

        # Bare rmsnorm(x_out) carried from the previous layer's fused tail (the
        # folded rmsnorm_in); None means "compute rmsnorm_in standalone" (layer 0
        # or after an unfused layer). Local -> auto-resets each forward.
        pending_xnorm = None
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            logger.debug(f"Layer {i}/{self.n_layer}")

            if decode_gpu:
                if x.device.type != "cuda":
                    x = self._to_gpu(x)
                with self.timer.track("rmsnorm"):
                    x_norm = self._rmsnorm(x, layer["input_ln"], backend="gpu")
                layer_cache = kv_caches[i] if kv_caches else None
                with self.timer.track("attention"):
                    attn_out, new_cache = self._attention(
                        x_norm, layer, kv_cache=layer_cache, pos_offset=pos_offset
                    )
                new_kv_caches.append(new_cache)
                with self.timer.track("add1"):
                    x = self._add(x, attn_out, backend="gpu")
                with self.timer.track("rmsnorm"):
                    x_norm = self._rmsnorm(x, layer["post_ln"], backend="gpu")
                with self.timer.track("mlp_gate"):
                    gate = self._linear(
                        x_norm, layer["gate_weight"], None, backend="gpu"
                    )
                with self.timer.track("mlp_up"):
                    up = self._linear(x_norm, layer["up_weight"], None, backend="gpu")
                with self.timer.track("swiglu"):
                    h = self._swiglu(gate, up, backend="gpu")
                with self.timer.track("mlp_down"):
                    mlp_out = self._linear(h, layer["down_weight"], None, backend="gpu")
                with self.timer.track("add2"):
                    x = self._add(x, mlp_out, backend="gpu")

            elif hetero:
                npu_w = self._cpu_layers[i] if hasattr(self, "_cpu_layers") else layer
                ln_be = self.op_backend["rmsnorm"]
                with self.timer.track("rmsnorm"):
                    if pending_xnorm is None:
                        # layer 0 (or after an unfused layer): standalone rmsnorm_in
                        x_norm = self._rmsnorm(x, npu_w["input_ln"], backend=ln_be)
                    else:
                        # bare rmsnorm(x) came from the previous layer's fused
                        # tail; apply this layer's input_ln gamma (f32) here. Same
                        # math as the standalone NPU rmsnorm_in (bare f32 * gamma).
                        gamma_in = npu_w["input_ln"].to(torch.float32)
                        x_norm = (torch.from_numpy(pending_xnorm) * gamma_in).reshape(
                            x.shape[0], x.shape[1], self.n_embd
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
                gate_be = self.op_backend["mlp_gate"]
                up_be = self.op_backend["mlp_up"]
                swiglu_be = self.op_backend["swiglu"]
                down_be = self.op_backend["mlp_down"]
                # Fused fast path: ONE load_pdi ELF for the whole per-layer NPU
                # tail -- add1 -> rmsnorm(post_ln) -> gate/up -> silu*mul -> down
                # -> add2 -- when all those ops route to NPU and the row count
                # fits a single M-block. gamma folded into gate/up weights keeps
                # the in-chain rmsnorm bare. run() takes the pre-attn hidden x and
                # attn_out and returns x + attn_out + mlp(rmsnorm(x + attn_out)).
                _fused_ok = (
                    self._fused_mlp is not None
                    and gate_be == "npu"
                    and up_be == "npu"
                    and swiglu_be == "npu"
                    and down_be == "npu"
                    and add_be == "npu"
                    and x.reshape(-1, self.n_embd).shape[0] <= self._fused_mlp.BM
                )
                if _fused_ok:
                    with self.timer.track("mlp_fused"):
                        x, pending_xnorm = self._fused_mlp.run(
                            i,
                            x,
                            attn_out,
                            npu_w["gate_weight"],
                            npu_w["up_weight"],
                            npu_w["down_weight"],
                            npu_w["post_ln"],
                        )
                else:
                    pending_xnorm = None
                    with self.timer.track("add1"):
                        x = self._add(x, attn_out, backend=add_be)
                    with self.timer.track("rmsnorm"):
                        x_norm = self._rmsnorm(x, npu_w["post_ln"], backend=ln_be)
                    with self.timer.track("mlp_gate"):
                        gate = self._linear(
                            x_norm, npu_w["gate_weight"], None, backend=gate_be
                        )
                    with self.timer.track("mlp_up"):
                        up = self._linear(
                            x_norm, npu_w["up_weight"], None, backend=up_be
                        )
                    with self.timer.track("swiglu"):
                        h = self._swiglu(gate, up, backend=swiglu_be)
                    with self.timer.track("mlp_down"):
                        mlp_out = self._linear(
                            h, npu_w["down_weight"], None, backend=down_be
                        )
                    with self.timer.track("add2"):
                        x = self._add(x, mlp_out, backend=add_be)

            else:
                with self.timer.track("rmsnorm"):
                    x_norm = self._rmsnorm(x, layer["input_ln"])
                layer_cache = kv_caches[i] if kv_caches else None
                with self.timer.track("attention"):
                    attn_out, new_cache = self._attention(
                        x_norm, layer, kv_cache=layer_cache, pos_offset=pos_offset
                    )
                new_kv_caches.append(new_cache)

                with self.timer.track("to_cpu"):
                    attn_out = self._to_cpu(attn_out)
                # Fused fast path (npu mode): ONE load_pdi ELF for the whole
                # per-layer NPU tail -- add1 -> rmsnorm(post_ln) -> gate/up ->
                # silu*mul -> down -> add2 -- when the row count fits a single
                # M-block; else the unfused path. gamma folded into gate/up
                # weights keeps the in-chain rmsnorm bare, so run() takes the
                # pre-attn hidden x and attn_out.
                _fused_ok = (
                    self._fused_mlp is not None
                    and self.backend == "npu"
                    and x.reshape(-1, self.n_embd).shape[0] <= self._fused_mlp.BM
                )
                if _fused_ok:
                    with self.timer.track("mlp_fused"):
                        # npu mode keeps standalone rmsnorm_in above; ignore the
                        # chain's folded rmsnorm_in output.
                        x, _ = self._fused_mlp.run(
                            i,
                            x,
                            attn_out,
                            layer["gate_weight"],
                            layer["up_weight"],
                            layer["down_weight"],
                            layer["post_ln"],
                        )
                else:
                    with self.timer.track("add1"):
                        x = self._add(x, attn_out)
                    with self.timer.track("rmsnorm"):
                        x_norm = self._rmsnorm(x, layer["post_ln"])
                    with self.timer.track("mlp_gate"):
                        gate = self._linear(x_norm, layer["gate_weight"])
                    with self.timer.track("mlp_up"):
                        up = self._linear(x_norm, layer["up_weight"])
                    with self.timer.track("swiglu"):
                        h = self._swiglu(gate, up)
                    with self.timer.track("mlp_down"):
                        mlp_out = self._linear(h, layer["down_weight"])
                    with self.timer.track("add2"):
                        x = self._add(x, mlp_out)

        # Final RMSNorm
        if hetero and not decode_gpu:
            with self.timer.track("norm"):
                ln_be = self.op_backend["rmsnorm"]
                nw = (
                    self._cpu_norm_weight
                    if hasattr(self, "_cpu_norm_weight")
                    else self.norm_weight
                )
                x = self._rmsnorm(x, nw, backend=ln_be)
        elif hetero:
            with self.timer.track("norm"):
                x = self._rmsnorm(x, self.norm_weight, backend="gpu")
        else:
            with self.timer.track("norm"):
                x = self._rmsnorm(x, self.norm_weight)

        # LM head (tied embeddings): x @ embed_tokens^T
        with self.timer.track("lm_head"):
            if self._lm_head is not None:
                logits = (
                    x.to(device="cuda", dtype=torch.float32) @ self._lm_head.t()
                ).cpu()
            else:
                logits = x.to(torch.float32) @ self.embed_tokens.to(torch.float32).t()
                if logits.device.type == "cuda":
                    logits = logits.cpu()

        return logits, new_kv_caches

    def _allocate_kv_caches(
        self, batch_size, max_seq_len, device, dtype=torch.bfloat16
    ):
        caches = []
        for _ in range(self.n_layer):
            cache_k = torch.zeros(
                batch_size,
                self.n_kv_head,
                max_seq_len,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
            cache_v = torch.zeros(
                batch_size,
                self.n_kv_head,
                max_seq_len,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
            caches.append((cache_k, cache_v, 0))
        return caches

    def generate(
        self, input_ids, max_new_tokens=20, eos_id=151645, progress_callback=None
    ):
        B, prompt_len = input_ids.shape
        total_seq_len = prompt_len + max_new_tokens
        device = input_ids.device
        if self.backend in ("gpu", "hetero", "hetero-fast"):
            device = "cuda"

        generated_ids = []
        timing = {"prefill_ms": 0, "decode_times_ms": []}

        kv_caches = self._allocate_kv_caches(B, total_seq_len, device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits, kv_caches = self.forward(input_ids, kv_caches=kv_caches)
        t1 = time.perf_counter()
        timing["prefill_ms"] = (t1 - t0) * 1000

        next_token = torch.argmax(logits[0, -1]).item()
        generated_ids.append(next_token)
        pos_offset = prompt_len
        if progress_callback is not None:
            progress_callback(len(generated_ids), max_new_tokens)

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

            if next_token == eos_id:
                break

        return generated_ids, timing
