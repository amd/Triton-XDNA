# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""`transformers`' own Gemma4 vision tower, on our de-tiled weights.

This is the oracle `gemma4_vision_gate.py` diffs against. It is worth being
precise about what it does and does not prove.

It runs **the same weights** through **a different implementation**. So it
catches everything about the forward: a wrong RoPE split, a transposed
projection, a dropped clip, a pooler keyed off the wrong axis. It cannot catch
a de-tile that is wrong in a way both sides would share, because there is only
one de-tile -- that is `llm_q4nx/test_gemma4_vision_q4nx.py`'s job, and it
checks against the bytes in the file rather than against any forward.

The two together cover it: the bytes are read correctly, and the forward built
on them agrees with upstream's.

`Gemma4MultimodalEmbedder` (the weightless norm and the projection into the
language-width space) lives on `Gemma4Model`, not on `Gemma4VisionModel`,
so it is constructed separately here and applied after -- which is also the
order `Gemma4Model.get_image_features` applies it in.
"""

import numpy as np
import torch

import gemma4_vision_q4nx as vw


def _vision_config():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    return Gemma4VisionConfig(
        hidden_size=vw.HIDDEN,
        num_hidden_layers=vw.N_LAYERS,
        num_attention_heads=vw.N_HEADS,
        num_key_value_heads=vw.N_HEADS,
        head_dim=vw.HEAD_DIM,
        intermediate_size=vw.INTER,
        patch_size=vw.PATCH,
        pooling_kernel_size=vw.POOLING_KERNEL_SIZE,
        position_embedding_size=vw.POSITION_EMBEDDING_SIZE,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": vw.ROPE_THETA,
        },
        hidden_activation="gelu_pytorch_tanh",
        use_clipped_linears=True,
        standardize=False,
        rms_norm_eps=vw.RMS_EPS,
        attention_bias=False,
        dtype=torch.float32,
    )


def _f32(a):
    """A de-tiled weight (numpy, possibly ml_dtypes bfloat16) as an f32 tensor."""
    a = np.asarray(a)
    if str(a.dtype) == "bfloat16":
        t = torch.from_numpy(np.ascontiguousarray(a).view(np.uint16)).view(
            torch.bfloat16
        )
        return t.to(torch.float32)
    return torch.from_numpy(np.ascontiguousarray(a)).to(torch.float32)


def _load_clipped(mod, w, clip):
    """One `Gemma4ClippableLinear`: the matrix plus its four bounds.

    Our weights are `[K, out]` and `nn.Linear` wants `[out, K]`, so every
    assignment here transposes. That single `.T` is the whole of the layout
    difference between this repository and upstream.
    """
    mod.linear.weight.data.copy_(_f32(w).T)
    lo_i, hi_i, lo_o, hi_o = clip
    for name, v in (
        ("input_min", lo_i),
        ("input_max", hi_i),
        ("output_min", lo_o),
        ("output_max", hi_o),
    ):
        getattr(mod, name).data.copy_(torch.tensor(v))


def build(weights=None, model=None):
    """`(vision_model, embed_vision)`, both in eval mode on our weights."""
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4MultimodalEmbedder,
        Gemma4VisionModel,
    )

    w = weights if weights is not None else vw.load_vision(model)
    vc = _vision_config()
    vm = Gemma4VisionModel(vc).to(torch.float32).eval()

    pe = vm.patch_embedder
    pe.input_proj.weight.data.copy_(_f32(w["patch_embd"]).T)
    pe.position_embedding_table.data.copy_(_f32(w["position_table"]))

    for L, lw in enumerate(w["layers"]):
        blk = vm.encoder.layers[L]
        c = lw["clip"]
        _load_clipped(blk.self_attn.q_proj, lw["q"], c["q"])
        _load_clipped(blk.self_attn.k_proj, lw["k"], c["k"])
        _load_clipped(blk.self_attn.v_proj, lw["v"], c["v"])
        _load_clipped(blk.self_attn.o_proj, lw["o"], c["o"])
        _load_clipped(blk.mlp.gate_proj, lw["gate"], c["gate"])
        _load_clipped(blk.mlp.up_proj, lw["up"], c["up"])
        _load_clipped(blk.mlp.down_proj, lw["down"], c["down"])
        blk.self_attn.q_norm.weight.data.copy_(_f32(lw["q_norm"]))
        blk.self_attn.k_norm.weight.data.copy_(_f32(lw["k_norm"]))
        # The four norms, in upstream's names. The mapping is the one place a
        # silent swap is plausible -- `norm2` is the PRE-feedforward norm and
        # `post_attn_norm` the post-attention one, and both are [768] -- so it
        # is written as four explicit lines rather than a loop over a dict.
        blk.input_layernorm.weight.data.copy_(_f32(lw["norm1"]))
        blk.post_attention_layernorm.weight.data.copy_(_f32(lw["post_attn_norm"]))
        blk.pre_feedforward_layernorm.weight.data.copy_(_f32(lw["norm2"]))
        blk.post_feedforward_layernorm.weight.data.copy_(_f32(lw["ffn_post_norm"]))

    # Not part of the vision model upstream either: it belongs to the
    # multimodal wrapper, and `output_proj_dims` is absent from a vision config
    # so the embedder reads `hidden_size` for its input width.
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    ev = Gemma4MultimodalEmbedder(
        vc, Gemma4TextConfig(hidden_size=w["embedding_proj"].shape[1])
    )
    ev = ev.to(torch.float32).eval()
    ev.embedding_projection.weight.data.copy_(_f32(w["embedding_proj"]).T)
    return vm, ev


@torch.no_grad()
def encode(pixels, pos_ids, weights=None, model=None, built=None, trace=None):
    """The reference soft tokens, `[n_soft_tokens, out_hidden]`.

    `trace`, when given, is filled with the same stage keys
    `gemma4_vision.Gemma4VisionTower.encode` fills, so the two can be zipped.
    The per-layer states are captured with forward hooks rather than by
    re-implementing the loop -- the point of an oracle is that none of its
    forward is ours.
    """
    vm, ev = built if built is not None else build(weights, model)
    px = torch.as_tensor(np.asarray(pixels), dtype=torch.float32)[None]
    pid = torch.as_tensor(np.asarray(pos_ids), dtype=torch.long)[None]

    handles = []
    if trace is not None:

        def _embed_hook(_m, _i, out):
            trace["patch_embed"] = out[0].clone()

        handles.append(vm.patch_embedder.register_forward_hook(_embed_hook))
        for L, blk in enumerate(vm.encoder.layers):

            def _hook(_m, _i, out, L=L):
                h = out[0] if isinstance(out, tuple) else out
                trace[f"layer{L}"] = h[0].clone()

            handles.append(blk.register_forward_hook(_hook))

        def _pool_hook(_m, _i, out):
            # The pooler returns (hidden, mask) and has already applied the
            # sqrt(hidden) scale; our `_pool` returns the same pair.
            trace["pool"] = out[0][0].clone()

        handles.append(vm.pooler.register_forward_hook(_pool_hook))

    try:
        out = vm(pixel_values=px, pixel_position_ids=pid)
    finally:
        for h in handles:
            h.remove()

    # `last_hidden_state` is already stripped of the slots no patch reached, so
    # this and `trace["pool"]` differ in row count by design -- the pool stage
    # is compared before the strip, the projection after.
    soft = ev(inputs_embeds=out.last_hidden_state)
    if trace is not None:
        trace["embed_vision"] = soft.clone()
    return soft
