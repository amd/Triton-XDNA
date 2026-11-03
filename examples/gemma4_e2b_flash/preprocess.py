# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Image file -> the patches and grid positions the vision tower takes.

`transformers` 5.8 ships Gemma4's own image processor, so this is a wrapper and
not an implementation. It exists for two reasons: to pick the PIL backend over
the torchvision one (the torchvision module imports `torchvision` at module
scope, which is not a dependency of this repository), and to give the rest of
the example a shape it can reason about -- one image, two arrays, a count --
rather than a `BatchFeature`.

The processor's stock defaults already match this checkpoint's `vision_config`
exactly: patch 16, pooling kernel 3, 280 soft tokens, rescale 1/255, mean 0,
std 1, no normalization. They are passed explicitly anyway, from `config`, so
that a checkpoint whose config moves produces a disagreement here rather than
silently preprocessing for the wrong model.

What comes back, for a 448x448 input:

* `pixels` `[2520, 768]` -- each row one 16x16 RGB patch, flattened as
  `(channel, row, col)`. The image is first resized preserving aspect ratio to
  the largest size that fits the patch budget and divides `pooling * patch`,
  which is 768x768 = 48x48 = 2304 patches; the remaining 216 rows are padding.
* `pos_ids` `[2520, 2]` -- the `(x, y)` grid coordinate of each patch, `(-1, -1)`
  on the padding rows. Both the position embedding and the pooler key off this,
  so it is not derivable from the row order and has to travel with the pixels.
* `n_soft_tokens` -- 256 here, the count of pooled tokens the prompt must carry
  placeholders for.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402


def processor(max_soft_tokens=None):
    """Gemma4's image processor, PIL backend, configured from `config`.

    `max_soft_tokens` is the **resolution knob**, and the single biggest lever
    on how much work everything downstream does. It is a budget, not a count:
    the image is resized to the largest size that fits it, so the tokens
    actually produced are the largest supported tier at or below it. The
    processor accepts 70, 140, 280, 560 and 1120.

    It costs more than linearly. On a 448x448 input:

        budget   patches   soft tokens   prompt
            70       630            64       84
           140      1260           121      141
           280      2520           256      276
           560      5040           529      549

    -- and the tower's attention is quadratic in patches, so the step from 70
    to 280 is 4x the patches and ~16x the attention.

    `config.MAX_SOFT_TOKENS` (280) is the checkpoint's own
    `default_output_length` and is the default here. **FastFlowLM ships the
    70 tier**, measured: its prefill log reads "1/1 with 84 tokens" on this
    image, which is the first row above exactly. So a comparison against FLM
    that leaves this at the default is comparing a 4x heavier workload, and
    will look four-times-worse for reasons that have nothing to do with the
    implementation.
    """
    from transformers.models.gemma4.image_processing_pil_gemma4 import (
        Gemma4ImageProcessorPil,
    )

    return Gemma4ImageProcessorPil(
        patch_size=config.VISION_PATCH,
        pooling_kernel_size=config.POOLING_KERNEL_SIZE,
        max_soft_tokens=(
            config.MAX_SOFT_TOKENS if max_soft_tokens is None else max_soft_tokens
        ),
    )


def load_image(path, proc=None, max_soft_tokens=None, image_size=None):
    """`path` -> `(pixels [P, 768], pos_ids [P, 2], n_soft_tokens)`.

    `image_size` squares the input to `image_size x image_size` **before** the
    processor sees it. The production hybrid pipeline normalizes to RGB at
    **256x256**, and this is how to match it.

    It does not change the shape of anything downstream. At the 70 budget both
    a 256x256 and a 448x448 input resize to 384x384 -> 24x24 -> 576 patches ->
    64 soft tokens, because the processor picks the largest size that fits the
    patch budget and divides `pooling * patch = 48`. What it changes is the
    resampling: from 256 the processor scales UP to 384, from 448 it scales
    DOWN. Same tensors, different pixels in them, so it is worth matching when
    comparing outputs with a pipeline that fixes 256 -- and worth knowing that
    it is not worth matching for a timing comparison, where it is a no-op.
    """
    from PIL import Image

    proc = proc or processor(max_soft_tokens)
    with Image.open(path) as im:
        im = im.convert("RGB")
        if image_size:
            im = im.resize((image_size, image_size), Image.BICUBIC)
        out = proc(images=[im], return_tensors=None)

    pixels = np.asarray(out["pixel_values"][0], dtype=np.float32)
    pos_ids = np.asarray(out["image_position_ids"][0], dtype=np.int64)
    # `num_soft_tokens_per_image` is per image even for one image, and comes
    # back as a 0-d array wrapping a list on this backend. Reduced to an int
    # here so no caller has to know that.
    n_soft = int(np.asarray(out["num_soft_tokens_per_image"]).reshape(-1)[0])

    if pixels.shape[1] != 3 * config.VISION_PATCH**2:
        raise ValueError(
            f"patches are {pixels.shape[1]} wide, expected "
            f"3*{config.VISION_PATCH}^2 = {3 * config.VISION_PATCH**2}"
        )
    valid = int((~(pos_ids == -1).all(axis=-1)).sum())
    k2 = config.POOLING_KERNEL_SIZE**2
    if valid != n_soft * k2:
        # The pooler divides by a fixed k^2, so a patch count that is not an
        # exact multiple would average some tiles against the wrong divisor.
        raise ValueError(
            f"{valid} valid patches do not pool to {n_soft} tokens at k={k2}"
        )
    return pixels, pos_ids, n_soft
