# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""An image turn's inputs: the image's patches, and the prompt that holds it.

Both Gemma4 sizes share the tower and the tokenizer's image tokens, so nothing
here depends on which model the turn is for.

`load_image` wraps transformers' own Gemma4 image processor, with the PIL
backend (its torchvision one imports `torchvision`, which is not a dependency
here). What comes back, for a 448x448 input at the default budget:

* `pixels` `[2520, 768]` -- each row one 16x16 RGB patch, flattened as
  `(channel, row, col)`. The image is first resized, preserving its aspect
  ratio, to the largest size that fits the patch budget and divides
  `pooling * patch`: 768x768 = 48x48 = 2304 patches; the other 216 rows are
  padding.
* `pos_ids` `[2520, 2]` -- each patch's `(x, y)` grid coordinate, `(-1, -1)` on
  the padding rows. The position embedding and the pooler both key off it.
* `n_soft` -- 256 here, the pooled tokens the prompt must hold placeholders for.
"""

import numpy as np

import gemma4_vision_q4nx as vw

#: The checkpoint's `default_output_length`.
MAX_SOFT_TOKENS = 280

#: The budgets the processor accepts. FastFlowLM ships 70.
SOFT_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


def processor(max_soft_tokens=None):
    """Gemma4's image processor, PIL backend.

    `max_soft_tokens` is the resolution budget, and the biggest lever on how
    much work everything downstream does. The image is resized to the largest
    size that fits it, so the tokens produced are the largest tier at or below
    it. On a 448x448 input:

        budget   patches   soft tokens   prompt
            70       630            64       84
           140      1260           121      141
           280      2520           256      276
           560      5040           529      549

    The tower's attention is quadratic in patches, so 70 -> 280 is 4x the
    patches and ~16x the attention. Compare against FastFlowLM at 70.
    """
    from transformers.models.gemma4.image_processing_pil_gemma4 import (
        Gemma4ImageProcessorPil,
    )

    return Gemma4ImageProcessorPil(
        patch_size=vw.PATCH,
        pooling_kernel_size=vw.POOLING_KERNEL_SIZE,
        max_soft_tokens=MAX_SOFT_TOKENS if max_soft_tokens is None else max_soft_tokens,
    )


def load_image(path, proc=None, max_soft_tokens=None, image_size=None):
    """`path` -> `(pixels [P, 768], pos_ids [P, 2], n_soft)`.

    `image_size` squares the input to `image_size x image_size` before the
    processor sees it, which only changes the resampling: a 256x256 and a
    448x448 input resize to the same patch grid at a given budget.
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
    # Per image, and a 0-d array wrapping a list on this backend.
    n_soft = int(np.asarray(out["num_soft_tokens_per_image"]).reshape(-1)[0])

    if pixels.shape[1] != 3 * vw.PATCH**2:
        raise ValueError(f"patches are {pixels.shape[1]} wide, expected 3*{vw.PATCH}^2")
    valid = int((~(pos_ids == -1).all(axis=-1)).sum())
    k2 = vw.POOLING_KERNEL_SIZE**2
    if valid != n_soft * k2:
        # The pooler divides by a fixed k^2.
        raise ValueError(f"{valid} valid patches do not pool to {n_soft} tokens")
    return pixels, pos_ids, n_soft


def build_prompt(tokenizer, text, n_soft, system=None):
    """Render the chat template with one image, and expand its placeholder.

    Returns `(ids, image_mask, ple_ids)`.

    The template emits one image token; `Gemma4Processor.replace_image_token`
    grows it into `boi + image * n_soft + eoi`, and so does this. `ple_ids`
    holds the pad token at the image positions: upstream gathers the
    per-layer embedding table with those, not with the placeholder's id.

    `system`, when given, is a system turn ahead of the image turn -- the
    prefix a reusing prefill caches across turns.
    """
    msg = []
    if system:
        msg.append({"role": "system", "content": [{"type": "text", "text": system}]})
    msg.append(
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
    )
    rendered = tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    )
    tok = tokenizer.image_token
    if rendered.count(tok) != 1:
        raise RuntimeError(
            f"the chat template emitted {rendered.count(tok)} image tokens, "
            "expected exactly 1"
        )
    rendered = rendered.replace(
        tok, f"{tokenizer.boi_token}{tok * n_soft}{tokenizer.eoi_token}"
    )
    # The template already emits <bos>.
    ids = tokenizer.encode(rendered, add_special_tokens=False)

    image_id = tokenizer.convert_tokens_to_ids(tok)
    mask = [i == image_id for i in ids]
    if sum(mask) != n_soft:
        raise RuntimeError(
            f"{sum(mask)} image placeholders survived tokenization, expected "
            f"{n_soft}"
        )
    pad = tokenizer.pad_token_id
    ple_ids = [pad if m else i for i, m in zip(ids, mask)]
    return ids, mask, ple_ids


def shapes_image(path, size=448):
    """Write the two-shape test image: a red square upper-left, a blue circle
    lower-right, on a light background. A correct description names both
    shapes, both colors and both positions."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (size, size), (245, 245, 245))
    d = ImageDraw.Draw(im)
    s = size / 448
    d.rectangle([60 * s, 60 * s, 200 * s, 200 * s], fill=(220, 30, 30))
    d.ellipse([250 * s, 250 * s, 400 * s, 400 * s], fill=(30, 80, 220))
    im.save(path)
    return path


#: Each group must appear in the shapes image's description. Positions are one
#: group because the model phrases them freely. mlir-air's gate, verbatim.
SHAPES_TERMS = [
    ("red", ("red",)),
    ("square", ("square", "rectangle")),
    ("blue", ("blue",)),
    ("circle", ("circle", "round")),
    ("position", ("left", "right", "upper", "lower", "top", "bottom")),
]


def shapes_missing(text):
    """The `SHAPES_TERMS` groups a description fails to mention."""
    low = text.lower()
    return [name for name, alts in SHAPES_TERMS if not any(a in low for a in alts)]
