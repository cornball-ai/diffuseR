# Generate FLUX 3 reference-image preprocessing fixtures: the
# single-reference "fill" path (canvas rounding, reflect / symmetric
# padding, crop box) on small synthetic images.
#
# Uses diffusers Flux3ImageProcessor (Apache-2.0). Run via
# tools/gen_fixtures.sh.

import os
import sys

import numpy as np
import PIL.Image
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.modular_pipelines.flux3.image_processor import Flux3ImageProcessor  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(55)
proc = Flux3ImageProcessor()

fx = {}
# (height, width): padded with reflect; a side <= 16 switches to symmetric
for name, (h, w) in {"a": (70, 100), "b": (12, 20), "c": (64, 128)}.items():
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    image, kwargs = proc.preprocess(PIL.Image.fromarray(arr), resize_mode="fill")
    fx[f"{name}_input"] = torch.from_numpy(arr.copy())
    fx[f"{name}_output"] = image.float()
    fx[f"{name}_crop"] = torch.tensor(kwargs["reference_crop"])

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "flux3_preprocess.safetensors"),
          metadata={"purpose": "diffuseR FLUX 3 test fixture"})
print(f"wrote {len(fx)} tensors: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in fx.items()))
