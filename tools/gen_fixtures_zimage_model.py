# Generate the Z-Image phase-3 fixture: a tiny random-init
# ZImageTransformer2DModel full forward (state dict + input + output),
# with both caption and image needing pad tokens.
#
# Runs the diffusers reference (Apache-2.0) on small fixed inputs. Run
# via tools/gen_fixtures.sh; never executed at package test/run time.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.models.transformers.transformer_z_image import (  # noqa: E402
    ZImageTransformer2DModel,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(83)
fx = {}

model = ZImageTransformer2DModel(
    in_channels=4,
    dim=32,
    n_layers=2,
    n_refiner_layers=1,
    n_heads=2,
    n_kv_heads=2,
    cap_feat_dim=24,
    axes_dims=[4, 6, 6],
    axes_lens=[128, 32, 32],
)
with torch.no_grad():
    for p in model.parameters():
        p.copy_(torch.randn_like(p) * 0.05)

# 12x20 latent -> 6x10 = 60 tokens, pads to 64; caption 37 pads to 64
x = torch.randn(4, 1, 12, 20)
cap = torch.randn(37, 24)
t = torch.tensor([0.4375])

with torch.no_grad():
    out = model([x], t, [cap], return_dict=False)[0][0]

for k, v in model.state_dict().items():
    fx[f"sd.{k}"] = v
fx["x"] = x
fx["cap"] = cap
fx["t"] = t
fx["out"] = out  # [4, 1, 12, 20]

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "zimage_model.safetensors"),
          metadata={"purpose": "diffuseR Z-Image test fixture"})
print(f"wrote {len(fx)} tensors to {OUT_DIR}/zimage_model.safetensors")
