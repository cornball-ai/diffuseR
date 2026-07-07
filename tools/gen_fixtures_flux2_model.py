# Generate a tiny random-init Flux2Transformer2DModel parity fixture for
# the R port, plus a save_pretrained checkpoint for loader tests.
#
# Uses the diffusers reference (Apache-2.0). Run via
# tools/gen_fixtures.sh; never executed at package test/run time.

import os
import shutil
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers import Flux2Transformer2DModel  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
CKPT_DIR = os.path.join(OUT_DIR, "flux2_tiny_ckpt")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(49)

model = Flux2Transformer2DModel(
    patch_size=1,
    in_channels=8,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=24,
    mlp_ratio=3.0,
    timestep_guidance_channels=256,
    axes_dims_rope=(2, 2, 2, 2),
    rope_theta=2000,
    guidance_embeds=False,
)
model.eval()

B, S_TXT = 2, 7
GRID_H, GRID_W = 6, 10
S_IMG = GRID_H * GRID_W

hidden = torch.randn(B, S_IMG, 8)
encoder = torch.randn(B, S_TXT, 24)
timestep = torch.tensor([1.0, 0.25])  # sigma space; model multiplies by 1000

txt_ids = torch.zeros(S_TXT, 4)
txt_ids[:, 3] = torch.arange(S_TXT)
img_ids = torch.zeros(GRID_H, GRID_W, 4)
img_ids[..., 1] = torch.arange(GRID_H)[:, None]
img_ids[..., 2] = torch.arange(GRID_W)[None, :]
img_ids = img_ids.reshape(S_IMG, 4)

with torch.no_grad():
    out = model(
        hidden_states=hidden,
        encoder_hidden_states=encoder,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=None,
        return_dict=False,
    )[0]

fx = {f"model.{k}": v for k, v in model.state_dict().items()}
fx.update(
    {
        "hidden": hidden,
        "encoder": encoder,
        "timestep": timestep,
        "txt_ids": txt_ids,
        "img_ids": img_ids,
        "out": out,
    }
)
fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "flux2_model.safetensors"),
          metadata={"purpose": "diffuseR FLUX.2 test fixture"})
print(f"wrote {len(fx)} tensors to {OUT_DIR}/flux2_model.safetensors")

if os.path.isdir(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
model.save_pretrained(CKPT_DIR, max_shard_size="30KB")
print(f"wrote checkpoint to {CKPT_DIR}: {sorted(os.listdir(CKPT_DIR))}")
