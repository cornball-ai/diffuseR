# Generate a tiny random-init FluxTransformer2DModel parity fixture for
# the R port, plus a sharded save_pretrained checkpoint used by the
# checkpoint-loading and quantization tests.
#
# Uses the diffusers reference implementation (Apache-2.0). Run via
# tools/gen_fixtures.sh; never executed at package test/run time.

import os
import shutil
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers import FluxTransformer2DModel  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
CKPT_DIR = os.path.join(OUT_DIR, "flux_tiny_ckpt")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(44)

model = FluxTransformer2DModel(
    patch_size=1,
    in_channels=4,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=16,
    pooled_projection_dim=12,
    axes_dims_rope=(2, 2, 4),
)
model.eval()

B, S_TXT = 2, 7
GRID_H, GRID_W = 8, 12
S_IMG = GRID_H * GRID_W

hidden = torch.randn(B, S_IMG, 4)
encoder = torch.randn(B, S_TXT, 16)
pooled = torch.randn(B, 12)
timestep = torch.tensor([1.0, 0.25])  # sigma space; model multiplies by 1000
txt_ids = torch.zeros(S_TXT, 3)
img_ids = torch.zeros(GRID_H, GRID_W, 3)
img_ids[..., 1] = torch.arange(GRID_H)[:, None]
img_ids[..., 2] = torch.arange(GRID_W)[None, :]
img_ids = img_ids.reshape(S_IMG, 3)

with torch.no_grad():
    out = model(
        hidden_states=hidden,
        encoder_hidden_states=encoder,
        pooled_projections=pooled,
        timestep=timestep,
        txt_ids=txt_ids,
        img_ids=img_ids,
        return_dict=False,
    )[0]

fx = {f"model.{k}": v for k, v in model.state_dict().items()}
fx.update(
    {
        "hidden": hidden,
        "encoder": encoder,
        "pooled": pooled,
        "timestep": timestep,
        "txt_ids": txt_ids,
        "img_ids": img_ids,
        "out": out,
    }
)
fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "flux_model.safetensors"))
print(f"wrote {len(fx)} tensors to {OUT_DIR}/flux_model.safetensors")

# Sharded checkpoint in the real HF layout (config.json + shards +
# diffusion_pytorch_model.safetensors.index.json) for loader tests
if os.path.isdir(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
model.save_pretrained(CKPT_DIR, max_shard_size="30KB")
print(f"wrote sharded checkpoint to {CKPT_DIR}: {sorted(os.listdir(CKPT_DIR))}")
