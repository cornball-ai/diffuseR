# Generate a tiny random-init Flux3Transformer2DModel parity fixture for
# the R port (text-to-image and one-reference forwards, position ids,
# the distilled sigma schedule), plus a save_pretrained checkpoint for
# loader tests.
#
# Uses the diffusers reference (Apache-2.0). Run via
# tools/gen_fixtures.sh; never executed at package test/run time.

import os
import shutil
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers import FlowMatchEulerDiscreteScheduler, Flux3Transformer2DModel  # noqa: E402
from diffusers.modular_pipelines.flux3.before_denoise import (  # noqa: E402
    Flux3TextRoPEInputsStep,
    _prepare_image_ids,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
CKPT_DIR = os.path.join(OUT_DIR, "flux3_tiny_ckpt")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(53)

model = Flux3Transformer2DModel(
    in_channels=8,
    cond_in_channels=8,
    num_layers=1,
    num_single_layers=2,
    attention_head_dim=16,
    num_attention_heads=2,
    joint_attention_dim=32,
    timestep_channels=256,
    mlp_ratio=3.0,
    axes_dims_rope=(4, 4, 4, 4),
    rope_theta=10000,
    flow_map=True,
)
model.eval()

B, S_TXT = 2, 7
GRID_H, GRID_W = 4, 6
COND_H, COND_W = 3, 5

hidden = torch.randn(B, GRID_H * GRID_W, 8)
encoder = torch.randn(B, S_TXT, 32)
cond = torch.randn(B, COND_H * COND_W, 8)
timestep = torch.tensor([1.0, 0.4])
target_timestep = torch.tensor([0.75, 0.0])

txt_ids = Flux3TextRoPEInputsStep._prepare_text_ids(encoder)
img_ids = _prepare_image_ids(GRID_H, GRID_W, 0.5)
cond_ids = _prepare_image_ids(COND_H, COND_W, 0.5, l_coord=-1)

with torch.no_grad():
    out_t2i = model(
        hidden_states=hidden,
        encoder_hidden_states=encoder,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        target_timestep=target_timestep,
        return_dict=False,
    )[0]
    out_ref = model(
        hidden_states=hidden,
        encoder_hidden_states=encoder,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        image_cond_hidden_states=cond,
        image_cond_ids=cond_ids,
        target_timestep=target_timestep,
        return_dict=False,
    )[0]

# Distilled schedule: static shift 6.93, sigmas linspace(1, 1/16, 16)
scheduler = FlowMatchEulerDiscreteScheduler(shift=6.93)
scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / 16, 16))

fx = {f"model.{k}": v for k, v in model.state_dict().items()}
fx.update(
    {
        "hidden": hidden,
        "encoder": encoder,
        "cond": cond,
        "timestep": timestep,
        "target_timestep": target_timestep,
        "txt_ids": txt_ids,
        "img_ids": img_ids,
        "cond_ids": cond_ids,
        "out_t2i": out_t2i,
        "out_ref": out_ref,
        "timesteps": scheduler.timesteps.float(),
        "sigmas": scheduler.sigmas.float(),
    }
)
fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "flux3_model.safetensors"),
          metadata={"purpose": "diffuseR FLUX 3 test fixture"})
print(f"wrote {len(fx)} tensors to {OUT_DIR}/flux3_model.safetensors")

if os.path.isdir(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
model.save_pretrained(CKPT_DIR, max_shard_size="250KB")
print(f"wrote checkpoint to {CKPT_DIR}: {sorted(os.listdir(CKPT_DIR))}")
