# Generate FLUX.2 transformer-block parity fixtures for the R port.
#
# Runs the diffusers reference modules (Apache-2.0) with tiny random-init
# configs. Run via tools/gen_fixtures.sh; never executed at package
# test/run time.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.models.transformers.transformer_flux2 import (  # noqa: E402
    Flux2FeedForward,
    Flux2Modulation,
    Flux2PosEmbed,
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(48)
fx = {}

# Tiny config: dim 16 = 2 heads x head_dim 8, rope axes (2,2,2,2), mlp 3.0
DIM, HEADS, HEAD_DIM = 16, 2, 8
B, S_TXT = 2, 7
GRID_H, GRID_W = 6, 10
S_IMG = GRID_H * GRID_W  # 60

# 4-axis rope over [txt; img] ids
txt_ids = torch.zeros(S_TXT, 4)
txt_ids[:, 3] = torch.arange(S_TXT)
img_ids = torch.zeros(GRID_H, GRID_W, 4)
img_ids[..., 1] = torch.arange(GRID_H)[:, None]
img_ids[..., 2] = torch.arange(GRID_W)[None, :]
img_ids = img_ids.reshape(S_IMG, 4)
ids = torch.cat([txt_ids, img_ids], dim=0)
rope_cos, rope_sin = Flux2PosEmbed(theta=2000, axes_dim=[2, 2, 2, 2])(ids)
fx["rope_cos"] = rope_cos
fx["rope_sin"] = rope_sin

img_x = torch.randn(B, S_IMG, DIM)
txt_x = torch.randn(B, S_TXT, DIM)
joint_x = torch.randn(B, S_TXT + S_IMG, DIM)
temb = torch.randn(B, DIM)
fx["img_x"] = img_x
fx["txt_x"] = txt_x
fx["joint_x"] = joint_x
fx["temb"] = temb


def add_state(prefix, module):
    for k, v in module.state_dict().items():
        fx[f"{prefix}.{k}"] = v


# --- Flux2Modulation (2-set and 1-set) -------------------------------------------
mod2 = Flux2Modulation(DIM, mod_param_sets=2)
add_state("mod2", mod2)
with torch.no_grad():
    fx["mod2_out"] = mod2(temb)

mod1 = Flux2Modulation(DIM, mod_param_sets=1)
add_state("mod1", mod1)
with torch.no_grad():
    fx["mod1_out"] = mod1(temb)

# --- Flux2FeedForward (SwiGLU) ------------------------------------------------------
ff = Flux2FeedForward(DIM, DIM, mult=3.0)
add_state("ff", ff)
with torch.no_grad():
    fx["ff_out"] = ff(joint_x)

# --- Flux2TransformerBlock (double) ---------------------------------------------------
dbl = Flux2TransformerBlock(dim=DIM, num_attention_heads=HEADS,
                            attention_head_dim=HEAD_DIM, mlp_ratio=3.0)
add_state("dbl", dbl)
with torch.no_grad():
    mod_img = mod2(temb)
    mod_txt = mod2(temb) * 0.5  # distinct txt modulation
    enc_out, hid_out = dbl(
        hidden_states=img_x,
        encoder_hidden_states=txt_x,
        temb_mod_img=mod_img,
        temb_mod_txt=mod_txt,
        image_rotary_emb=(rope_cos, rope_sin),
    )
fx["dbl_mod_img"] = mod_img
fx["dbl_mod_txt"] = mod_txt
fx["dbl_enc_out"] = enc_out
fx["dbl_hid_out"] = hid_out

# --- Flux2SingleTransformerBlock --------------------------------------------------------
sgl = Flux2SingleTransformerBlock(dim=DIM, num_attention_heads=HEADS,
                                  attention_head_dim=HEAD_DIM, mlp_ratio=3.0)
add_state("sgl", sgl)
with torch.no_grad():
    mod_single = mod1(temb)
    joint_out = sgl(
        hidden_states=joint_x,
        encoder_hidden_states=None,
        temb_mod=mod_single,
        image_rotary_emb=(rope_cos, rope_sin),
    )
fx["sgl_mod"] = mod_single
fx["sgl_out"] = joint_out

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "dit_flux2.safetensors"),
          metadata={"purpose": "diffuseR FLUX.2 test fixture"})
print(f"wrote {len(fx)} tensors to {OUT_DIR}/dit_flux2.safetensors")
