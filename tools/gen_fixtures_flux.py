# Generate FLUX parity fixtures for the R port.
#
# Runs the diffusers reference implementation (Apache-2.0) on small fixed
# inputs and saves {inputs, expected outputs} as safetensors fixtures that
# the R tinytest suite compares against. Run via tools/gen_fixtures.sh;
# never executed at package test/run time.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.models.embeddings import apply_rotary_emb  # noqa: E402
from diffusers.models.transformers.transformer_flux import FluxPosEmbed  # noqa: E402
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)
fx = {}

# --- latent image ids ---------------------------------------------------------
# Asymmetric grid (8 rows x 12 cols) so a row/col swap fails the test.
img_ids = FluxPipeline._prepare_latent_image_ids(
    batch_size=1, height=8, width=12, device="cpu", dtype=torch.float32
)
fx["img_ids"] = img_ids  # [96, 3]

# --- FluxPosEmbed: full-size axes (16, 56, 56) --------------------------------
txt_ids = torch.zeros(7, 3)
ids = torch.cat([txt_ids, img_ids], dim=0)  # [103, 3]
fx["ids"] = ids

pos_full = FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56])
cos_full, sin_full = pos_full(ids)
fx["pos_full_cos"] = cos_full  # [103, 128]
fx["pos_full_sin"] = sin_full

# --- FluxPosEmbed: tiny axes (2, 2, 4) used by the block-level fixtures -------
pos_tiny = FluxPosEmbed(theta=10000, axes_dim=[2, 2, 4])
cos_tiny, sin_tiny = pos_tiny(ids)
fx["pos_tiny_cos"] = cos_tiny  # [103, 8]
fx["pos_tiny_sin"] = sin_tiny

# --- apply_rotary_emb on [B, H, S, D] (sequence_dim=2, unbind_dim=-1) ----------
B, H, S, D = 2, 4, 103, 8
x = torch.randn(B, H, S, D)
fx["rot_x"] = x
fx["rot_out"] = apply_rotary_emb(x, (cos_tiny, sin_tiny), sequence_dim=2)

# fp16 dtype preservation through apply
x16 = torch.randn(B, H, S, D, dtype=torch.float16)
fx["rot_x_f16"] = x16
fx["rot_out_f16"] = apply_rotary_emb(x16, (cos_tiny, sin_tiny), sequence_dim=2)

# --- latent pack / unpack -------------------------------------------------------
# Latent [2, 16, 8, 12] corresponds to a 64x96 pixel image (vae_scale_factor 8).
lat = torch.randn(2, 16, 8, 12)
packed = FluxPipeline._pack_latents(lat, 2, 16, 8, 12)
fx["pack_in"] = lat
fx["pack_out"] = packed  # [2, 24, 64]
fx["unpack_out"] = FluxPipeline._unpack_latents(packed, height=64, width=96, vae_scale_factor=8)

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "rope_flux.safetensors"))
print(f"wrote {len(fx)} tensors to {OUT_DIR}/rope_flux.safetensors")
