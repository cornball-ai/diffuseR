# Generate FLUX.2 phase-1 parity fixtures for the R port: 4-axis RoPE,
# position-id builders, patchify/pack/unpack chain, and empirical mu.
#
# Runs the diffusers reference (Apache-2.0) on small fixed inputs. Run
# via tools/gen_fixtures.sh; never executed at package test/run time.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed  # noqa: E402
from diffusers.pipelines.flux2.pipeline_flux2_klein import (  # noqa: E402
    Flux2KleinPipeline,
    compute_empirical_mu,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(47)
fx = {}

# --- position ids ---------------------------------------------------------------
S_TXT = 7
GRID_H, GRID_W = 6, 10  # asymmetric so an axis swap fails

dummy_txt = torch.zeros(1, S_TXT, 4)
txt_ids = Flux2KleinPipeline._prepare_text_ids(dummy_txt)[0]  # [7, 4]
fx["txt_ids"] = txt_ids.float()

lat = torch.randn(1, 128, GRID_H, GRID_W)
latent_ids = Flux2KleinPipeline._prepare_latent_ids(lat)[0]  # [60, 4]
fx["latent_ids"] = latent_ids.float()

# --- Flux2PosEmbed: full axes (32,32,32,32) theta 2000 + tiny axes -------------
ids = torch.cat([txt_ids.float(), latent_ids.float()], dim=0)  # [67, 4]
fx["ids"] = ids

cos_full, sin_full = Flux2PosEmbed(theta=2000, axes_dim=[32, 32, 32, 32])(ids)
fx["pos_full_cos"] = cos_full  # [67, 128]
fx["pos_full_sin"] = sin_full

cos_tiny, sin_tiny = Flux2PosEmbed(theta=2000, axes_dim=[2, 2, 2, 2])(ids)
fx["pos_tiny_cos"] = cos_tiny  # [67, 8]
fx["pos_tiny_sin"] = sin_tiny

# --- patchify / pack / unpack chain ----------------------------------------------
z32 = torch.randn(2, 32, 12, 20)  # unpatchified 32-channel latent
patched = Flux2KleinPipeline._patchify_latents(z32)  # [2, 128, 6, 10]
packed = Flux2KleinPipeline._pack_latents(patched)  # [2, 60, 128]
fx["z32"] = z32
fx["patched"] = patched
fx["packed"] = packed
fx["unpatched"] = Flux2KleinPipeline._unpatchify_latents(patched)

ids_b = Flux2KleinPipeline._prepare_latent_ids(patched)  # [2, 60, 4]
unpacked = Flux2KleinPipeline._unpack_latents_with_ids(
    packed, ids_b, height=6, width=10
)  # [2, 128, 6, 10]
fx["unpacked"] = unpacked

# --- empirical mu -----------------------------------------------------------------
cases = [(1024, 4), (4096, 4), (4096, 28), (256, 50), (4352, 4), (8192, 10)]
fx["mu_seq"] = torch.tensor([c[0] for c in cases], dtype=torch.float64)
fx["mu_steps"] = torch.tensor([c[1] for c in cases], dtype=torch.float64)
fx["mu_vals"] = torch.tensor(
    [compute_empirical_mu(s, n) for s, n in cases], dtype=torch.float64
)

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "rope_flux2.safetensors"),
          metadata={"purpose": "diffuseR FLUX.2 test fixture"})
print(f"wrote {len(fx)} tensors to {OUT_DIR}/rope_flux2.safetensors")
