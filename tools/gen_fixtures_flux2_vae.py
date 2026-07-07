# Generate FLUX.2 VAE decode parity fixtures for the R port.
#
# Uses the diffusers AutoencoderKLFlux2 (Apache-2.0) with a tiny
# random-init config and randomized BatchNorm running statistics. Run
# via tools/gen_fixtures.sh.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers import AutoencoderKLFlux2  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(51)

vae = AutoencoderKLFlux2(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D",) * 4,
    up_block_types=("UpDecoderBlock2D",) * 4,
    block_out_channels=(8, 16, 32, 32),
    layers_per_block=2,
    latent_channels=32,
    norm_num_groups=8,
    use_quant_conv=True,
    use_post_quant_conv=True,
    sample_size=64,
)
vae.eval()

# Randomize the BN running statistics so the (de)normalization is a real
# transform in the fixtures
with torch.no_grad():
    vae.bn.running_mean.normal_(0.0, 0.5)
    vae.bn.running_var.uniform_(0.5, 2.0)

latent = torch.randn(1, 32, 4, 4)
with torch.no_grad():
    image = vae.decode(latent, return_dict=False)[0]

# Reference BN normalize/denormalize on a patchified map
patched = torch.randn(1, 128, 3, 5)
bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
bn_std = torch.sqrt(vae.bn.running_var + vae.config.batch_norm_eps).view(1, -1, 1, 1)
normalized = (patched - bn_mean) / bn_std
denormalized = normalized * bn_std + bn_mean

sd = {k: v.contiguous() for k, v in vae.state_dict().items()}
save_file(sd, os.path.join(OUT_DIR, "vae_flux2_tiny.safetensors"),
          metadata={"purpose": "diffuseR FLUX.2 test fixture"})

io = {
    "latent": latent,
    "image": image,
    "patched": patched,
    "normalized": normalized,
    "denormalized": denormalized,
    "bn_mean": vae.bn.running_mean.clone(),
    "bn_var": vae.bn.running_var.clone(),
}
io = {k: v.contiguous() for k, v in io.items()}
save_file(io, os.path.join(OUT_DIR, "vae_flux2_io.safetensors"),
          metadata={"purpose": "diffuseR FLUX.2 test fixture"})
print(f"wrote vae_flux2_tiny + {len(io)} io tensors to {OUT_DIR}")
