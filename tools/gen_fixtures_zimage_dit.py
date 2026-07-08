# Generate Z-Image phase-2 parity fixtures: transformer block (modulated
# and unmodulated), final layer, timestep embedder, and cap embedder,
# all tiny random-init.
#
# Runs the diffusers reference (Apache-2.0) on small fixed inputs. Run
# via tools/gen_fixtures.sh; never executed at package test/run time.

import os
import sys

import torch
import torch.nn as nn
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.models.normalization import RMSNorm  # noqa: E402
from diffusers.models.transformers.transformer_z_image import (  # noqa: E402
    FinalLayer,
    RopeEmbedder,
    TimestepEmbedder,
    ZImageTransformerBlock,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(59)
fx = {}

DIM, HEADS, HEAD_DIM = 32, 2, 16
SEQ = 24

# Shared inputs: token positions mixing a cap ramp and an image grid
rope = RopeEmbedder(theta=256.0, axes_dims=[4, 6, 6], axes_lens=[64, 32, 32])
ids = torch.cat(
    [
        torch.tensor([[i + 1, 0, 0] for i in range(8)], dtype=torch.int32),
        torch.stack(
            torch.meshgrid(
                torch.arange(9, 10, dtype=torch.int32),
                torch.arange(4, dtype=torch.int32),
                torch.arange(4, dtype=torch.int32),
                indexing="ij",
            ),
            dim=-1,
        ).flatten(0, 2),
    ]
)
assert ids.shape == (SEQ, 3)
freqs_cis = rope(ids)  # [24, 8] complex64

x = torch.randn(1, SEQ, DIM)
adaln = torch.randn(1, DIM)

fx["ids"] = ids.float()
fx["freqs_cos"] = freqs_cis.real.repeat_interleave(2, dim=-1)  # [24, 16]
fx["freqs_sin"] = freqs_cis.imag.repeat_interleave(2, dim=-1)
fx["x"] = x
fx["adaln"] = adaln

# --- modulated block (noise refiner / main trunk) --------------------------------
torch.manual_seed(61)
blk = ZImageTransformerBlock(0, DIM, HEADS, HEADS, norm_eps=1e-5, qk_norm=True,
                             modulation=True)
with torch.no_grad():
    for p in blk.parameters():
        p.copy_(torch.randn_like(p) * 0.05)
    out = blk(x, None, freqs_cis.unsqueeze(0), adaln)
for k, v in blk.state_dict().items():
    fx[f"mod.{k}"] = v
fx["mod_out"] = out

# --- unmodulated block (context refiner) ------------------------------------------
torch.manual_seed(67)
ublk = ZImageTransformerBlock(0, DIM, HEADS, HEADS, norm_eps=1e-5, qk_norm=True,
                              modulation=False)
with torch.no_grad():
    for p in ublk.parameters():
        p.copy_(torch.randn_like(p) * 0.05)
    uout = ublk(x, None, freqs_cis.unsqueeze(0))
for k, v in ublk.state_dict().items():
    fx[f"unmod.{k}"] = v
fx["unmod_out"] = uout

# --- final layer -------------------------------------------------------------------
torch.manual_seed(71)
fin = FinalLayer(DIM, 2 * 2 * 1 * 16)
with torch.no_grad():
    for p in fin.parameters():
        p.copy_(torch.randn_like(p) * 0.05)
    fout = fin(x, c=adaln)
for k, v in fin.state_dict().items():
    fx[f"final.{k}"] = v
fx["final_out"] = fout

# --- timestep embedder --------------------------------------------------------------
torch.manual_seed(73)
temb = TimestepEmbedder(min(DIM, 256), mid_size=48)
with torch.no_grad():
    for p in temb.parameters():
        p.copy_(torch.randn_like(p) * 0.05)
    t_in = torch.tensor([0.0, 437.5, 1000.0])
    t_out = temb(t_in)
for k, v in temb.state_dict().items():
    fx[f"temb.{k}"] = v
fx["temb_in"] = t_in
fx["temb_out"] = t_out

# --- cap embedder (RMSNorm + Linear, Sequential keys 0/1) ----------------------------
torch.manual_seed(79)
CAP_DIM = 24
cap_embedder = nn.Sequential(RMSNorm(CAP_DIM, eps=1e-5), nn.Linear(CAP_DIM, DIM, bias=True))
with torch.no_grad():
    for p in cap_embedder.parameters():
        p.copy_(torch.randn_like(p) * 0.05)
    cap_in = torch.randn(5, CAP_DIM)
    cap_out = cap_embedder(cap_in)
for k, v in cap_embedder.state_dict().items():
    fx[f"cap.{k}"] = v
fx["cap_in"] = cap_in
fx["cap_out"] = cap_out

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "dit_zimage.safetensors"),
          metadata={"purpose": "diffuseR Z-Image test fixture"})
print(f"wrote {len(fx)} tensors to {OUT_DIR}/dit_zimage.safetensors")
