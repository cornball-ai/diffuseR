# Generate connector parity fixtures for the LTX-2.3 R port.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(11)
fx = {}

conn = LTX2TextConnectors(
    caption_channels=8,
    text_proj_in_factor=3,
    video_connector_num_attention_heads=2,
    video_connector_attention_head_dim=8,
    video_connector_num_layers=2,
    video_connector_num_learnable_registers=4,
    video_gated_attn=True,
    audio_connector_num_attention_heads=2,
    audio_connector_attention_head_dim=4,
    audio_connector_num_layers=2,
    audio_connector_num_learnable_registers=4,
    audio_gated_attn=True,
    rope_type="split",
    per_modality_projections=True,
    video_hidden_dim=16,
    audio_hidden_dim=8,
    proj_bias=True,
).eval()

for name, p in conn.state_dict().items():
    fx[f"conn.{name}"] = p

B, S, C, L = 2, 8, 8, 3
states = torch.randn(B, S, C, L)
mask = torch.ones(B, S)
mask[0, :3] = 0  # left padding on first batch element
mask[1, :1] = 0

fx["c_states"] = states
fx["c_mask"] = mask

with torch.no_grad():
    video_emb, audio_emb, out_mask = conn(states, mask)
fx["c_video_emb"] = video_emb
fx["c_audio_emb"] = audio_emb
fx["c_out_mask"] = out_mask.to(torch.float32)

# 3D (flattened) input variant should give identical output
with torch.no_grad():
    video_emb3, audio_emb3, _ = conn(states.flatten(2), mask)
fx["c_video_emb3"] = video_emb3

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "connectors_ltx23.safetensors"))
print(f"wrote {len(fx)} tensors to {OUT_DIR}/connectors_ltx23.safetensors")
