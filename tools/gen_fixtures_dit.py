# Generate DiT parity fixtures for the LTX-2.3 R port.
# Tiny LTX-2.3-configured reference transformer: weights + inputs + outputs.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.models.transformers.transformer_ltx2 import (  # noqa: E402
    LTX2AdaLayerNormSingle,
    LTX2Attention,
    LTX2VideoTransformer3DModel,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(7)
fx = {}

# --- Gated attention with split RoPE ------------------------------------------
attn = LTX2Attention(
    query_dim=16, heads=2, kv_heads=2, dim_head=8,
    rope_type="split", apply_gated_attention=True,
).eval()
for name, p in attn.state_dict().items():
    fx[f"attn.{name}"] = p

B, S = 2, 6
x = torch.randn(B, S, 16)
cos = torch.randn(B, 2, S, 4)
sin = torch.randn(B, 2, S, 4)
mask = torch.zeros(B, 1, S)
mask[:, :, -2:] = -10000.0
fx["attn_x"] = x
fx["attn_cos"] = cos
fx["attn_sin"] = sin
fx["attn_mask"] = mask
with torch.no_grad():
    fx["attn_out"] = attn(x, attention_mask=mask, query_rotary_emb=(cos, sin))
    fx["attn_out_nomask"] = attn(x, query_rotary_emb=(cos, sin))

# --- AdaLN single with 9 mod params --------------------------------------------
ada = LTX2AdaLayerNormSingle(16, num_mod_params=9).eval()
for name, p in ada.state_dict().items():
    fx[f"ada.{name}"] = p
t = torch.tensor([700.0, 300.0])
with torch.no_grad():
    ada_out, ada_emb = ada(t, batch_size=2, hidden_dtype=torch.float32)
fx["ada_t"] = t
fx["ada_out"] = ada_out
fx["ada_emb"] = ada_emb

# --- Tiny full model (LTX-2.3 flags) --------------------------------------------
model = LTX2VideoTransformer3DModel(
    in_channels=4,
    out_channels=4,
    num_attention_heads=2,
    attention_head_dim=8,
    cross_attention_dim=16,
    audio_in_channels=4,
    audio_out_channels=4,
    audio_num_attention_heads=2,
    audio_attention_head_dim=4,
    audio_cross_attention_dim=8,
    num_layers=2,
    gated_attn=True,
    cross_attn_mod=True,
    audio_gated_attn=True,
    audio_cross_attn_mod=True,
    use_prompt_embeddings=False,
    perturbed_attn=True,
    rope_type="split",
).eval()
sd = model.state_dict()
for name, p in sd.items():
    fx[f"model.{name}"] = p
print(f"tiny model params: {len(sd)}")

num_frames, height, width = 2, 3, 4
n_video = num_frames * height * width
n_audio = 5
n_text = 7
hidden = torch.randn(1, n_video, 4)
audio_hidden = torch.randn(1, n_audio, 4)
enc = torch.randn(1, n_text, 16)
audio_enc = torch.randn(1, n_text, 8)
enc_mask = torch.ones(1, n_text)
enc_mask[:, -2:] = 0
timestep = torch.tensor([700.0])
sigma = torch.tensor([0.7])

fx["m_hidden"] = hidden
fx["m_audio_hidden"] = audio_hidden
fx["m_enc"] = enc
fx["m_audio_enc"] = audio_enc
fx["m_enc_mask"] = enc_mask
fx["m_timestep"] = timestep
fx["m_sigma"] = sigma

with torch.no_grad():
    out = model(
        hidden_states=hidden,
        audio_hidden_states=audio_hidden,
        encoder_hidden_states=enc,
        audio_encoder_hidden_states=audio_enc,
        timestep=timestep,
        sigma=sigma,
        encoder_attention_mask=enc_mask,
        audio_encoder_attention_mask=enc_mask,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=24.0,
        audio_num_frames=n_audio,
        use_cross_timestep=True,
        return_dict=False,
    )
fx["m_out_video"] = out[0]
fx["m_out_audio"] = out[1]

# Same forward with modalities isolated (no a2v/v2a cross attention)
with torch.no_grad():
    out_iso = model(
        hidden_states=hidden,
        audio_hidden_states=audio_hidden,
        encoder_hidden_states=enc,
        audio_encoder_hidden_states=audio_enc,
        timestep=timestep,
        sigma=sigma,
        num_frames=num_frames,
        height=height,
        width=width,
        audio_num_frames=n_audio,
        isolate_modalities=True,
        use_cross_timestep=True,
        return_dict=False,
    )
fx["m_out_video_iso"] = out_iso[0]
fx["m_out_audio_iso"] = out_iso[1]

# STG: perturb all batch elements at block 1
with torch.no_grad():
    out_stg = model(
        hidden_states=hidden,
        audio_hidden_states=audio_hidden,
        encoder_hidden_states=enc,
        audio_encoder_hidden_states=audio_enc,
        timestep=timestep,
        sigma=sigma,
        num_frames=num_frames,
        height=height,
        width=width,
        audio_num_frames=n_audio,
        spatio_temporal_guidance_blocks=[1],
        use_cross_timestep=True,
        return_dict=False,
    )
fx["m_out_video_stg"] = out_stg[0]
fx["m_out_audio_stg"] = out_stg[1]

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "dit_ltx23.safetensors"))
print(f"wrote {len(fx)} tensors to {OUT_DIR}/dit_ltx23.safetensors")
