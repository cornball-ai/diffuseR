# Generate a tiny random-init Qwen3-VL text-encoder fixture for the FLUX 3
# prompt path: a right-padded prompt window with the pipeline's explicit
# position ids (padding at position 1), hidden states stacked from the
# configured layers. Saves a save_pretrained checkpoint (vision tower
# included, so the loader's skip is exercised).
#
# Uses HF transformers' Qwen3VLForConditionalGeneration (Apache-2.0) and
# mirrors diffusers Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds
# (Apache-2.0). Run via tools/gen_fixtures.sh.

import os
import shutil

import torch
from safetensors.torch import save_file
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
CKPT_DIR = os.path.join(OUT_DIR, "flux3_qwen_tiny_ckpt")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(54)

config = Qwen3VLConfig(
    text_config=dict(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=5e6,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1],
                      "mrope_interleaved": True},
        tie_word_embeddings=True,
    ),
    vision_config=dict(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=32,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
    ),
    image_token_id=97,
    video_token_id=98,
    vision_start_token_id=95,
    vision_end_token_id=96,
    tie_word_embeddings=True,
)
model = Qwen3VLForConditionalGeneration(config)
model.eval()

# One prompt, right-padded to a window of 12 (pad id 0)
real = [5, 23, 61, 7, 19, 88, 42, 3, 9]
window = 12
input_ids = torch.tensor([real + [0] * (window - len(real))])
attention_mask = torch.tensor([[1] * len(real) + [0] * (window - len(real))])
position_ids = torch.arange(window)[None].masked_fill(attention_mask == 0, 1)

layers = (1, 2, 4)
with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, output_hidden_states=True,
                use_cache=False)
stacked = torch.stack([out.hidden_states[k] for k in layers], dim=1)
b, c, s, h = stacked.shape
stacked = stacked.permute(0, 2, 1, 3).reshape(b, s, c * h)

fx = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "position_ids": position_ids,
    "stacked": stacked,
}
fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "flux3_qwen.safetensors"),
          metadata={"purpose": "diffuseR FLUX 3 test fixture"})
print(f"wrote {len(fx)} tensors to {OUT_DIR}/flux3_qwen.safetensors")

if os.path.isdir(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
model.save_pretrained(CKPT_DIR)
print(f"wrote checkpoint to {CKPT_DIR}: {sorted(os.listdir(CKPT_DIR))}")
