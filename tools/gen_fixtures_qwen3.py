# Generate Qwen3 encoder parity fixtures for the R port.
#
# Uses HF transformers' Qwen3ForCausalLM (Apache-2.0) with a tiny
# random-init config; captures the mid-stack hidden states the FLUX.2
# klein pipeline consumes. Run via tools/gen_fixtures.sh.

import os
import shutil

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
CKPT_DIR = os.path.join(OUT_DIR, "qwen3_tiny_ckpt")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(50)

config = Qwen3Config(
    vocab_size=100,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    rope_theta=1e6,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    attention_bias=False,
    max_position_embeddings=512,
)
model = Qwen3ForCausalLM(config)
model.eval()

# Right-padded batch with attention mask (the klein pipeline passes the
# mask to the text encoder)
input_ids = torch.tensor([
    [5, 23, 61, 7, 19, 88, 42, 3, 9, 0, 0, 0],
    [33, 14, 2, 71, 55, 0, 0, 0, 0, 0, 0, 0],
])
attention_mask = torch.tensor([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
])

with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=False)

# Pipeline-style stack of mid-stack layers (1, 2, 3) -> [B, L, 3*hidden]
layers = (1, 2, 3)
stacked = torch.stack([out.hidden_states[k] for k in layers], dim=1)
stacked = stacked.permute(0, 2, 1, 3).reshape(input_ids.shape[0],
                                              input_ids.shape[1],
                                              3 * config.hidden_size)

fx = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "h1": out.hidden_states[1],
    "h2": out.hidden_states[2],
    "h3": out.hidden_states[3],
    "stacked": stacked,
}
fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "qwen3_flux2.safetensors"),
          metadata={"purpose": "diffuseR FLUX.2 test fixture"})
print(f"wrote {len(fx)} tensors to {OUT_DIR}/qwen3_flux2.safetensors")

if os.path.isdir(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
model.save_pretrained(CKPT_DIR)
print(f"wrote checkpoint to {CKPT_DIR}: {sorted(os.listdir(CKPT_DIR))}")
