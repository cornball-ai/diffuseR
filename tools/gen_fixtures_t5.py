# Generate T5 encoder parity fixtures for the R port.
#
# Uses HF transformers' T5EncoderModel (Apache-2.0) with a tiny
# random-init config, plus an exact integer fixture for the relative
# position bucketing. Run via tools/gen_fixtures.sh; never executed at
# package test/run time.

import os
import shutil

import torch
from safetensors.torch import save_file
from transformers import T5Config, T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Attention

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
CKPT_DIR = os.path.join(OUT_DIR, "t5_tiny_ckpt")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(45)
fx = {}

# --- relative position bucketing (exact integer parity) -----------------------
rel = torch.arange(-200, 201).unsqueeze(0)
buckets = T5Attention._relative_position_bucket(
    rel, bidirectional=True, num_buckets=32, max_distance=128
)
fx["rel_positions"] = rel
fx["rel_buckets"] = buckets

# --- tiny encoder ---------------------------------------------------------------
config = T5Config(
    vocab_size=100,
    d_model=16,
    d_kv=4,
    num_heads=4,
    d_ff=32,
    num_layers=2,
    feed_forward_proj="gated-gelu",
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    layer_norm_epsilon=1e-6,
    dropout_rate=0.0,
)
model = T5EncoderModel(config)
model.eval()

# Padded batch, and (matching FLUX) NO attention mask
input_ids = torch.tensor([
    [5, 23, 61, 7, 19, 88, 42, 3, 1, 0, 0, 0],
    [33, 14, 2, 71, 1, 0, 0, 0, 0, 0, 0, 0],
])
with torch.no_grad():
    out = model(input_ids=input_ids).last_hidden_state

fx["input_ids"] = input_ids
fx["out"] = out
fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "t5_flux.safetensors"))
print(f"wrote {len(fx)} tensors to {OUT_DIR}/t5_flux.safetensors")

# Checkpoint dir in the transformers layout for the loader test
if os.path.isdir(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
model.save_pretrained(CKPT_DIR)
print(f"wrote checkpoint to {CKPT_DIR}: {sorted(os.listdir(CKPT_DIR))}")
