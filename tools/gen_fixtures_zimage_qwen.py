# Generate the Z-Image phase-4 fixture: a tiny random-init Qwen3Model
# pinning the hidden_states[-2] index convention and the
# mask-slice-to-variable-length caption used by the Z-Image pipeline.
#
# Run via tools/gen_fixtures.sh; never executed at package test/run time.

import os
import sys

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3Model

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(89)
fx = {}

N_LAYERS = 4
config = Qwen3Config(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=N_LAYERS,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    rope_theta=1e6,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
)
model = Qwen3Model(config)
with torch.no_grad():
    for p in model.parameters():
        p.copy_(torch.randn_like(p) * 0.05)
model.eval()

# 9 real tokens + 3 right pads
input_ids = torch.tensor([[5, 17, 99, 3, 42, 8, 120, 64, 7, 0, 0, 0]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])

with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True)

# hidden_states has N_LAYERS + 1 entries; [-2] = after layer N_LAYERS - 1
penult = out.hidden_states[-2]
assert torch.equal(penult, out.hidden_states[N_LAYERS - 1])

for k, v in model.state_dict().items():
    fx[f"sd.{k}"] = v
fx["input_ids"] = input_ids.float()
fx["attention_mask"] = attention_mask.float()
fx["penult"] = penult                              # [1, 12, 32]
fx["penult_sliced"] = penult[0][attention_mask[0].bool()]  # [9, 32]

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "zimage_qwen.safetensors"),
          metadata={"purpose": "diffuseR Z-Image test fixture"})
print(f"wrote {len(fx)} tensors to {OUT_DIR}/zimage_qwen.safetensors")
