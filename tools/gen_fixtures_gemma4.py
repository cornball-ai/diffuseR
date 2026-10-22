# Generate Gemma 4 (LTX-2.5 text encoder) parity fixtures for the R port.
#
# Uses HF transformers' Gemma4UnifiedTextModel (Apache-2.0) with a tiny
# random-init config that keeps every LTX-2.5 feature: 5:1-style sliding
# and full layers, full layers with a doubled head_dim, one KV head and
# k == v, proportional RoPE with partial_rotary_factor 0.25, v_norm and
# per-layer layer_scalar. All weights are perturbed off their init so
# norm scales (x * w, not x * (1 + w)) and layer scalars are exercised.
#
# Also builds a tiny tokenizer.json with the Gemma 4 pipeline (Replace
# " " -> U+2581, byte fallback, no BOS template) and records HF tokenizers
# ids for a set of prompts. Run via tools/gen_fixtures.sh (needs the
# tokenizers wheel, which transformers depends on).

import json
import os

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import Gemma4UnifiedTextConfig, Gemma4UnifiedTextModel

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(54)

layer_types = ["sliding_attention", "sliding_attention", "full_attention",
               "sliding_attention", "sliding_attention", "full_attention"]
text_config = dict(
    vocab_size=300,  # covers the tiny tokenizer below, for the encode test
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=len(layer_types),
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    global_head_dim=16,
    num_global_key_value_heads=1,
    attention_k_eq_v=True,
    layer_types=layer_types,
    sliding_window=3,
    rope_parameters={
        "full_attention": {"rope_type": "proportional",
                           "partial_rotary_factor": 0.25,
                           "rope_theta": 1_000_000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
    },
    hidden_activation="gelu_pytorch_tanh",
    rms_norm_eps=1e-6,
    max_position_embeddings=512,
    num_kv_shared_layers=0,
    hidden_size_per_layer_input=0,
    enable_moe_block=False,
    use_double_wide_mlp=False,
    pad_token_id=0,
)
config = Gemma4UnifiedTextConfig(**text_config)
config._attn_implementation = "eager"
model = Gemma4UnifiedTextModel(config)
model.eval()

with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("norm.weight"):
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        else:
            p.add_(0.05 * torch.randn_like(p))
    for layer in model.layers:
        layer.layer_scalar.copy_(0.5 + torch.rand_like(layer.layer_scalar))

# Left-padded batch (LTX pads on the left), longer than the sliding
# window so the window mask binds
input_ids = torch.tensor([
    [5, 23, 61, 7, 19, 88, 42, 3, 9],
    [0, 0, 0, 33, 14, 2, 71, 55, 12],
])
attention_mask = (input_ids != 0).long()
attention_mask[0] = 1

with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=False)
hidden = torch.stack(out.hidden_states, dim=-1)  # [B, S, H, L + 1]

fx = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "hidden_states": hidden,
    "last_hidden_state": out.last_hidden_state,
}
fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "gemma4_text.safetensors"),
          metadata={"purpose": "diffuseR Gemma 4 test fixture"})
print(f"wrote {len(fx)} tensors to {OUT_DIR}/gemma4_text.safetensors")

# Tiny checkpoint in the layout of the LTX-2.5 bundled encoder file:
# model.* text weights, the connectors' text projection, a vision tensor
# the text-only loader must skip, and gemma_config metadata
ckpt = {f"model.{k}": v.detach().clone().contiguous()
        for k, v in model.state_dict().items()}
ckpt["text_embedding_projection.video_aggregate_embed.weight"] = torch.randn(8, 32 * 7)
ckpt["text_embedding_projection.video_aggregate_embed.bias"] = torch.randn(8)
ckpt["vision_model.patch_dense.weight"] = torch.randn(4, 4)
gemma_config = {"architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "gemma_version": "gemma4-12b-ltx-v1",
                "text_config": text_config}

# --- Tokenizer ------------------------------------------------------------------

# Pieces: specials at the Gemma ids, the 256 byte-fallback tokens, single
# characters, and merged pieces reachable through the merge list
specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
chars = list("abcdehilmnorstw.") + ["▁"]
merges = [("▁", "t"), ("h", "e"), ("▁t", "he"), ("l", "l"),
          ("e", "ll"), ("o", "w"), ("▁", "c"), ("▁c", "a"),
          ("a", "t"), ("▁ca", "t"), ("▁", "s"), ("i", "t"),
          ("▁s", "it"), ("▁", "o"), ("n", "▁")]
vocab = {}
for s in specials:
    vocab[s] = len(vocab)
for b in range(256):
    vocab[f"<0x{b:02X}>"] = len(vocab)
for c in chars:
    vocab.setdefault(c, len(vocab))
for a, b in merges:
    vocab.setdefault(a + b, len(vocab))

tok_json = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [{"id": vocab[s], "content": s, "single_word": False,
                      "lstrip": False, "rstrip": False, "normalized": False,
                      "special": True} for s in specials],
    "normalizer": {"type": "Replace", "pattern": {"String": " "},
                   "content": "▁"},
    "pre_tokenizer": {"type": "Split", "pattern": {"String": " "},
                      "behavior": "MergedWithPrevious", "invert": False},
    "post_processor": {"type": "TemplateProcessing",
                       "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                       "pair": [{"Sequence": {"id": "A", "type_id": 0}},
                                {"Sequence": {"id": "B", "type_id": 1}}],
                       "special_tokens": {}},
    "decoder": None,
    "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>",
              "continuing_subword_prefix": None, "end_of_word_suffix": None,
              "fuse_unk": True, "byte_fallback": True, "ignore_merges": False,
              "vocab": vocab, "merges": [[a, b] for a, b in merges]},
}
tok_path = os.path.join(OUT_DIR, "gemma4_tiny_tokenizer.json")
with open(tok_path, "w", encoding="utf-8") as f:
    json.dump(tok_json, f, ensure_ascii=False)

# The LTX file carries its tokenizer.json as a uint8 tensor
with open(tok_path, "rb") as f:
    tok_bytes = f.read()
ckpt["tokenizer_json"] = torch.tensor(list(tok_bytes), dtype=torch.uint8)
save_file(ckpt, os.path.join(OUT_DIR, "gemma4_tiny_ckpt.safetensors"),
          metadata={"gemma_config": json.dumps(gemma_config), "format": "pt"})
print(f"wrote {len(ckpt)} tensors to {OUT_DIR}/gemma4_tiny_ckpt.safetensors")

tok = Tokenizer.from_file(tok_path)
prompts = ["the cat sat", "hello cat", " the  cat ", "tell", "caté ok",
           "the<eos>cat", "sit on the mat."]
cases = [{"text": p, "ids": tok.encode(p, add_special_tokens=True).ids}
         for p in prompts]
with open(os.path.join(OUT_DIR, "gemma4_tokenizer_cases.json"), "w",
          encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False)
print(f"wrote {len(cases)} tokenizer cases")

# Optional: ids from the real LTX-2.5 tokenizer, for an at-home R test
real = os.environ.get("GEMMA4_TOKENIZER_JSON")
if real and os.path.exists(real):
    rtok = Tokenizer.from_file(real)
    rprompts = [
        "A golden retriever runs across a sunlit meadow, slow motion.",
        "Close-up: rain drips off a neon sign that reads \"OPEN 24/7\" — cinematic.",
        "一只猫在窗台上睡觉。",
        "  two  spaces and a tab\there ",
        "Emoji test \U0001F680\U0001F319 and café crème brûlée",
    ]
    rcases = [{"text": p, "ids": rtok.encode(p, add_special_tokens=True).ids}
              for p in rprompts]
    with open(os.path.join(OUT_DIR, "gemma4_real_tokenizer_cases.json"), "w",
              encoding="utf-8") as f:
        json.dump(rcases, f, ensure_ascii=False)
    print(f"wrote {len(rcases)} real-tokenizer cases")
