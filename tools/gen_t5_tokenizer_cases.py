# Generate T5 Unigram tokenizer parity cases for the R port.
#
# Downloads spiece.model from the public google/t5-v1_1-xxl repo (791 KB)
# and converts it to the fast tokenizer.json - the same conversion that
# produced FLUX.1-schnell's tokenizer_2/tokenizer.json. Writes:
#   - tools/cache/tokenizer_t5.json (dev copy, gitignored)
#   - inst/tinytest/fixtures/t5_tokenizer_cases.json (checked in)
#
# Run:
#   uv run --no-project --with transformers --with sentencepiece \
#     --with protobuf --with torch --index https://download.pytorch.org/whl/cpu \
#     --index-strategy unsafe-best-match python tools/gen_t5_tokenizer_cases.py

import json
import os

from huggingface_hub import hf_hub_download
from transformers import T5TokenizerFast

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR = os.path.join(ROOT, "tools", "cache")
FIXTURE = os.path.join(ROOT, "inst", "tinytest", "fixtures", "t5_tokenizer_cases.json")
os.makedirs(CACHE_DIR, exist_ok=True)

spiece = hf_hub_download("google/t5-v1_1-xxl", "spiece.model")
tok = T5TokenizerFast(vocab_file=spiece)
tok.backend_tokenizer.save(os.path.join(CACHE_DIR, "tokenizer_t5.json"))

PROMPTS = [
    "a photo of a cat",
    "A sunset over mountains, ultra detailed, 8k",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "it's a beautiful day; isn't it?",
    "3.14159 and 2,000,000 dollars",
    "state-of-the-art text-to-image generation",
    "  leading spaces",
    "trailing spaces   ",
    "double  and   triple   spaces",
    "UPPERCASE lowercase MiXeD",
    "email@example.com and https://example.org/path?q=1",
    'quotes "double" and \'single\'',
    "(parentheses) [brackets] {braces}",
    "semi;colon co:lon sla/sh back\\slash",
    "underscores_and_snake_case",
    "a",
    "",
    "café résumé naïve",
    "em—dash and en–dash",
    "100% of $50 + €20",
    "An astronaut riding a horse on Mars, photorealistic",
    "watercolor painting of a fox in a snowy forest",
    "the mitochondria is the powerhouse of the cell",
    ("The transformer architecture uses self-attention mechanisms "
     "to model long-range dependencies in sequences. ") * 12,
]

cases = []
for text in PROMPTS:
    ids = tok(text, add_special_tokens=True)["input_ids"]
    cases.append({"text": text, "ids": ids})

# Padding/truncation behavior at a small max_length
padded = []
for text in [PROMPTS[0], PROMPTS[3], PROMPTS[24]]:
    enc = tok(text, max_length=16, padding="max_length", truncation=True)
    padded.append({
        "text": text,
        "max_length": 16,
        "ids": enc["input_ids"],
        "mask": enc["attention_mask"],
    })

with open(FIXTURE, "w", encoding="utf-8") as f:
    json.dump({"cases": cases, "padded": padded}, f, ensure_ascii=False, indent=1)
print(f"wrote {len(cases)} cases + {len(padded)} padded cases to {FIXTURE}")
