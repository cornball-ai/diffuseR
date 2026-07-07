# Generate Qwen2 ByteLevel-BPE tokenizer parity cases for the R port
# (FLUX.2 klein's tokenizer), plus the rendered Qwen3 chat template.
#
# Downloads the tokenizer from the ungated black-forest-labs/
# FLUX.2-klein-4B repo (a few MB). Writes:
#   - tools/cache/tokenizer_qwen.json (dev copy, gitignored)
#   - inst/tinytest/fixtures/qwen_tokenizer_cases.json (checked in)
#
# Run:
#   uv run --no-project --with transformers --with torch \
#     --index https://download.pytorch.org/whl/cpu \
#     --index-strategy unsafe-best-match python tools/gen_qwen_tokenizer_cases.py

import json
import os

from transformers import AutoTokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR = os.path.join(ROOT, "tools", "cache")
FIXTURE = os.path.join(ROOT, "inst", "tinytest", "fixtures", "qwen_tokenizer_cases.json")
os.makedirs(CACHE_DIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained("black-forest-labs/FLUX.2-klein-4B",
                                    subfolder="tokenizer")
tok.backend_tokenizer.save(os.path.join(CACHE_DIR, "tokenizer_qwen.json"))

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
    "underscores_and_snake_case",
    "a",
    "",
    "café résumé naïve",
    "em—dash and en–dash",
    "100% of $50 + €20",
    "emoji 🦊 and 中文字符 mixed in",
    "newline\nand\ttab",
    "An astronaut riding a horse on Mars, photorealistic",
    "watercolor painting of a fox in a snowy forest",
    ("The transformer architecture uses self-attention mechanisms "
     "to model long-range dependencies. ") * 3,
]

# Raw tokenizer parity (no special tokens, no template)
cases = [{"text": t, "ids": tok(t, add_special_tokens=False)["input_ids"]}
         for t in PROMPTS]

# Pipeline-style: chat template (user message, generation prompt, no
# thinking) rendered then padded/truncated exactly like the klein pipeline
templated = []
for text in [PROMPTS[0], PROMPTS[20], PROMPTS[24]]:
    messages = [{"role": "user", "content": text}]
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    enc = tok(rendered, padding="max_length", truncation=True, max_length=64)
    templated.append({
        "text": text,
        "rendered": rendered,
        "max_length": 64,
        "ids": enc["input_ids"],
        "mask": enc["attention_mask"],
    })

meta = {
    "pad_token": tok.pad_token,
    "pad_token_id": tok.pad_token_id,
    "padding_side": tok.padding_side,
}

with open(FIXTURE, "w", encoding="utf-8") as f:
    json.dump({"cases": cases, "templated": templated, "meta": meta},
              f, ensure_ascii=False, indent=1)
print(f"wrote {len(cases)} cases + {len(templated)} templated to {FIXTURE}")
print("meta:", meta)
print("rendered example:", json.dumps(templated[0]["rendered"]))
