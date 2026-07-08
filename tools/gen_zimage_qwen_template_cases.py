# Generate Z-Image chat-template parity cases: the Qwen3 template with
# enable_thinking=True (no think block), rendered by the SHIPPED
# Tongyi-MAI/Z-Image-Turbo tokenizer_config and padded like the
# pipeline. The vocab/merges/tokenizer.json are byte-identical to
# FLUX.2-klein's (verified by blob oid), so raw BPE parity is already
# covered by qwen_tokenizer_cases.json; only the template render is
# pinned here.
#
# Writes inst/tinytest/fixtures/zimage_template_cases.json (checked in).
#
# Run:
#   uv run --no-project --with transformers --with torch \
#     --index https://download.pytorch.org/whl/cpu \
#     --index-strategy unsafe-best-match python tools/gen_zimage_qwen_template_cases.py

import json
import os

from transformers import AutoTokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIXTURE = os.path.join(ROOT, "inst", "tinytest", "fixtures", "zimage_template_cases.json")

tok = AutoTokenizer.from_pretrained("Tongyi-MAI/Z-Image-Turbo", subfolder="tokenizer")

PROMPTS = [
    "a photo of a cat",
    "emoji 🦊 and 中文字符 mixed in",
    "An astronaut riding a horse on Mars, photorealistic",
    "一幅为名为“造相「Z-IMAGE-TURBO」”的项目设计的创意海报。",
    "",
]

templated = []
for text in PROMPTS:
    messages = [{"role": "user", "content": text}]
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
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
    json.dump({"templated": templated, "meta": meta},
              f, ensure_ascii=False, indent=1)
print(f"wrote {len(templated)} templated cases to {FIXTURE}")
print("meta:", meta)
print("rendered example:", json.dumps(templated[0]["rendered"]))
