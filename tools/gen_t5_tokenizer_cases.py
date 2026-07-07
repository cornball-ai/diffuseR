# Generate T5 Unigram tokenizer parity cases for the R port.
#
# Uses FLUX.1-schnell's shipped tokenizer_2/tokenizer.json from the
# HuggingFace cache when available (the authoritative artifact:
# Metaspace prepend_scheme "always"); otherwise converts spiece.model
# from the public google/t5-v1_1-xxl repo and patches the prepend scheme
# to match. Writes:
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
from transformers import PreTrainedTokenizerFast

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR = os.path.join(ROOT, "tools", "cache")
FIXTURE = os.path.join(ROOT, "inst", "tinytest", "fixtures", "t5_tokenizer_cases.json")
os.makedirs(CACHE_DIR, exist_ok=True)
TOK_JSON = os.path.join(CACHE_DIR, "tokenizer_t5.json")

def is_real_tokenizer(path):
    try:
        with open(path) as f:
            return len(json.load(f)["model"]["vocab"]) > 30000
    except Exception:
        return False


if not is_real_tokenizer(TOK_JSON):
    import glob

    shipped = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/"
        "snapshots/*/tokenizer_2/tokenizer.json"
    ))
    if shipped:
        with open(shipped[0], "rb") as f_in, open(TOK_JSON, "wb") as f_out:
            f_out.write(f_in.read())
        print(f"using shipped tokenizer: {shipped[0]}")
    else:
        # Conversion fallback: requires transformers<5 (v5 dropped the
        # sentencepiece slow->fast converter and emits a 104-token stub)
        from transformers import T5TokenizerFast

        spiece = hf_hub_download("google/t5-v1_1-xxl", "spiece.model")
        T5TokenizerFast(vocab_file=spiece).backend_tokenizer.save(TOK_JSON)
        tj = json.load(open(TOK_JSON))
        assert len(tj["model"]["vocab"]) > 30000, \
            "conversion produced a stub vocab; pin transformers<5"
        for part in ("pre_tokenizer", "decoder"):
            if tj.get(part, {}).get("type") == "Metaspace":
                tj[part]["prepend_scheme"] = "always"
        json.dump(tj, open(TOK_JSON, "w"), ensure_ascii=False)
        print("using converted spiece.model, prepend_scheme patched to 'always'")
else:
    print(f"using existing {TOK_JSON}")

tok = PreTrainedTokenizerFast(
    tokenizer_file=TOK_JSON, pad_token="<pad>", eos_token="</s>",
    unk_token="<unk>",
)

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
