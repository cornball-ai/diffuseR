% tinyrox says don't edit this manually, but it can't stop you!
\name{tokenizer_qwen}
\alias{tokenizer_qwen}
\title{Qwen2 Byte-Level BPE Tokenizer}
\description{
Pure R implementation of the Qwen2 tokenizer (HuggingFace
tokenizer.json, BPE model with ByteLevel pre-tokenization), as used by
FLUX.2 klein's Qwen3 text encoder. Text is split with the GPT-4-style
regex, each pre-token's UTF-8 bytes are mapped through the GPT-2
byte-to-unicode table, and rank-based BPE merges produce the ids.
Added tokens (\code{<|im_start|>}, \code{<think>}, ...) are split out
literally before byte-level encoding.
}
\details{
Limitation: the NFC normalizer is not applied (base R has no NFC);
input is assumed to already be NFC, which holds for ordinary text.

}
