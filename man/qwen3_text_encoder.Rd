% tinyrox says don't edit this manually, but it can't stop you!
\name{qwen3_text_encoder}
\alias{qwen3_text_encoder}
\title{Qwen3 Text Encoder}
\description{
Fresh R port of the Qwen3 decoder stack from HuggingFace transformers
(Apache-2.0, src/transformers/models/qwen3/), used by FLUX.2 klein as
its text encoder (Qwen3-4B: 36 layers, hidden 2560, 32 query / 8 KV
heads, head_dim 128, SwiGLU 9728, RoPE theta 1e6). The pipeline
consumes mid-stack hidden states (layers 9, 18, 27 for klein-4B)
concatenated per token, so the forward runs only as deep as the last
requested layer; the LM head is never needed (embeddings are tied).
Causal attention with the tokenizer's padding mask, matching the
reference exactly.
}
