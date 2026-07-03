% tinyrox says don't edit this manually, but it can't stop you!
\name{encode_with_gemma3}
\alias{encode_with_gemma3}
\title{Encode text with Gemma3 for LTX-2}
\usage{
encode_with_gemma3(prompts, model = NULL, tokenizer = NULL,
                   max_sequence_length = 1024L, device = "cuda",
                   dtype = "float16", verbose = TRUE)
}
\arguments{
\item{prompts}{Character vector of prompts.}

\item{model}{Gemma3 text model (or path to load from).}

\item{tokenizer}{Gemma3 tokenizer (or path to load from).}

\item{max_sequence_length}{Integer. Maximum sequence length.}

\item{device}{Character. Device for computation.}

\item{dtype}{Character. Data type.}

\item{verbose}{Logical. Print progress.}
}
\value{
List with prompt_embeds (raw stacked hidden states,
  shape \code{[batch, seq_len, hidden_size, num_layers + 1]}) and
  prompt_attention_mask.
}
\description{
Full pipeline for encoding text prompts using Gemma3 text encoder.
Returns the raw stacked per-layer hidden states (embedding layer plus all
transformer layers) for downstream connector modules, which handle
normalization and projection themselves.
}
