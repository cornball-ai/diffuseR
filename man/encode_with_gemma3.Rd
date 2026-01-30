% tinyrox says don't edit this manually, but it can't stop you!
\name{encode_with_gemma3}
\alias{encode_with_gemma3}
\title{Encode text with Gemma3 for LTX-2}
\usage{
encode_with_gemma3(
  prompts,
  model = NULL,
  tokenizer = NULL,
  max_sequence_length = 1024L,
  scale_factor = 8,
  device = "cuda",
  dtype = "float16",
  verbose = TRUE
)
}
\arguments{
\item{prompts}{Character vector of prompts.}

\item{model}{Gemma3 text model (or path to load from).}

\item{tokenizer}{Gemma3 tokenizer (or path to load from).}

\item{max_sequence_length}{Integer. Maximum sequence length.}

\item{scale_factor}{Numeric. Scale factor for packing (default 8).}

\item{device}{Character. Device for computation.}

\item{dtype}{Character. Data type.}

\item{verbose}{Logical. Print progress.}
}
\value{
List with prompt_embeds and prompt_attention_mask.
}
\description{
Full pipeline for encoding text prompts using Gemma3 text encoder.
Returns packed embeddings ready for LTX-2 connectors.
}
