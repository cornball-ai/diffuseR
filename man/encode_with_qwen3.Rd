% tinyrox says don't edit this manually, but it can't stop you!
\name{encode_with_qwen3}
\alias{encode_with_qwen3}
\title{Encode prompts with the Qwen3 encoder for FLUX.2}
\usage{
encode_with_qwen3(prompts, model, tokenizer, max_sequence_length = 512L,
                  out_layers = c(9L, 18L, 27L), device = NULL)
}
\arguments{
\item{prompts}{Character vector.}

\item{model}{A \code{\link{qwen3_encoder}}.}

\item{tokenizer}{A \code{\link{qwen_bpe_tokenizer}}.}

\item{max_sequence_length}{Integer. Fixed token length (klein: 512).}

\item{out_layers}{Integer vector. Hidden-state layers (klein-4B:
9, 18, 27).}

\item{device}{Device for the input ids (defaults to the model's).}
}
\value{
Tensor [length(prompts), max_sequence_length,
  3 * hidden_size].
}
\description{
Tokenizes with the chat template, runs the encoder with the padding
mask, and concatenates the requested mid-stack hidden states per
token, matching Flux2KleinPipeline._get_qwen3_prompt_embeds.
}
