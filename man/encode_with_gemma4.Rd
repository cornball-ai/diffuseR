% tinyrox says don't edit this manually, but it can't stop you!
\name{encode_with_gemma4}
\alias{encode_with_gemma4}
\title{Encode prompts with Gemma 4 for LTX-2.5}
\usage{
encode_with_gemma4(
  prompts,
  model,
  tokenizer,
  max_sequence_length = 1024L,
  device = "cuda",
  verbose = TRUE
)
}
\arguments{
\item{prompts}{Character vector.}

\item{model}{A \code{\link{gemma4_text_model}}. A pinned CPU-resident
model moves to \code{device} for the encode and back afterwards.}

\item{tokenizer}{A \code{\link{gemma4_tokenizer}}.}

\item{max_sequence_length}{Integer. Padded token length.}

\item{device}{Character. Compute device.}

\item{verbose}{Logical.}
}
\value{
List with \code{prompt_embeds} \code{[batch, seq_len,
  hidden_size, num_layers + 1]} and \code{prompt_attention_mask}
  \code{[batch, seq_len]}, the same layout as
  \code{\link{encode_with_gemma3}}.
}
\description{
The LTX-2.5 text-encoder step: strips each prompt, tokenizes with left
padding to \code{max_sequence_length}, and returns every hidden state
(embedding output plus each layer, the last final-normed) stacked on a
new last dimension for the connectors.
}
