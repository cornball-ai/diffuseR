% tinyrox says don't edit this manually, but it can't stop you!
\name{encode_bpe}
\alias{encode_bpe}
\title{Encode text to token IDs}
\usage{
encode_bpe(
  tokenizer,
  text,
  add_special_tokens = TRUE,
  max_length = NULL,
  padding = "none",
  truncation = FALSE,
  return_tensors = "list"
)
}
\arguments{
\item{tokenizer}{A bpe_tokenizer object.}

\item{text}{Character string or vector to encode.}

\item{add_special_tokens}{Logical. Add BOS/EOS tokens.}

\item{max_length}{Integer. Maximum sequence length (NULL for no limit).}

\item{padding}{Character. Padding strategy: "none", "max_length", or "longest".}

\item{truncation}{Logical. Truncate to max_length.}

\item{return_tensors}{Character. Return type: "list" or "pt" (torch tensors).}
}
\value{
List with input_ids and attention_mask.
}
\description{
Encode text to token IDs
}
