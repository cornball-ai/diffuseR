% tinyrox says don't edit this manually, but it can't stop you!
\name{tokenize_gemma3}
\alias{tokenize_gemma3}
\title{Tokenize text for Gemma3}
\usage{
tokenize_gemma3(
  tokenizer,
  text,
  max_length = 1024L,
  padding = "max_length",
  return_tensors = "pt"
)
}
\arguments{
\item{tokenizer}{Gemma3 tokenizer object.}

\item{text}{Character vector of prompts.}

\item{max_length}{Integer. Maximum sequence length.}

\item{padding}{Character. Padding strategy ("left", "right", "max_length", "none").}

\item{return_tensors}{Character. Return type ("list" or "pt" for torch tensors).}
}
\value{
List with input_ids and attention_mask.
}
\description{
Tokenize text for Gemma3
}
