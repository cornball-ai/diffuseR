% tinyrox says don't edit this manually, but it can't stop you!
\name{encode_gemma4}
\alias{encode_gemma4}
\title{Encode prompts with the Gemma 4 tokenizer}
\usage{
encode_gemma4(tokenizer, texts, max_length = 1024L)
}
\arguments{
\item{tokenizer}{A \code{\link{gemma4_tokenizer}}.}

\item{texts}{Character vector of prompts.}

\item{max_length}{Integer. Fixed sequence length, or NULL for ragged
unpadded output.}
}
\value{
List with \code{input_ids} and \code{attention_mask}: integer
  matrices [length(texts), max_length] (ragged lists when
  \code{max_length} is NULL). Ids are 0-based.
}
\description{
Tokenizes, applies the post-processor's special tokens (none for
Gemma 4), truncates to \code{max_length} keeping the start, and pads
on the left with \code{<pad>}, as the LTX pipelines do.
}
