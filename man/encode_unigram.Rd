% tinyrox says don't edit this manually, but it can't stop you!
\name{encode_unigram}
\alias{encode_unigram}
\title{Encode text with a Unigram tokenizer}
\usage{
encode_unigram(tokenizer, texts, max_length = 256L, add_eos = TRUE, pad = TRUE)
}
\arguments{
\item{tokenizer}{A \code{\link{unigram_tokenizer}}.}

\item{texts}{Character vector of prompts.}

\item{max_length}{Integer. Fixed sequence length (NULL for no
truncation/padding).}

\item{add_eos}{Logical. Append the EOS token.}

\item{pad}{Logical. Right-pad to \code{max_length}.}
}
\value{
List with \code{input_ids} and \code{attention_mask}, each an
  integer matrix [length(texts), max_length] (or ragged lists when
  \code{max_length} is NULL). Ids are 0-based (HuggingFace
  convention); add 1 for R torch embedding lookups.
}
\description{
Normalizes (strip-right, multi-space collapse, control whitespace to
space), applies the Metaspace pre-tokenizer, segments each pre-token
by Viterbi over the Unigram scores, fuses consecutive unknowns, and
appends EOS. T5 semantics: right padding with \code{<pad>} (id 0),
truncation to \code{max_length - 1} before the EOS.
}
