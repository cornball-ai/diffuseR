% tinyrox says don't edit this manually, but it can't stop you!
\name{decode_bpe}
\alias{decode_bpe}
\title{Decode token IDs to text}
\usage{
decode_bpe(tokenizer, ids, skip_special_tokens = TRUE)
}
\arguments{
\item{tokenizer}{A bpe_tokenizer object.}

\item{ids}{Integer vector or matrix of token IDs.}

\item{skip_special_tokens}{Logical. Skip special tokens in output.}
}
\value{
Character string or vector.
}
\description{
Decode token IDs to text
}
