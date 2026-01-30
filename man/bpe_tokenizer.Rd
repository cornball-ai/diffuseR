% tinyrox says don't edit this manually, but it can't stop you!
\name{bpe_tokenizer}
\alias{bpe_tokenizer}
\title{BPE Tokenizer}
\usage{
bpe_tokenizer(tokenizer_path)
}
\arguments{
\item{tokenizer_path}{Path to tokenizer.json or directory containing it.}
}
\value{
A bpe_tokenizer object.
}
\description{
Native R implementation of Byte-Pair Encoding tokenizer.
Loads from HuggingFace tokenizer.json format.
}
