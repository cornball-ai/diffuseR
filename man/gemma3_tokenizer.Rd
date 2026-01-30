% tinyrox says don't edit this manually, but it can't stop you!
\name{gemma3_tokenizer}
\alias{gemma3_tokenizer}
\title{Gemma3 Tokenizer}
\usage{
gemma3_tokenizer(tokenizer_path)
}
\arguments{
\item{tokenizer_path}{Character. Path to tokenizer directory or tokenizer.json file.}
}
\value{
A gemma3_tokenizer object (extends bpe_tokenizer).
}
\description{
Native R tokenizer for Gemma3 using BPE.
Loads from HuggingFace tokenizer.json format.
}
