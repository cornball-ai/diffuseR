% tinyrox says don't edit this manually, but it can't stop you!
\name{encode_with_t5}
\alias{encode_with_t5}
\title{Encode prompts with the T5 encoder}
\usage{
encode_with_t5(prompts, model, tokenizer, max_sequence_length = 256L,
               device = NULL)
}
\arguments{
\item{prompts}{Character vector.}

\item{model}{A \code{\link{t5_encoder}}.}

\item{tokenizer}{A \code{\link{unigram_tokenizer}}.}

\item{max_sequence_length}{Integer. Fixed token length (schnell: 256).}

\item{device}{Device for the input ids (defaults to the model's).}
}
\value{
Tensor [length(prompts), max_sequence_length, d_model].
}
\description{
Tokenizes with \code{\link{encode_unigram}} (right padding to
\code{max_sequence_length}) and runs the encoder. Matching the FLUX
reference pipeline, no attention mask is used.
}
