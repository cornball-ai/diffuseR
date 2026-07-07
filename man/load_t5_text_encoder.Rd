% tinyrox says don't edit this manually, but it can't stop you!
\name{load_t5_text_encoder}
\alias{load_t5_text_encoder}
\title{Load a T5 encoder from a transformers directory}
\usage{
load_t5_text_encoder(model_path, device = "cpu", dtype = "float32",
                     verbose = TRUE, ...)
}
\arguments{
\item{model_path}{Directory with \code{config.json} and
\code{model*.safetensors} (FLUX.1-schnell's \code{text_encoder_2}).}

\item{device}{Character. Target device.}

\item{dtype}{Character. "float32" (CPU default; T5 overflows in
float16) or "bfloat16".}

\item{verbose}{Logical.}

\item{...}{Overrides for \code{\link{t5_encoder}} arguments.}
}
\value{
The loaded \code{t5_encoder} in eval mode.
}
\description{
Streams the (possibly sharded) safetensors weights into
\code{\link{t5_encoder}}, stripping the \code{encoder.} key prefix
and aliasing \code{embed_tokens} to the shared embedding.
}
