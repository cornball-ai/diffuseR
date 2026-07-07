% tinyrox says don't edit this manually, but it can't stop you!
\name{load_qwen3_text_encoder}
\alias{load_qwen3_text_encoder}
\title{Load a Qwen3 encoder from a transformers directory}
\usage{
load_qwen3_text_encoder(model_path, device = "cpu", dtype = "float32",
                        verbose = TRUE, ...)
}
\arguments{
\item{model_path}{Directory with \code{config.json} and
\code{model*.safetensors} (FLUX.2-klein's \code{text_encoder}).}

\item{device}{Character. Target device.}

\item{dtype}{Character. "bfloat16" (GPU) or "float32" (CPU).}

\item{verbose}{Logical.}

\item{...}{Overrides for \code{\link{qwen3_encoder}} arguments.}
}
\value{
The loaded \code{qwen3_encoder} in eval mode.
}
\description{
Streams the (possibly sharded) safetensors weights into
\code{\link{qwen3_encoder}}. The LM head is tied to the embeddings
and skipped.
}
