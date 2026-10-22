% tinyrox says don't edit this manually, but it can't stop you!
\name{load_gemma4_text_encoder}
\alias{load_gemma4_text_encoder}
\title{Load the Gemma 4 text encoder (LTX-2.5)}
\usage{
load_gemma4_text_encoder(
  path,
  device = "cpu",
  dtype = "bfloat16",
  pin = getOption("diffuseR.pin_staging", TRUE),
  verbose = TRUE
)
}
\arguments{
\item{path}{Path to the LTX-2.5 text encoder file (e.g.
\code{gemma4-12b-with-proj-ltx-2.5-bf16.safetensors}), which embeds
its config as \code{gemma_config} metadata.}

\item{device}{Character. Device for the loaded model.}

\item{dtype}{Character. "bfloat16" (checkpoint native) or "float32".}

\item{pin}{Logical. When loading to the CPU, page-lock the weights so
\code{\link{encode_with_gemma4}} can move the model to the GPU per
encode at DMA speed. Default follows
\code{options(diffuseR.pin_staging)}.}

\item{verbose}{Logical.}
}
\value{
A \code{gemma4_text_model}.
}
\description{
Streams the text weights of an LTX-2.5 Gemma 4 checkpoint into a
\code{\link{gemma4_text_model}} skeleton, one tensor at a time, so
the 26 GB bf16 file is never fully in memory. The vision and audio
towers in the same file are skipped.
}
