% tinyrox says don't edit this manually, but it can't stop you!
\name{load_gemma4_nf4}
\alias{load_gemma4_nf4}
\title{Load the LTX-2.5 Gemma 4 text encoder from an NF4 artifact}
\usage{
load_gemma4_nf4(
  artifact_dir,
  device = "cuda",
  dtype = "bfloat16",
  pin = getOption("diffuseR.pin_staging", TRUE),
  stream = FALSE,
  stage_from = c("host", "disk"),
  verbose = TRUE
)
}
\arguments{
\item{artifact_dir}{Directory produced by \code{\link{gemma4_quantize_nf4}}.}

\item{device}{"cuda" or "cpu".}

\item{dtype}{Compute dtype ("bfloat16" default, or "float32").}

\item{pin}{Logical. When loading to the CPU, page-lock the weights so
\code{\link{encode_with_gemma4}} can move the model to the GPU per
encode at DMA speed. Default follows
\code{options(diffuseR.pin_staging)}.}

\item{stream}{Logical. With \code{device = "cpu"}, stage the model
layer by layer instead of whole: an encode on the GPU then holds
only the embeddings and one layer at a time, for cards that cannot
hold the ~8 GB NF4 model.}

\item{stage_from}{"host" keeps the staged weights in host memory;
"disk" keeps none and reads each layer from the artifact (memory
mapped) when the encode needs it, so the encoder costs almost no
host RAM. "disk" needs \code{device = "cpu"} and a CUDA encode.}

\item{verbose}{Logical.}
}
\value{
A \code{gemma4_text_model}.
}
\description{
Builds the model as a skeleton at the compute dtype, swaps the
projection linears for NF4 modules filled from the artifact
(dequantized per forward), copies the residents (parameters and the
\code{layer_scalar} buffers), and hard-errors on anything the artifact
does not fill. The connectors' projection weights in the artifact are
left for the connector loader.
}
