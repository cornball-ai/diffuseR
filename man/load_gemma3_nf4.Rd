% tinyrox says don't edit this manually, but it can't stop you!
\name{load_gemma3_nf4}
\alias{load_gemma3_nf4}
\title{Load a Gemma3 text encoder from an NF4 artifact}
\usage{
load_gemma3_nf4(artifact_dir, device = "cuda", dtype = "bfloat16",
                pin = getOption("diffuseR.pin_staging", TRUE), verbose = TRUE)
}
\arguments{
\item{artifact_dir}{Directory produced by
\code{\link{gemma3_quantize_nf4}}.}

\item{device}{"cuda" or "cpu".}

\item{dtype}{Compute dtype ("bfloat16" default).}

\item{pin}{Logical. When loading to the CPU, page-lock the weights
so \code{\link{encode_with_gemma3}} can swap the model to the GPU
at DMA speed per encode and back for free (see
\code{\link{staging_ltx23}}). Default follows
\code{options(diffuseR.pin_staging)}.}

\item{verbose}{Logical.}
}
\value{
A \code{gemma3_text_model} ready for
  \code{\link{encode_with_gemma3}}.
}
\description{
Builds the model as a skeleton at the compute dtype, swaps the
projection linears for NF4 modules filled from the artifact
(dequantized per forward through the shared byte-LUT), copies the
residents, and hard-errors on any parameter the artifact does not
fill.
}
