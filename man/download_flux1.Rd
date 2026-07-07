% tinyrox says don't edit this manually, but it can't stop you!
\name{download_flux1}
\alias{download_flux1}
\title{Download FLUX.1-schnell and build the quantized artifact}
\usage{
download_flux1(quantize = TRUE, precision = c("nf4", "fp8"), output_dir = NULL,
               text_encoders = TRUE, verbose = TRUE)
}
\arguments{
\item{quantize}{Logical. Build the quantized artifact after
downloading.}

\item{precision}{"nf4" (~7 GB, GPU-resident on 16 GB cards) or
"fp8" (~12 GB, CPU-resident, streamed; near-bf16 quality).}

\item{output_dir}{Directory for the quantized artifact.}

\item{text_encoders}{Logical. Also fetch the CLIP + T5 text encoders,
tokenizer, VAE, and scheduler config (~10 GB).}

\item{verbose}{Logical.}
}
\value{
Invisibly, a list with \code{transformer_dir},
  \code{artifact_dir}, and \code{support} (named file paths).
}
\description{
Skips work already done: a valid quantized manifest short-circuits
the transformer download; cached files are not re-fetched. Needs
\code{HF_TOKEN} set for the gated repo (see the error message it
raises without one). The bf16 transformer source (~24 GB in the
HuggingFace cache) may be deleted after quantization.
}
