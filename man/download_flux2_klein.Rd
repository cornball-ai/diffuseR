% tinyrox says don't edit this manually, but it can't stop you!
\name{download_flux2_klein}
\alias{download_flux2_klein}
\title{Download FLUX.2-klein-4B and build the quantized artifact}
\usage{
download_flux2_klein(quantize = TRUE, precision = c("fp8", "nf4"),
                     output_dir = NULL, text_encoders = TRUE, verbose = TRUE)
}
\arguments{
\item{quantize}{Logical. Build the quantized artifact.}

\item{precision}{"fp8" (~4 GB, GPU-resident; near-bf16 quality) or
"nf4" (~2.3 GB).}

\item{output_dir}{Directory for the quantized artifact.}

\item{text_encoders}{Logical. Also fetch the Qwen3 text encoder,
tokenizer, VAE, and scheduler config (~8.3 GB).}

\item{verbose}{Logical.}
}
\value{
Invisibly, a list with \code{transformer_dir},
  \code{artifact_dir}, and \code{support} (named file paths).
}
\description{
Skips work already done: a valid quantized manifest short-circuits
the transformer download; cached files are not re-fetched. No token
is needed (the repo is ungated). The bf16 transformer source
(~7.8 GB in the HuggingFace cache) may be deleted after quantization.
}
