% tinyrox says don't edit this manually, but it can't stop you!
\name{download_zimage_turbo}
\alias{download_zimage_turbo}
\title{Download Z-Image-Turbo and build the quantized artifact}
\usage{
download_zimage_turbo(quantize = TRUE, precision = c("auto", "fp8", "nf4"),
                      output_dir = NULL, text_encoders = TRUE, verbose = TRUE)
}
\arguments{
\item{quantize}{Logical. Build the quantized artifact.}

\item{precision}{"auto" (default: fp8 when safetensors supports
float8, else nf4), "fp8" (~6.3 GB, GPU-resident; near-bf16
quality), or "nf4" (~3.6 GB).}

\item{output_dir}{Directory for the quantized artifact.}

\item{text_encoders}{Logical. Also fetch the Qwen3-4B text encoder,
tokenizer, VAE, and scheduler config (~8.2 GB).}

\item{verbose}{Logical.}
}
\value{
Invisibly, a list with \code{transformer_dir},
  \code{artifact_dir}, and \code{support} (named file paths).
}
\description{
Skips work already done: a valid quantized manifest short-circuits
the transformer download; cached files are not re-fetched. No token
is needed (the repo is ungated). The float32 transformer source
(~24.6 GB in the HuggingFace cache) may be deleted after
quantization.
}
