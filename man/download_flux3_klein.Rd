% tinyrox says don't edit this manually, but it can't stop you!
\name{download_flux3_klein}
\alias{download_flux3_klein}
\title{Download FLUX 3 Image Klein and build the quantized artifact}
\usage{
download_flux3_klein(
  quantize = TRUE,
  precision = c("auto", "fp8", "nf4"),
  output_dir = NULL,
  text_encoders = TRUE,
  verbose = TRUE
)
}
\arguments{
\item{quantize}{Logical. Build the quantized artifact.}

\item{precision}{"auto" (default: fp8 when safetensors supports
float8, else nf4), "fp8" (~6 GB), or "nf4".}

\item{output_dir}{Directory for the quantized artifact.}

\item{text_encoders}{Logical. Also fetch the Qwen3-VL text encoder,
tokenizer and VAE (~9.2 GB).}

\item{verbose}{Logical.}
}
\value{
Invisibly, a list with \code{transformer_dir},
  \code{artifact_dir}, and \code{support} (named file paths).
}
\description{
Skips work already done: a valid quantized manifest short-circuits
the transformer download; cached files are not re-fetched. Needs
\code{HF_TOKEN} for an account that accepted the license. The bf16
transformer source (~11 GB in the HuggingFace cache) may be deleted
after quantization.
}
\details{
The FLUX Kommunity License permits non-commercial use, and commercial
use of outputs by Qualifying Users (under US$5M gross annualized
revenue) who filter or review outputs before distributing them and
disclose AI generation where the law requires it. Read LICENSE.md in
the downloaded snapshot for the full terms. The quantized artifact is
a Derivative; keep it local.

}
