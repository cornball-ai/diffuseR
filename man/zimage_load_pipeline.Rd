% tinyrox says don't edit this manually, but it can't stop you!
\name{zimage_load_pipeline}
\alias{zimage_load_pipeline}
\title{Load the Z-Image-Turbo pipeline}
\usage{
zimage_load_pipeline(model_dir = NULL, device = "cuda",
                     precision = c("auto", "fp8", "nf4"), text_device = NULL,
                     attn_chunk = NULL, phase_offload = TRUE, pin = NULL,
                     verbose = TRUE)
}
\arguments{
\item{model_dir}{Quantized artifact directory (default: the
\code{download_zimage_turbo} location for \code{precision}), or a
raw diffusers transformer directory.}

\item{device}{Character. Compute device.}

\item{precision}{"auto" (default: reuse an existing artifact, else
fp8 when safetensors supports float8, else nf4), "fp8", or "nf4".}

\item{text_device}{Device for the Qwen3 encoder (default:
\code{device}; it encodes in its own phase and offloads).}

\item{attn_chunk}{Integer or NULL. Attention query-chunk override.}

\item{phase_offload}{Logical. One GPU tenant per phase.}

\item{pin}{Logical or NULL. Page-lock the phase-swapped weights for
DMA-rate transfer (see \code{\link{staging}}). NULL (default)
resolves via \code{options(diffuseR.pin_staging)} then the
host-RAM-aware \code{\link{recommend}} decision.}

\item{verbose}{Logical.}
}
\value{
A \code{zimage_pipeline} list.
}
\description{
Loads the quantized transformer artifact plus the 16-channel VAE
decoder, Qwen3-4B text encoder, and tokenizer from the HuggingFace
cache populated by \code{\link{download_zimage_turbo}}. With fp8
precision the ~6.3 GB transformer rides to the GPU per phase.
}
