% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_load_pipeline}
\alias{flux_load_pipeline}
\title{Load the FLUX.1-schnell pipeline}
\usage{
flux_load_pipeline(model_dir = NULL, device = "cuda", precision = NULL,
                   text_device = "cpu", attn_chunk = NULL,
                   phase_offload = TRUE, verbose = TRUE)
}
\arguments{
\item{model_dir}{Quantized artifact directory (default: the
\code{download_flux1} location for \code{precision}), or a raw
diffusers transformer directory for full-precision loading.}

\item{device}{Character. Compute device.}

\item{precision}{"nf4" or "fp8"; NULL picks the
\code{\link{flux_memory_profile}} recommendation.}

\item{text_device}{Device for the text encoders ("cpu" default; the
T5-XXL runs float32 there).}

\item{attn_chunk}{Integer or NULL. Attention query-chunk override.}

\item{phase_offload}{Logical. One GPU tenant per phase.}

\item{verbose}{Logical.}
}
\value{
A \code{flux_pipeline} list.
}
\description{
Loads the quantized transformer artifact plus the VAE decoder, CLIP
and T5 text encoders, tokenizer, and scheduler config (from the
HuggingFace cache populated by \code{\link{download_flux1}}).
Components load to the CPU when \code{phase_offload} is on and move
to the GPU only for their phase of the generation.
}
