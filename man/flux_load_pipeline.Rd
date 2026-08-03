% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_load_pipeline}
\alias{flux_load_pipeline}
\title{Load the FLUX.1-schnell pipeline}
\usage{
flux_load_pipeline(model_dir = NULL, device = "cuda", precision = NULL,
                   text_device = NULL, attn_chunk = NULL, phase_offload = TRUE,
                   pin = NULL, verbose = TRUE)
}
\arguments{
\item{model_dir}{Quantized artifact directory (default: the
\code{download_flux1} location for \code{precision}), or a raw
diffusers transformer directory for full-precision loading.}

\item{device}{Character. Compute device.}

\item{precision}{"nf4" or "fp8"; NULL picks the
\code{\link{flux_memory_profile}} recommendation.}

\item{text_device}{Where the text encoders compute. NULL (default)
takes the \code{\link{flux_memory_profile}} recommendation: "cuda"
on 14 GB+ cards (T5-XXL onloads in bfloat16 for its encode) and
"cpu" below that (T5-XXL runs float32 in place, since its ~9.8 GB
GPU encode phase would not fit).}

\item{attn_chunk}{Integer or NULL. Attention query-chunk override.}

\item{phase_offload}{Logical. One GPU tenant per phase.}

\item{pin}{Logical or NULL. Page-lock the phase-swapped weights for
DMA-rate transfer (see \code{\link{staging}}). NULL (default)
resolves via \code{options(diffuseR.pin_staging)} then the
host-RAM-aware \code{\link{recommend}} decision.}

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
