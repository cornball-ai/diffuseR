% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_load_pipeline}
\alias{flux2_load_pipeline}
\title{Load the FLUX.2 klein pipeline}
\usage{
flux2_load_pipeline(
  model_dir = NULL,
  device = "cuda",
  precision = c("auto", "fp8", "nf4", "bf16"),
  text_device = NULL,
  attn_chunk = NULL,
  phase_offload = TRUE,
  pin = NULL,
  verbose = TRUE,
  revision = NULL
)
}
\arguments{
\item{model_dir}{Quantized artifact directory (default: the
\code{download_flux2_klein} location for \code{precision}), or a
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

\item{revision}{Optional exact 40-hex commit for the support files (VAE,
Qwen3 encoder, tokenizer) this pulls from the \code{black-forest-labs}
cache. With one they resolve straight out of
\code{snapshots/<revision>/}, so the load works against a cache holding
only that snapshot -- no \code{refs/} entry and no network. The
transformer comes from \code{model_dir} and is unaffected.}
}
\value{
A \code{flux2_pipeline} list.
}
\description{
Loads the quantized transformer artifact plus the FLUX.2 VAE decoder,
Qwen3 text encoder, and tokenizer from the HuggingFace cache
populated by \code{\link{download_flux2_klein}}. With fp8 precision
the ~4 GB transformer stays GPU-resident.
}
