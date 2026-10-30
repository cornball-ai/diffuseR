% tinyrox says don't edit this manually, but it can't stop you!
\name{zimage_load_pipeline}
\alias{zimage_load_pipeline}
\title{Load the Z-Image-Turbo pipeline}
\usage{
zimage_load_pipeline(
  model_dir = NULL,
  device = "cuda",
  precision = c("auto", "fp8", "nf4", "bf16"),
  text_device = NULL,
  attn_chunk = NULL,
  phase_offload = TRUE,
  pin = NULL,
  verbose = TRUE,
  revision = NULL,
  stream_text = FALSE,
  resident_gb = Inf
)
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

\item{revision}{Optional exact 40-hex commit for the support files (VAE,
scheduler config, Qwen3 encoder, tokenizer) this pulls from the
\code{Tongyi-MAI} cache. With one they resolve straight out of
\code{snapshots/<revision>/}, so the load works against a cache holding
only that snapshot -- no \code{refs/} entry and no network. The
transformer comes from \code{model_dir} and is unaffected.}

\item{stream_text}{Logical. Stream the bf16 Qwen3 layers through the
GPU one at a time during the encode instead of onloading the ~8 GB
encoder whole (see \code{\link{flux2_load_pipeline}}). Needs
\code{phase_offload} on CUDA.}

\item{resident_gb}{Numeric. GB of fp8 transformer weights kept on the
GPU through the denoise; the rest stay in (pinned) host memory and
copy in inside each forward (as \code{\link{flux3_load_pipeline}}).
\code{Inf} (default) keeps all ~5.9 GB resident. fp8 on CUDA only.}
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
