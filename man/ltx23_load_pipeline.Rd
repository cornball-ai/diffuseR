% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_load_pipeline}
\alias{ltx23_load_pipeline}
\title{Load the LTX-2.3 generation components from a single-file checkpoint}
\usage{
ltx23_load_pipeline(
  checkpoint_path,
  device = "cuda",
  dtype = "bfloat16",
  transformer_device = "cpu",
  components = c("dit", "connectors", "vae", "audio_vae", "vocoder"),
  pin = TRUE,
  attn_chunk = NULL,
  phase_offload = TRUE,
  resident_gb = Inf,
  resident_from = c("host", "disk"),
  stage_from = c("host", "disk"),
  verbose = TRUE
)
}
\arguments{
\item{checkpoint_path}{Path to the single-file checkpoint (e.g.
\code{ltx-2.3-22b-distilled-1.1.safetensors}) or to an fp8 artifact
directory produced by \code{\link{ltx23_quantize_fp8}}. With the fp8
artifact, the transformer loads with CPU-resident fp8 weights that
stream to \code{device} during the forward pass. An open
\code{ltx23_checkpoint} is also accepted, such as the LTX-2.5
component set from \code{\link{ltx25_open_checkpoint}}.}

\item{device}{Character. Device for the small components (VAEs,
vocoder, connectors) and, with fp8, the transformer residents.}

\item{dtype}{Character. "bfloat16" (checkpoint native) or "float32".}

\item{transformer_device}{Character. Device for the transformer
weights when loading the plain (non-fp8) checkpoint.}

\item{components}{Character vector. Which components to load.}

\item{pin}{Logical. Pin host memory: the fp8 weights, and the
phase-offload staging buffers (with \code{options(diffuseR.pin_staging)}).
Freed pinned memory stays in torch's host cache, so pass
\code{FALSE} for a pipeline that will be dropped to make room.}

\item{attn_chunk}{Integer or NULL. Query-chunk size for attention
(see \code{\link{ltx23_set_attn_chunk}}).}

\item{phase_offload}{Logical. Load the small components (connectors,
VAEs, vocoder) to the CPU; the pipeline moves each onto the compute
device only for its phase.}

\item{resident_gb}{Numeric. GB of the transformer kept on the GPU
during denoising (default \code{Inf}: all of it). With less, its
leading blocks stay resident and the rest stream from (pinned) host
memory one block at a time every step, for cards that cannot hold
the ~12.5 GB NF4 transformer; under 7 GB the text connectors
stream too. Needs \code{phase_offload} on CUDA. See
\code{\link{ltx23_memory_profile}}.}

\item{resident_from}{Where the GPU-resident part of an NF4
transformer comes back from after each render's decode phase:
\code{"host"} (default) keeps a pinned host copy, \code{"disk"}
keeps none and re-reads it from the artifact at the next onload
(the OS page cache serves it when RAM allows). \code{"disk"} saves
host RAM equal to the resident part (up to ~12 GB) for a re-read
per render. Needs \code{phase_offload} on CUDA.}

\item{stage_from}{Where every phase-offloaded weight lives between
uses: \code{"host"} (default) in host memory; \code{"disk"} nowhere
but the artifact. Loading then reads no weights, and each onload
(the transformer's resident part per render, its streamed blocks
every step, the connectors, VAEs and vocoder per phase) reads from
the memory-mapped shards, so the pipeline holds almost no host
RAM and the OS page cache keeps what fits. Implies
\code{resident_from = "disk"}; the connectors always stream. For
machines with less host RAM than the model; each step re-reads the
streamed blocks (~1.4 GB/s cold from NVMe, ~4.5 GB/s from the page
cache). Needs an NF4 artifact directory and \code{phase_offload}
on CUDA.}

\item{verbose}{Logical.}
}
\value{
A list with the loaded modules and the checkpoint config,
  class \code{ltx23_pipeline}.
}
\description{
Builds the transformer, connectors, video VAE, audio VAE, and vocoder
with the LTX 2.3 configuration and streams the checkpoint weights into
them. The Gemma3 text encoder ships separately (see
\code{\link{load_gemma3_text_encoder}}).
}
