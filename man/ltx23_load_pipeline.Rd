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

\item{pin}{Logical. Pin fp8 host memory (fp8 artifact only).}

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
