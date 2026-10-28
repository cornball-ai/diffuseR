% tinyrox says don't edit this manually, but it can't stop you!
\name{flux3_load_pipeline}
\alias{flux3_load_pipeline}
\title{Load the FLUX 3 Image Klein pipeline}
\usage{
flux3_load_pipeline(
  model_dir = NULL,
  device = "cuda",
  precision = c("auto", "fp8", "nf4", "bf16"),
  text_device = NULL,
  attn_chunk = NULL,
  phase_offload = TRUE,
  pin = NULL,
  encoder = TRUE,
  revision = .flux3_revision,
  dtype = NULL,
  verbose = TRUE
)
}
\arguments{
\item{model_dir}{Quantized artifact directory (default: the
\code{download_flux3_klein} location for \code{precision}), or a raw
diffusers transformer directory.}

\item{device}{Character. Compute device.}

\item{precision}{"auto" (default: reuse an existing artifact, else
fp8 when safetensors supports float8, else nf4), "fp8", "nf4", or
"bf16" (the raw diffusers transformer).}

\item{text_device}{Device for the Qwen3-VL encoder (default:
\code{device}; it encodes in its own phase and offloads). "cpu"
keeps it in float32 on the host.}

\item{attn_chunk}{Integer or NULL. Attention query-chunk override.}

\item{phase_offload}{Logical. One GPU tenant per phase.}

\item{pin}{Logical or NULL. Page-lock the phase-swapped weights (see
\code{\link{staging}}).}

\item{encoder}{Logical. Also load the VAE encoder, which
\code{\link{img2img_flux3}} needs.}

\item{revision}{Exact 40-hex commit for the support files (default:
the validated snapshot).}

\item{dtype}{Compute dtype, "bfloat16" or "float32". NULL (default):
bfloat16 on CUDA, float32 on the CPU. "bfloat16" on the CPU halves
host memory (the raw transformer is ~11 GB instead of ~22 GB).}

\item{verbose}{Logical.}
}
\value{
A \code{flux3_pipeline} list.
}
\description{
Loads the quantized transformer artifact plus the VAE (encoder and
decoder), the Qwen3-VL text encoder (text model only) and the
tokenizer from the HuggingFace cache populated by
\code{\link{download_flux3_klein}}. With phase offload each component
visits the GPU only for its own phase, so the GPU peak is the largest
phase rather than the sum: the ~8 GB Qwen3-VL encode, or the fp8
transformer (~6 GB) plus activations.
}
