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
  vram_gb = NULL,
  stream_text = NULL,
  resident_gb = NULL,
  text_precision = c("auto", "fp8", "nf4", "bf16"),
  text_dir = NULL,
  support_dir = NULL,
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

\item{vram_gb}{Numeric or NULL. Card size the streaming defaults are
planned for (default: the detected total).}

\item{stream_text}{Logical or NULL. Stream the Qwen3-VL layers through
the GPU one at a time from host memory, for cards that cannot hold
the whole encoder (default: below 10 GB).}

\item{resident_gb}{Numeric or NULL. GB of fp8 transformer weights
kept on the GPU; the rest stays in (pinned) host memory and streams
in inside each forward. NULL: all of it from 12 GB up, else
\code{vram_gb - 6.5} (1.5 GB on an 8 GB card, enough headroom for
1024x1024 text-to-image and a 1 MP edit). fp8 only.}

\item{text_precision}{Qwen3-VL weights: "auto" (default: the fp8,
then NF4, artifact built by \code{\link{download_flux3_klein}},
else bf16), "fp8" (~4.4 GB of host memory instead of ~8 GB),
"nf4" (~2.8 GB), or "bf16" (the cached source).}

\item{text_dir}{Qwen3-VL text encoder directory: an fp8 or NF4
artifact (\code{\link{qwen3_quantize}}) or a bf16
\code{text_encoder} directory. Its own format decides the
precision; a \code{text_precision} other than "auto" must match it.
NULL (default) finds it by \code{text_precision}.}

\item{support_dir}{Directory laid out like the upstream repository
holding at least \code{vae/config.json},
\code{vae/diffusion_pytorch_model.safetensors} and
\code{text_encoder/tokenizer.json}, read instead of the
HuggingFace cache. With \code{model_dir} and \code{text_dir} it
names every file the pipeline reads, for offline mounts.}

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
