% tinyrox says don't edit this manually, but it can't stop you!
\name{load_qwen3_text_encoder}
\alias{load_qwen3_text_encoder}
\title{Load a Qwen3 encoder from a transformers directory}
\usage{
load_qwen3_text_encoder(
  model_path,
  device = "cpu",
  dtype = "float32",
  verbose = TRUE,
  ...
)
}
\arguments{
\item{model_path}{Directory with \code{config.json} and
\code{model*.safetensors} (FLUX.2-klein's or FLUX 3's
\code{text_encoder}), or a \code{\link{qwen3_quantize}} artifact.}

\item{device}{Character. Target device.}

\item{dtype}{Character. "bfloat16" (GPU) or "float32" (CPU).}

\item{verbose}{Logical.}

\item{...}{Overrides for \code{\link{qwen3_encoder}} arguments.}
}
\value{
The loaded \code{qwen3_encoder} in eval mode.
}
\description{
Streams the (possibly sharded) safetensors weights into
\code{\link{qwen3_encoder}}. The LM head is tied to the embeddings
and skipped. A Qwen3-VL checkpoint (FLUX 3's text encoder: config
under \code{text_config}, weights under \code{model.language_model.*})
loads its text model only; the vision tower is skipped. For text-only
input with 1D positions, Qwen3-VL's interleaved MRoPE reduces to the
plain Qwen3 rotary (all three axes carry the same position).
}
\details{
A directory written by \code{\link{qwen3_quantize}} loads with fp8 or
NF4 projection linears that dequantize per forward. fp8 weights are
plain module fields, not parameters: they move with \code{device}
here, but \code{$to()} on the returned model moves only the residents.

}
