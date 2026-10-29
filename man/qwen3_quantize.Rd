% tinyrox says don't edit this manually, but it can't stop you!
\name{qwen3_quantize}
\alias{qwen3_quantize}
\title{Quantize a Qwen3 text encoder to fp8 or NF4 shards}
\usage{
qwen3_quantize(
  model_path,
  output_dir,
  format = c("fp8", "nf4"),
  shard_bytes = 1.9e+09,
  force = FALSE,
  verbose = TRUE
)
}
\arguments{
\item{model_path}{Source directory with \code{config.json} and
\code{model*.safetensors}.}

\item{output_dir}{Output directory for shards + manifest.}

\item{format}{"fp8" or "nf4".}

\item{shard_bytes}{Numeric. Target shard size in bytes (1.9e9 keeps
shards readable by safetensors older than 0.3.0).}

\item{force}{Logical. Re-quantize even if a valid manifest exists.}

\item{verbose}{Logical.}
}
\value{
Invisibly, the manifest list.
}
\description{
Streams a transformers Qwen3 (FLUX.2 klein, Z-Image) or Qwen3-VL
(FLUX 3) text encoder tensor by tensor. The per-layer projection
weights (q/k/v/o and gate/up/down, 252 tensors in the 36-layer 4B)
are stored as float8_e4m3fn with a per-tensor absmax/448
\code{<key>_scale}, or as NF4 (packed uint8 + \code{<key>_absmax});
embeddings and norms are copied at bfloat16. The vision tower and the
tied LM head are dropped, and keys are written in the
\code{\link{qwen3_encoder}} layout (\code{model.layers.*}). The
manifest embeds the text config, so the loader needs no other file.
}
\details{
Host memory: the 4B text model is ~8 GB in bfloat16, ~4.4 GB as fp8
and ~2.8 GB as NF4 (the 0.8 GB embedding stays bfloat16 in both).
Load the artifact with \code{\link{load_qwen3_text_encoder}} on its
directory.

}
