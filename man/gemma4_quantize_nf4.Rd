% tinyrox says don't edit this manually, but it can't stop you!
\name{gemma4_quantize_nf4}
\alias{gemma4_quantize_nf4}
\title{Quantize the LTX-2.5 Gemma 4 text encoder to NF4 shards}
\usage{
gemma4_quantize_nf4(
  path,
  output_dir = NULL,
  shard_bytes = 1.9e+09,
  force = FALSE,
  verbose = TRUE
)
}
\arguments{
\item{path}{The LTX-2.5 text encoder file
(\code{gemma4-12b-with-proj-ltx-2.5-bf16.safetensors}).}

\item{output_dir}{Output directory (default: \code{gemma4-nf4} under
\code{tools::R_user_dir}).}

\item{shard_bytes}{Numeric. Target shard size in bytes; the 1.9e9
default keeps shards readable by safetensors older than 0.3.0.}

\item{force}{Logical. Re-quantize even if a valid manifest exists.}

\item{verbose}{Logical.}
}
\value{
Invisibly, the manifest list.
}
\description{
Streams the LTX-2.5 encoder file tensor by tensor. The per-layer
projection weights (q/k/v/o and gate/up/down; 328 tensors in the 12B,
since full-attention layers have no v_proj) are stored as NF4 (packed
uint8 + \code{<key>_absmax} float32 blocks); embeddings, norms and the
per-layer \code{layer_scalar} are copied at the resident dtype. Vision
and audio towers are dropped.
}
\details{
The artifact is self-contained for the LTX-2.5 text side: it also
keeps the connectors' \code{text_embedding_projection.*} weights
(resident dtype, original keys) and writes the embedded tokenizer as
\code{tokenizer.json}, so the 26 GB source file is not needed after
quantizing. Load it with \code{\link{load_gemma4_nf4}} (or
\code{\link{load_gemma4_text_encoder}} on the directory) and the
tokenizer with \code{gemma4_tokenizer(output_dir)}.

Peak host memory is about one shard (\code{shard_bytes}) plus the
largest single source tensor and its save buffer (the 2 GB bf16
embedding); measured under a 20 GB cgroup cap on the 12B.

}
