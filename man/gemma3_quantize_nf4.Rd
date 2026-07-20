% tinyrox says don't edit this manually, but it can't stop you!
\name{gemma3_quantize_nf4}
\alias{gemma3_quantize_nf4}
\title{Quantize a Gemma3 text encoder to NF4 shards}
\usage{
gemma3_quantize_nf4(model_path, output_dir = NULL, shard_bytes = 1.9e+09,
                    force = FALSE, verbose = TRUE)
}
\arguments{
\item{model_path}{HuggingFace snapshot directory (config.json +
model-*.safetensors).}

\item{output_dir}{Output directory for shards + manifest (default:
\code{gemma3-nf4} under \code{tools::R_user_dir}).}

\item{shard_bytes}{Numeric. Target shard size in bytes; the 1.9e9
default keeps shards readable by stock CRAN safetensors.}

\item{force}{Logical. Re-quantize even if a valid manifest exists.}

\item{verbose}{Logical.}
}
\value{
Invisibly, the manifest list.
}
\description{
Streams the HuggingFace Gemma3 checkpoint tensor by tensor. The
language model's projection weights (q/k/v/o and gate/up/down, ~11B
of the 12B parameters) are stored as NF4 (packed uint8 +
\code{<key>_absmax} float32 blocks); embeddings and norms are copied
at the resident dtype. Vision-tower and projector weights are
dropped - the text encoder never uses them. The result is a ~8 GB
artifact that fits a 16 GB card during the encode phase (vs 45 GB
of host RAM for the fp32 CPU path).
}
\details{
Keys are stored normalized (\code{language_model.} / \code{model.}
prefixes stripped), matching the module tree of
\code{\link{gemma3_text_model}}.

}
