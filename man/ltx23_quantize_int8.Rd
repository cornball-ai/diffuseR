% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_quantize_int8}
\alias{ltx23_quantize_int8}
\title{Quantize an LTX-2.3 checkpoint to int8 shards}
\usage{
ltx23_quantize_int8(checkpoint_path,
                    output_dir = file.path(tools::R_user_dir("diffuseR", "data"), "ltx2.3-int8"),
                    shard_bytes = 1.9e+09, force = FALSE, verbose = TRUE)
}
\arguments{
\item{checkpoint_path}{Source .safetensors (bf16 single file).}

\item{output_dir}{Output directory for shards + manifest.}

\item{shard_bytes}{Numeric. Target shard size in bytes (the 1.9e9
default keeps shards readable by stock CRAN safetensors).}

\item{force}{Logical. Re-quantize even if a valid manifest exists.}

\item{verbose}{Logical.}
}
\value{
Invisibly, the manifest list.
}
\description{
Same streaming layout and cast policy as
\code{\link{ltx23_quantize_fp8}}, but cast-set weights are stored as
int8 with per-output-channel \code{<key>_scale} float32 vectors.
Non-cast tensors are copied through unchanged. The manifest carries
\code{format = "int8"}. No special safetensors build is required.
}
