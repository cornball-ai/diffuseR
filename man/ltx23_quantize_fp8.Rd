% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_quantize_fp8}
\alias{ltx23_quantize_fp8}
\title{Quantize an LTX-2.3 checkpoint to FP8 shards}
\usage{
ltx23_quantize_fp8(
  checkpoint_path,
  output_dir = NULL,
  shard_bytes = 1.9e+09,
  force = FALSE,
  verbose = TRUE
)
}
\arguments{
\item{checkpoint_path}{Source .safetensors (46 GB bf16 single file).}

\item{output_dir}{Output directory for shards + manifest; NULL (the
default) resolves under tools::R_user_dir("diffuseR", "data").}

\item{shard_bytes}{Numeric. Target shard size in bytes. The default
1.9e9 keeps every shard under the 2^31-byte (~2.15 GB) offset
ceiling, so the artifact reads on a safetensors that lacks the
overflow fix (mlverse/safetensors#14, which reached CRAN in 0.3.0).
Larger values (e.g. 4e9) need that fix present, so keep them for
local artifacts and use the default for anything redistributed.}

\item{force}{Logical. Re-quantize even if a valid manifest exists.}

\item{verbose}{Logical.}
}
\value{
Invisibly, the manifest list.
}
\description{
Streams the single-file bf16 checkpoint tensor by tensor. DiT
attention/FFN linear weights are stored as float8_e4m3fn with a
float32 absmax/448 per-tensor scale (\code{<key>_scale} sibling);
everything else is copied through unchanged. Output shards carry the
original key names plus a manifest for skip-if-exists.
}
