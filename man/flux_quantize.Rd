% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_quantize}
\alias{flux_quantize}
\title{Quantize a FLUX transformer to NF4 or fp8 shards}
\usage{
flux_quantize(
  transformer_dir,
  output_dir = NULL,
  format = c("nf4", "fp8"),
  shard_bytes = 1.9e+09,
  force = FALSE,
  verbose = TRUE
)
}
\arguments{
\item{transformer_dir}{Source diffusers transformer directory.}

\item{output_dir}{Output directory for shards + manifest (default:
the per-format location under \code{tools::R_user_dir}).}

\item{format}{"nf4" or "fp8".}

\item{shard_bytes}{Numeric. Target shard size in bytes. The default
1.9e9 keeps every shard under the 2^31-byte (~2.15 GB) ceiling in
safetensors older than 0.3.0, so the artifact remains readable there.
Larger values (e.g. 4e9) require safetensors 0.3.0 or newer and are
best kept for local artifacts; use the default for redistribution.}

\item{force}{Logical. Re-quantize even if a valid manifest exists.}

\item{verbose}{Logical.}
}
\value{
Invisibly, the manifest list.
}
\description{
Streams the bf16 diffusers checkpoint tensor by tensor. Cast-set
weights (see \code{\link{flux_is_quant_key}}) are stored as NF4
(packed uint8 + \code{<key>_absmax} float32 blocks) or as
float8_e4m3fn with an absmax/448 per-tensor \code{<key>_scale};
everything else is copied through unchanged. The manifest embeds the
transformer config, so the source checkpoint is not needed again
after quantization.
}
