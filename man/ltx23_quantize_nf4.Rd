% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_quantize_nf4}
\alias{ltx23_quantize_nf4}
\title{Quantize an LTX-2.3 checkpoint to NF4 shards}
\usage{
ltx23_quantize_nf4(checkpoint_path,
                   output_dir = file.path(tools::R_user_dir("diffuseR", "data"), "ltx2.3-nf4"),
                   shard_bytes = 1.9e+09, force = FALSE, verbose = TRUE)
}
\arguments{
\item{checkpoint_path}{Source .safetensors (bf16 single file).}

\item{output_dir}{Output directory for shards + manifest.}

\item{shard_bytes}{Numeric. Target shard size in bytes. The default
1.9e9 keeps every shard under the 2^31-byte (~2.15 GB) ceiling that
stock CRAN safetensors can read. Pass a larger value (e.g. 4e9) only
for local builds you will read back with a fork-patched safetensors.}

\item{force}{Logical. Re-quantize even if a valid manifest exists.}

\item{verbose}{Logical.}
}
\value{
Invisibly, the manifest list.
}
\description{
Same streaming layout and cast policy as
\code{\link{ltx23_quantize_fp8}}, but cast-set weights are stored as
NF4 (\code{<key>} packed uint8 + \code{<key>_absmax} float32 blocks +
the original shape recovered from the model config at load time).
Non-cast tensors are copied through unchanged. The manifest carries
\code{format = "nf4"}.
}
