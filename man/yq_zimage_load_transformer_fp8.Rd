% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_zimage_load_transformer_fp8}
\alias{yq_zimage_load_transformer_fp8}
\title{Load the quantized Z-Image DiT into an anvl pytree (fp8, dequantized)}
\usage{
yq_zimage_load_transformer_fp8(dir, patch_key = "2-1", device = "cpu")
}
\arguments{
\item{dir}{The fp8 artifact directory (manifest.json + shards).}

\item{patch_key}{Character. Embedder / final-layer module-dict key.}

\item{device}{Character. Target device.}
}
\value{
DiT weights pytree.
}
\description{
Reads the sharded fp8 transformer artifact built by
\code{\link{flux_quantize}} (manifest.json + F8_E4M3 shards). The 238
cast linears (block attention + feed-forward) are read as F8_E4M3,
upcast to f32 by yunque's reader, and dequantized by their per-tensor
\code{<key>_scale} (\code{weight_f32 = fp8 * scale}); every other
weight (embedders, timestep MLP, adaLN, norms, final layer, pad
tokens) is a BF16 upcast with no scale. The result is the same pytree
\code{\link{yq_zimage_dit}} expects — identical to
\code{\link{yq_zimage_load_weights}}, only the reader differs.
}
