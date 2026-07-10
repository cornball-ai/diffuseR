% tinyrox says don't edit this manually, but it can't stop you!
\name{unet_safetensors}
\alias{unet_safetensors}
\title{Load HF safetensors weights into the native SD/SDXL UNet}
\description{
The native UNet modules mirror the diffusers
\code{UNet2DConditionModel} state-dict keys 1:1, with the sole
exception that the time- (and, for SDXL, add-) embedding MLPs are
flattened from dotted to underscored names
(\code{time_embedding.linear_1} -> \code{time_embedding_linear_1}).
These loaders read \code{unet/diffusion_pytorch_model.safetensors}
(single file or sharded via its \code{.index.json}) and copy each
weight into the matching native parameter, verifying that every native
parameter is filled and no key or shape is left unmatched.
}
\details{
Reads route through the shared sharded opener, so an oversize (>2 GB)
single-file checkpoint on stock CRAN safetensors surfaces the
actionable "rebuild with smaller shards or install the fork" message
rather than a raw 32-bit overflow.

}
