% tinyrox says don't edit this manually, but it can't stop you!
\name{checkpoint_flux}
\alias{checkpoint_flux}
\title{FLUX Checkpoint Readers}
\description{
FLUX transformers ship in the diffusers layout: a directory with
\code{config.json}, one or more \code{diffusion_pytorch_model*.safetensors}
shards, and (when sharded) a
\code{diffusion_pytorch_model.safetensors.index.json} weight map.
These helpers open that layout behind the same checkpoint interface as
\code{\link{ltx23_open_checkpoint}}, so the LTX group loaders and
quantization machinery work unchanged. FLUX module names mirror the
checkpoint keys 1:1 - no key mapping is needed.
}
