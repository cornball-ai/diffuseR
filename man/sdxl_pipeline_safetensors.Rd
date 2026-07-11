% tinyrox says don't edit this manually, but it can't stop you!
\name{sdxl_pipeline_safetensors}
\alias{sdxl_pipeline_safetensors}
\title{Native SDXL pipeline from diffusers safetensors}
\description{
Assemble and run the native SDXL pipeline directly from a HuggingFace
diffusers directory (\code{unet/}, \code{vae/}, \code{text_encoder/},
\code{text_encoder_2/}), with no TorchScript \code{.pt} step - so it
works on Blackwell and loads the same weights everyone else uses. The
counterpart to \code{\link{sd_pipeline_from_safetensors}}, adding the
second text encoder (OpenCLIP ViT-bigG) and the added-conditioning
embeddings (pooled \code{text_embeds} + \code{time_ids}) SDXL needs.
}
