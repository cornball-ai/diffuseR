% tinyrox says don't edit this manually, but it can't stop you!
\name{sd_pipeline_safetensors}
\alias{sd_pipeline_safetensors}
\title{Native Stable Diffusion pipelines from diffusers safetensors}
\description{
Assemble and run the native SD pipeline directly from a HuggingFace
diffusers directory (\code{unet/}, \code{vae/}, \code{text_encoder/}),
with no TorchScript \code{.pt} step - so it works on Blackwell and
loads the same weights everyone else uses. SD21 is wired end to end
here; SDXL still needs its second text encoder and added-conditioning
embeddings (tracked in tasks/todo.md).
}
