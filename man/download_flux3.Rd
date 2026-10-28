% tinyrox says don't edit this manually, but it can't stop you!
\name{download_flux3}
\alias{download_flux3}
\title{Download and Prepare FLUX 3 Image Klein Weights}
\description{
Downloads FLUX 3 Image Klein from HuggingFace (gated; FLUX Kommunity
License) and quantizes the 5.6B transformer to a local fp8 (~6 GB) or
NF4 artifact. Only the diffusers-format files are fetched: the bf16
transformer shards, the Qwen3-VL text encoder, the VAE and the
tokenizer. BFL's own root \code{model.safetensors} and the
\code{variants/} quantizations are not used.
}
