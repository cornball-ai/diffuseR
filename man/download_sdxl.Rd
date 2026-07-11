% tinyrox says don't edit this manually, but it can't stop you!
\name{download_sdxl}
\alias{download_sdxl}
\title{Download the Stable Diffusion XL diffusers weights}
\usage{
download_sdxl(verbose = TRUE)
}
\arguments{
\item{verbose}{Logical.}
}
\value{
Invisibly, the diffusers directory (the parent of
  \code{unet/}, \code{vae/}, \code{text_encoder/},
  \code{text_encoder_2/}).
}
\description{
Fetches the UNet (re-sharded to sub-2 GB shards), VAE, and both CLIP
text encoders from the \code{cornball-ai/sdxl-R} HuggingFace dataset
(fp16 diffusers safetensors, converted from the original
\code{stabilityai/stable-diffusion-xl-base-1.0} OpenRAIL++ weights).
About 7 GB, one-time. The native tokenizer and DDIM scheduler need no
downloads.
}
