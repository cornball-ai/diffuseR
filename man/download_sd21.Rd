% tinyrox says don't edit this manually, but it can't stop you!
\name{download_sd21}
\alias{download_sd21}
\title{Download the Stable Diffusion 2.1 diffusers weights}
\usage{
download_sd21(verbose = TRUE)
}
\arguments{
\item{verbose}{Logical.}
}
\value{
Invisibly, the diffusers directory (the parent of
  \code{unet/}, \code{vae/}, \code{text_encoder/}).
}
\description{
Fetches the UNet, VAE, and CLIP text encoder (config + safetensors)
from \code{stabilityai/stable-diffusion-2-1} into the hfhub cache. The
native tokenizer and DDIM scheduler need no downloads. About 5 GB
(fp32 UNet); one-time.
}
