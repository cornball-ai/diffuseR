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
Fetches the UNet, VAE, and CLIP text encoder from the
\code{cornball-ai/sd21-R} HuggingFace dataset (fp16 diffusers
safetensors, converted from the original OpenRAIL weights; the
upstream \code{stabilityai} repo was deprecated). About 2.5 GB,
one-time. The native tokenizer and DDIM scheduler need no downloads.
}
