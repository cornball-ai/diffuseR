% tinyrox says don't edit this manually, but it can't stop you!
\name{download_sd15}
\alias{download_sd15}
\title{Download Stable Diffusion 1.5}
\usage{
download_sd15(verbose = TRUE)
}
\arguments{
\item{verbose}{Logical.}
}
\value{
Invisibly, the diffusers-layout directory.
}
\description{
Fetches the fp16 UNet, VAE and CLIP text encoder (about 2.1 GB) from
\code{stable-diffusion-v1-5/stable-diffusion-v1-5} and links them under
the file names the native loaders read.
}
