% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx25_load_diffusion_decoder}
\alias{ltx25_load_diffusion_decoder}
\title{Load the LTX-2.5 diffusion video decoder}
\usage{
ltx25_load_diffusion_decoder(
  path,
  device = "cpu",
  dtype = torch::torch_bfloat16(),
  ...
)
}
\arguments{
\item{path}{The 2.5 video VAE file
(\code{vae/ltx-2.5-video-vae-bf16.safetensors}, the diffusion
decoder; the encoder half is ignored).}

\item{...}{Architecture arguments for \code{\link{ltx25_diffusion_decoder}}
(the defaults are the shipped decoder).}

\item{device,dtype}{Where and in what precision to build it.}
}
\value{
An \code{\link{ltx25_diffusion_decoder}} in eval mode.
}
\description{
Load the LTX-2.5 diffusion video decoder
}
