% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_zimage_vae_decode}
\alias{yq_zimage_vae_decode}
\title{Z-Image / FLUX.1 VAE decoder forward (anvl)}
\usage{
yq_zimage_vae_decode(z, w)
}
\arguments{
\item{z}{AnvlArray \code{[B, 16, H, W]} latents, already
scaling/shift de-normalized.}

\item{w}{VAE weights pytree from \code{\link{yq_zimage_vae_load_weights}}.}
}
\value{
AnvlArray \code{[B, 3, 8H, 8W]} pixels in [-1, 1].
}
\description{
Reuses the shared AutoencoderKL block helpers from \code{R/anvl_vae.R};
\code{anvl::jit()} the wrapper \code{function(z) yq_zimage_vae_decode(z, w)}.
\code{z} must already be de-normalized (see
\code{\link{yq_zimage_vae_prepare}}). Unlike the FLUX.2 decoder there is
no \code{post_quant_conv}: \code{conv_in} runs directly on the latent.
}
