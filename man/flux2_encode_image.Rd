% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_encode_image}
\alias{flux2_encode_image}
\title{Encode a reference image into FLUX.2 tokens}
\usage{
flux2_encode_image(encoder, image, bn_eps = 1e-04)
}
\arguments{
\item{encoder}{A loaded \code{\link{flux2_vae_encoder}}.}

\item{image}{Tensor [1, 3, H, W] in [-1, 1]; H and W multiples of 16.}

\item{bn_eps}{Numeric. BatchNorm epsilon.}
}
\value{
\code{list(tokens, height, width)}: tokens [1, H/16 * W/16, 128]
  on the encoder's device and dtype, and the packed grid size.
}
\description{
VAE-encode (posterior mean), patchify 2x2, normalize with the VAE
BatchNorm statistics and pack to channels-last tokens. Reference:
Flux2KleinPipeline._encode_vae_image and prepare_image_latents.
}
