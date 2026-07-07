% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_bn_normalize}
\alias{flux2_bn_normalize}
\title{Normalize patchified latents with the VAE BatchNorm statistics}
\usage{
flux2_bn_normalize(latents, bn_mean, bn_var, eps = 1e-04, inverse = FALSE)
}
\arguments{
\item{latents}{Tensor [B, 128, H, W] (patchified).}

\item{eps}{Numeric. BatchNorm epsilon.}

\item{inverse}{Logical. De-normalize (decode path) instead.}

\item{bn_mean,bn_var}{Float tensors [128].}
}
\value{
Tensor like \code{latents}.
}
\description{
FLUX.2 has no scalar scaling/shift factor; latents are standardized
per packed channel with the VAE's \code{bn.running_mean} /
\code{bn.running_var} (eps 1e-4). Reference: encode/decode paths of
Flux2KleinPipeline.
}
