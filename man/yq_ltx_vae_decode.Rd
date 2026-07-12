% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_vae_decode}
\alias{yq_ltx_vae_decode}
\title{LTX-2.3 VAE decoder forward (anvl)}
\usage{
yq_ltx_vae_decode(z, w, causal = FALSE)
}
\arguments{
\item{z}{AnvlArray \code{[B, C_lat, F, H, W]} de-normalized latents.}

\item{w}{Weights pytree from \code{\link{yq_ltx_vae_load_weights}}.}

\item{causal}{Logical. Temporal padding mode (decoder default FALSE).}
}
\value{
AnvlArray \code{[B, 3, 8F-7, 32H, 32W]} pixels in [-1, 1].
}
\description{
\code{anvl::jit()} the wrapper \code{function(z) yq_ltx_vae_decode(z, w)}.
\code{z} must already be latent de-normalized (see
\code{\link{yq_ltx_vae_prepare}}). \code{causal} is a static (trace-time)
branch selector; the LTX decoder runs non-causal (\code{FALSE}).
}
