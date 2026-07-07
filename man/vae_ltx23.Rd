% tinyrox says don't edit this manually, but it can't stop you!
\name{vae_ltx23}
\alias{vae_ltx23}
\title{LTX-2.3 Causal Video VAE}
\description{
Fresh R port of the LTX-2 video autoencoder from the diffusers
reference (Apache-2.0, autoencoder_kl_ltx2.py), with LTX 2.3 defaults:
encoder blocks (256, 512, 1024, 1024), a 4-up-block decoder with mixed
(spatiotemporal, spatiotemporal, temporal, spatial) upsampling, no
upsample residuals, and zeros spatial padding throughout. The encoder
is causal; the decoder is not.
}
