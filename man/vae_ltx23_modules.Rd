% tinyrox says don't edit this manually, but it can't stop you!
\name{vae_ltx23_modules}
\alias{vae_ltx23_modules}
\title{LTX-2.3 Video VAE Building Blocks}
\description{
Fresh R port of the LTX-2 causal video autoencoder blocks from the
diffusers reference (Apache-2.0,
src/diffusers/models/autoencoders/autoencoder_kl_ltx2.py). Training
and unused inference branches (noise injection, timestep conditioning,
plain-conv downsampling) are intentionally not ported; the 2.3
checkpoints carry no such weights.
}
