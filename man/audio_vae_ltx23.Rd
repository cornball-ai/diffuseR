% tinyrox says don't edit this manually, but it can't stop you!
\name{audio_vae_ltx23}
\alias{audio_vae_ltx23}
\title{LTX-2.3 Audio VAE (decoder)}
\description{
Fresh R port of the LTX-2 audio autoencoder decoder from the
diffusers reference (Apache-2.0, autoencoder_kl_ltx2_audio.py),
configured per the checkpoint: pixel norm, height-axis causality,
base 128 channels with multipliers (1, 2, 4), 8 latent channels,
64 mel bins, no attention. The encoder is not ported (text-to-video
never encodes audio); its checkpoint keys are skipped deliberately.
}
