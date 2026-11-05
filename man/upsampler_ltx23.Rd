% tinyrox says don't edit this manually, but it can't stop you!
\name{upsampler_ltx23}
\alias{upsampler_ltx23}
\title{LTX Latent Upsamplers}
\description{
Fresh R port of the LTX latent upsampler from the diffusers reference
(Apache-2.0, pipelines/ltx2/latent_upsampler.py and
pipeline_ltx2_latent_upsample.py), in the two layouts the LTX-2.3 and
LTX-2.5 releases ship: Conv3d ResBlock stages around either a
per-frame 2x pixel-shuffle spatial upsampler (no rational resampler)
or a Conv3d 2x temporal pixel shuffle. Operates on unnormalized
latents.
}
