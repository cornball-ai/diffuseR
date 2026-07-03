% tinyrox says don't edit this manually, but it can't stop you!
\name{upsampler_ltx23}
\alias{upsampler_ltx23}
\title{LTX-2.3 Spatial Latent Upsampler}
\description{
Fresh R port of the LTX latent upsampler from the diffusers reference
(Apache-2.0, pipelines/ltx2/latent_upsampler.py and
pipeline_ltx2_latent_upsample.py), with the LTX 2.3 configuration:
Conv3d ResBlock stages around a per-frame 2x pixel-shuffle spatial
upsampler (no rational resampler). Operates on unnormalized latents.
}
