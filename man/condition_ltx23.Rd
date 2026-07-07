% tinyrox says don't edit this manually, but it can't stop you!
\name{condition_ltx23}
\alias{condition_ltx23}
\title{LTX-2.3 Prefix Conditioning (image-to-video, video continuation)}
\description{
Fresh R port of the frame-conditioning mechanics from the diffusers
reference (Apache-2.0, pipelines/ltx2/pipeline_ltx2_image2video.py
and pipeline_ltx2_condition.py), restricted to prefix conditioning
at latent index 0 with strength 1: a single start image (i2v) or the
leading pixel frames of a previous clip (continuation). Conditioned
latent tokens are initialized from the VAE-encoded pixels, see a
per-token timestep of zero, and are frozen through the Euler loop.
}
