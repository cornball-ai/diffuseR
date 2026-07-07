% tinyrox says don't edit this manually, but it can't stop you!
\name{rope_flux2}
\alias{rope_flux2}
\title{FLUX.2 Position Ids and Empirical Shift}
\description{
Fresh R port of the FLUX.2 position-id builders and the empirical
timestep-shift formula from the diffusers reference (Apache-2.0,
src/diffusers/pipelines/flux2/pipeline_flux2_klein.py). FLUX.2 uses
4-axis rotary position ids (T, H, W, L): text tokens carry only the
L axis (sequence position), image latents carry H and W, and the T
axis distinguishes reference images (unused for txt2img). Frequencies
come from \code{\link{flux_pos_embed}} with
\code{axes_dim = c(32, 32, 32, 32)} and \code{theta = 2000}.
}
