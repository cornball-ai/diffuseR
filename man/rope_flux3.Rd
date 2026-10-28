% tinyrox says don't edit this manually, but it can't stop you!
\name{rope_flux3}
\alias{rope_flux3}
\title{FLUX 3 position ids}
\description{
4-axis (T, H, W, L) position ids for the FLUX 3 transformer, from the
diffusers reference (Apache-2.0,
src/diffusers/modular_pipelines/flux3/before_denoise.py). Image tokens
sit at T = 0 with the H/W axes scaled by the image RoPE scale (0.5);
the generated image has L = 0 and the i-th reference L = -i. Text
advances along L. Rotary frequencies come from
\code{\link{flux_pos_embed}} with \code{axes_dim = c(32, 32, 32, 32)},
\code{theta = 10000}.
}
