% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_zimage_vae}
\alias{anvl_zimage_vae}
\title{Z-Image / FLUX.1 16-channel VAE decoder (anvl port of AutoencoderKL decode)}
\description{
anvl re-implementation of \code{diffuseR::vae_decoder_native} at
\code{latent_channels = 16}: the FLUX.1 / Z-Image \code{AutoencoderKL}
decode path. Structurally identical to the SD 2.1 and FLUX.2 VAE
decoders (\code{conv_in}; a mid block of ResNet, self attention,
ResNet; four up blocks of three ResNet blocks with nearest-2x
upsampling on the first three; a GroupNorm-32 -> SiLU -> \code{conv_out}
output head), so the block helpers (\code{.yq_vae_resnet},
\code{.yq_vae_attention}, \code{.yq_vae_up_block}) and conv helpers are
shared with \code{R/anvl_vae.R}.
}
\details{
Two things distinguish the FLUX.1 / Z-Image VAE from the SD and FLUX.2
ports:
\itemize{
\item \strong{16 latent channels} (SD was 4, FLUX.2 was 32).
\item \strong{No \code{post_quant_conv}} \emph{and} \strong{no
BatchNorm}. The FLUX family VAE sets \code{use_post_quant_conv =
FALSE}, so the decoder runs \code{conv_in} straight on the latent.
The latent de-normalization is a scalar shift+scale
(\code{z / scaling_factor + shift_factor}), not SD's single 0.18215
divide and not FLUX.2's per-channel BatchNorm; see
\code{\link{yq_zimage_vae_prepare}}.
}

Decodes \code{[B, 16, H, W]} latents to \code{[B, 3, 8H, 8W]} pixels in
[-1, 1].

}
