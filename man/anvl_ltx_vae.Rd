% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_ltx_vae}
\alias{anvl_ltx_vae}
\title{LTX-2.3 3D video VAE decoder (anvl port of ltx23_video_decoder3d)}
\description{
anvl re-implementation of \code{diffuseR::ltx23_video_decoder3d}: the
causal-conv video decoder (\code{conv_in}; a mid block of ResNet
blocks; four up blocks each an optional channel-changing ResNet, a
pixel-shuffle upsampler, and a ResNet stack; a per-channel RMS-norm
-> SiLU -> \code{conv_out} head; channel-to-space un-patchification).
Decodes \code{[B, C_lat, F, H, W]} latents to
\code{[B, 3, 8F-7, 32H, 32W]} pixels in [-1, 1].
}
\details{
Structural differences from the 2D AutoencoderKL decoders (SD/FLUX/
Z-Image, \code{R/anvl_vae.R}):
\itemize{
\item \strong{Causal Conv3d} not Conv2d. The time axis is padded by
\emph{edge replication} (repeat the first/last frame), not zeros:
causal mode replicates the first frame \code{k_t-1} times on the
left; non-causal mode (the decoder default) replicates the first
frame \code{(k_t-1)/2} times on the left and the last frame the
same on the right. The spatial axes get symmetric zero padding.
Built with \code{\link[anvl]{prim_convolution}} on the NCDHW
dimension numbers plus a padding matrix whose temporal row is
\code{c(0, 0)} (already replicated) and spatial rows are
\code{c(k//2, k//2)}.
\item \strong{Per-channel RMS norm} not GroupNorm. Normalizes each
spatiotemporal position by the root-mean-square across the channel
dim, no learned parameters (\code{ltx23_per_channel_rms_norm}).
\item \strong{Pixel-shuffle up/downsampling} not nearest-neighbour.
The upsampler convs, then rearranges channels into space/time
(\code{to_space}), dropping the causally duplicated leading frames
when it upsamples time.
\item \strong{Per-channel latent de-normalization} (latents_mean/std),
applied host-side by \code{\link{yq_ltx_vae_prepare}}.
}

The 2.3 decoder has no ResNet shortcuts (every ResNet is in==out) and
no upsample residuals, so those branches are intentionally not ported.

}
