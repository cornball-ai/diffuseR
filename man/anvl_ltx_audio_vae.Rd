% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_ltx_audio_vae}
\alias{anvl_ltx_audio_vae}
\title{LTX-2.3 audio VAE decoder (anvl port of ltx23_audio_decoder)}
\description{
anvl re-implementation of \code{diffuseR::ltx23_audio_decoder}: the
causal-conv audio decoder that turns packed audio latents into a mel
spectrogram for the vocoder (the vocoder mel->waveform stage is a
separate component, not ported here). Structure: \code{conv_in}; a mid
block of two ResNet blocks; up stages (one per channel multiplier)
each a stack of channel-changing ResNet blocks plus an optional
nearest-2x upsampler; a per-channel RMS-norm -> SiLU -> \code{conv_out}
head; then a crop/pad to the target (channels, time, mel bins).
}
\details{
The audio VAE treats the mel spectrogram as a 2D \code{[B, C, T, M]}
tensor (T = time frames on the "height" axis, M = mel bins on the
"width" axis). It differs from the 3D video VAE
(\code{R/anvl_ltx_vae.R}) in three ways that matter for the port:
\itemize{
\item \strong{Causal Conv2d with ZERO padding}, not edge replication.
The time axis is padded on the leading (past) side by \code{k_t-1}
zeros; the mel axis gets symmetric zero padding. Implemented as an
explicit \code{\link[anvl]{nv_pad}} (mirroring the reference
\code{nnf_pad}) followed by an unpadded \code{\link[anvl]{nv_conv2d}}.
\item \strong{Nearest-2x + causal-conv upsampling}, not pixel shuffle.
\code{\link[yunque]{yq_upsample_nearest2d}} doubles both axes, a
causal conv follows, then the first (causally duplicated) time
frame is dropped.
\item \strong{Channel-changing ResNet blocks} with a 1x1 causal-conv
\code{nin_shortcut} on the residual. Unlike the 2.3 video decoder
(every ResNet in==out), each audio up stage's first block changes
channels, so the shortcut branch is required.
}

Per-channel RMS norm (\code{ltx23_per_channel_rms_norm}, eps 1e-6) is
parameterless, so the entire state_dict is Conv2d \code{weight}/
\code{bias} pairs.

}
