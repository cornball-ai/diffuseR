% tinyrox says don't edit this manually, but it can't stop you!
\name{audio_encode_ltx23}
\alias{audio_encode_ltx23}
\title{Audio Conditioning Frontend for LTX-2.3}
\description{
Turns user audio into the normalized, packed audio latents the joint
denoiser conditions on (lip sync): decode to 16 kHz stereo PCM,
log-mel via a causal STFT (filter 1024, hop 160, 64 slaney-normed
mel bins to 8 kHz — the checkpoint's preprocessing spec), then the
audio VAE encoder in argmax mode. The STFT and mel-filterbank
constructors were verified against the checkpoint's stored vocoder
bases (identical up to bf16 rounding), so the convention matches
training.
}
