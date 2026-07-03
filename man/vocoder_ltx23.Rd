% tinyrox says don't edit this manually, but it can't stop you!
\name{vocoder_ltx23}
\alias{vocoder_ltx23}
\title{LTX-2.3 Vocoder with Bandwidth Extension}
\description{
Fresh R port of the LTX-2 BigVGAN-style vocoder from the diffusers
reference (Apache-2.0, pipelines/ltx2/vocoder.py). The 2.3 vocoder
runs a 16 kHz stage (hidden 1536, snakebeta activations with
anti-aliased up/downsampling), re-analyzes its output into a causal
log-mel spectrogram, and feeds a bandwidth-extension vocoder whose
residual is added to a Hann-resampled skip path for 48 kHz output.
The Kaiser sinc / Hann filters and STFT bases are checkpoint buffers.
Runs in float32 (small model; snakebeta is precision-sensitive).
}
