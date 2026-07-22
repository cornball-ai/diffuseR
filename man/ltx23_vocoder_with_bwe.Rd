% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_vocoder_with_bwe}
\alias{ltx23_vocoder_with_bwe}
\title{LTX-2.3 vocoder with bandwidth extension}
\usage{
ltx23_vocoder_with_bwe(in_channels = 128L, hidden_channels = 1536L,
                       out_channels = 2L,
                       upsample_kernel_sizes = c(11L, 4L, 4L, 4L, 4L, 4L),
                       upsample_factors = c(5L, 2L, 2L, 2L, 2L, 2L),
                       resnet_kernel_sizes = c(3L, 7L, 11L),
                       resnet_dilations = NULL, bwe_in_channels = 128L,
                       bwe_hidden_channels = 512L,
                       bwe_upsample_kernel_sizes = c(12L, 11L, 4L, 4L, 4L),
                       bwe_upsample_factors = c(6L, 5L, 2L, 2L, 2L),
                       bwe_resnet_kernel_sizes = c(3L, 7L, 11L),
                       bwe_resnet_dilations = NULL, filter_length = 512L,
                       hop_length = 80L, window_length = 512L,
                       num_mel_channels = 64L, input_sampling_rate = 16000L,
                       output_sampling_rate = 48000L)
}
\arguments{
\item{out_channels}{Integer. Audio channels.}

\item{hop_length}{Integer. Mel analysis hop.}

\item{in_channels,bwe_in_channels}{Integers. Flattened mel input channels.}

\item{hidden_channels,bwe_hidden_channels}{Integers.}

\item{upsample_kernel_sizes,upsample_factors,bwe_upsample_kernel_sizes,bwe_upsample_factors}{Integer vectors. Per-stage transposed-conv configs.}

\item{resnet_kernel_sizes,bwe_resnet_kernel_sizes}{Integer vectors.}

\item{resnet_dilations,bwe_resnet_dilations}{Lists of integer vectors.}

\item{filter_length,window_length,num_mel_channels}{Integers. Mel
re-analysis configuration.}

\item{input_sampling_rate,output_sampling_rate}{Integers.}
}
\description{
Full mel [B, 2, T, 64] -> 48 kHz stereo waveform pipeline: 16 kHz
vocoder, causal mel re-analysis, BWE vocoder residual added to a
Hann-resampled skip path, clamped to [-1, 1].
}
