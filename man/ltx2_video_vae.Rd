% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx2_video_vae}
\alias{ltx2_video_vae}
\title{LTX2 Video VAE}
\usage{
ltx2_video_vae(
  in_channels,
  out_channels,
  latent_channels,
  block_out_channels,
  512L,
  1024L,
  2048L),
  decoder_block_out_channels,
  512L,
  1024L),
  layers_per_block,
  6L,
  6L,
  2L,
  2L),
  decoder_layers_per_block,
  5L,
  5L,
  5L),
  spatio_temporal_scaling,
  TRUE,
  TRUE,
  TRUE),
  decoder_spatio_temporal_scaling,
  TRUE,
  TRUE),
  decoder_inject_noise,
  FALSE,
  FALSE,
  FALSE),
  downsample_type,
  "temporal",
  "spatiotemporal",
  "spatiotemporal"),
  upsample_residual,
  TRUE,
  TRUE),
  upsample_factor,
  2L,
  2L),
  timestep_conditioning,
  patch_size,
  patch_size_t,
  resnet_norm_eps
)
}
\arguments{
\item{in_channels}{Integer. Input channels.}

\item{out_channels}{Integer. Output channels.}

\item{latent_channels}{Integer. Latent space channels.}

\item{block_out_channels}{Integer vector. Encoder block channels.}

\item{decoder_block_out_channels}{Integer vector. Decoder block channels.}

\item{layers_per_block}{Integer vector. Encoder layers per block.}

\item{decoder_layers_per_block}{Integer vector. Decoder layers per block.}

\item{spatio_temporal_scaling}{Logical vector. Encoder scaling.}

\item{decoder_spatio_temporal_scaling}{Logical vector. Decoder scaling.}

\item{decoder_inject_noise}{Logical vector. Noise injection per decoder block.}

\item{downsample_type}{Character vector. Downsampling types.}

\item{upsample_residual}{Logical vector. Upsampler residual flags.}

\item{upsample_factor}{Integer vector. Upsampler factors.}

\item{timestep_conditioning}{Logical. Whether to use timestep conditioning.}

\item{patch_size}{Integer. Spatial patch size.}

\item{patch_size_t}{Integer. Temporal patch size.}

\item{resnet_norm_eps}{Numeric. Normalization epsilon.}

\item{scaling_factor}{Numeric. Latent scaling factor.}

\item{encoder_causal}{Logical. Encoder causality.}

\item{decoder_causal}{Logical. Decoder causality.}

\item{encoder_spatial_padding_mode}{Character. Encoder padding mode.}

\item{decoder_spatial_padding_mode}{Character. Decoder padding mode.}
}
\description{
Full VAE with encoder and decoder, supporting tiled encoding/decoding
for GPU-poor memory management.
}
