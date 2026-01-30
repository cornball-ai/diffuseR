% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx2_video_encoder3d}
\alias{ltx2_video_encoder3d}
\title{LTX2 Video Encoder}
\usage{
ltx2_video_encoder3d(
  in_channels = 3L,
  out_channels = 128L,
  block_out_channels = c(256L, 512L, 1024L, 2048L),
  spatio_temporal_scaling = c(TRUE, TRUE, TRUE, TRUE),
  layers_per_block = c(4L, 6L, 6L, 2L, 2L),
  downsample_type = c("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
  patch_size = 4L,
  patch_size_t = 1L,
  resnet_norm_eps = 1e-06,
  is_causal = TRUE,
  spatial_padding_mode = "zeros"
)
}
\arguments{
\item{in_channels}{Integer. Input channels (typically 3 for RGB).}

\item{out_channels}{Integer. Latent channels.}

\item{block_out_channels}{Integer vector. Output channels per block.}

\item{spatio_temporal_scaling}{Logical vector. Whether each block downscales.}

\item{layers_per_block}{Integer vector. Number of layers per block.}

\item{downsample_type}{Character vector. Type of downsampling per block.}

\item{patch_size}{Integer. Spatial patch size.}

\item{patch_size_t}{Integer. Temporal patch size.}

\item{resnet_norm_eps}{Numeric. Epsilon for normalization.}

\item{is_causal}{Logical. Whether to use causal convolutions.}

\item{spatial_padding_mode}{Character. Padding mode.}
}
\description{
Encodes video frames into latent space with 3D causal convolutions.
}
