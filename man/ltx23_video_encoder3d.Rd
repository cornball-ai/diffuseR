% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_video_encoder3d}
\alias{ltx23_video_encoder3d}
\title{LTX-2.3 video encoder}
\usage{
ltx23_video_encoder3d(in_channels = 3L, out_channels = 128L,
                      block_out_channels = c(256L, 512L, 1024L, 1024L),
                      spatio_temporal_scaling = c(TRUE, TRUE, TRUE, TRUE),
                      layers_per_block = c(4L, 6L, 4L, 2L, 2L),
                      downsample_type = NULL, patch_size = 4L,
                      patch_size_t = 1L, resnet_norm_eps = 1e-06,
                      is_causal = TRUE, spatial_padding_mode = "zeros")
}
\arguments{
\item{block_out_channels}{Integer vector. Per-block output channels.}

\item{spatio_temporal_scaling}{Logical vector per block.}

\item{layers_per_block}{Integer vector (blocks then mid).}

\item{downsample_type}{Character vector per block.}

\item{resnet_norm_eps}{Numeric.}

\item{is_causal}{Logical.}

\item{spatial_padding_mode}{Character.}

\item{in_channels,out_channels}{Integers. Pixel and latent channels.}

\item{patch_size,patch_size_t}{Integers. Pixel patchification.}
}
\description{
Pixel video [B, 3, F, H, W] -> latent statistics
[B, 2 * latent_channels, F/8, H/32, W/32] (mean and a uniform log-var
channel broadcast across the latent channels).
}
