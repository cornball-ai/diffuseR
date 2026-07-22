% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_video_decoder3d}
\alias{ltx23_video_decoder3d}
\title{LTX-2.3 video decoder}
\usage{
ltx23_video_decoder3d(in_channels = 128L, out_channels = 3L,
                      block_out_channels = c(256L, 512L, 512L, 1024L),
                      spatio_temporal_scaling = c(TRUE, TRUE, TRUE, TRUE),
                      layers_per_block = c(4L, 6L, 4L, 2L, 2L),
                      upsample_type = NULL, patch_size = 4L, patch_size_t = 1L,
                      resnet_norm_eps = 1e-06, is_causal = FALSE,
                      upsample_residual = c(FALSE, FALSE, FALSE, FALSE),
                      upsample_factor = c(2L, 2L, 1L, 2L),
                      spatial_padding_mode = "zeros")
}
\arguments{
\item{block_out_channels}{Integer vector (config order).}

\item{spatio_temporal_scaling}{Logical vector per up block.}

\item{layers_per_block}{Integer vector (config order; first entry is
the mid block after reversal).}

\item{upsample_type}{Character vector per up block (not reversed).}

\item{resnet_norm_eps}{Numeric.}

\item{is_causal}{Logical. FALSE for LTX (symmetric temporal padding).}

\item{upsample_residual}{Logical vector per up block.}

\item{upsample_factor}{Integer vector per up block.}

\item{spatial_padding_mode}{Character.}

\item{in_channels,out_channels}{Integers. Latent and pixel channels.}

\item{patch_size,patch_size_t}{Integers.}
}
\description{
Latents [B, 128, F, H, W] -> pixel video [B, 3, 8F - 7, 32H, 32W].
Block channel lists are given encoder-side (as in the config) and
reversed internally; \code{upsample_type} is indexed directly.
}
