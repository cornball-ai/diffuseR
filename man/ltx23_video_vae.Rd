% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_video_vae}
\alias{ltx23_video_vae}
\title{LTX-2.3 video VAE}
\usage{
ltx23_video_vae(in_channels = 3L, out_channels = 3L, latent_channels = 128L,
                block_out_channels = c(256L, 512L, 1024L, 1024L),
                decoder_block_out_channels = c(256L, 512L, 512L, 1024L),
                layers_per_block = c(4L, 6L, 4L, 2L, 2L),
                decoder_layers_per_block = c(4L, 6L, 4L, 2L, 2L),
                spatio_temporal_scaling = c(TRUE, TRUE, TRUE, TRUE),
                decoder_spatio_temporal_scaling = c(TRUE, TRUE, TRUE, TRUE),
                downsample_type = NULL, upsample_type = NULL,
                upsample_residual = c(FALSE, FALSE, FALSE, FALSE),
                upsample_factor = c(2L, 2L, 1L, 2L), patch_size = 4L,
                patch_size_t = 1L, resnet_norm_eps = 1e-06,
                encoder_causal = TRUE, decoder_causal = FALSE,
                encoder_spatial_padding_mode = "zeros",
                decoder_spatial_padding_mode = "zeros")
}
\arguments{
\item{latent_channels}{Integer.}

\item{resnet_norm_eps}{Numeric.}

\item{in_channels,out_channels}{Integers. Pixel channels.}

\item{block_out_channels,layers_per_block,spatio_temporal_scaling,downsample_type}{Encoder configuration (see \code{\link{ltx23_video_encoder3d}}).}

\item{decoder_block_out_channels,decoder_layers_per_block,decoder_spatio_temporal_scaling,upsample_type,upsample_residual,upsample_factor}{Decoder configuration (see \code{\link{ltx23_video_decoder3d}}).}

\item{patch_size,patch_size_t}{Integers. Pixel patchification.}

\item{encoder_causal,decoder_causal}{Logicals. Temporal padding modes.}

\item{encoder_spatial_padding_mode,decoder_spatial_padding_mode}{Characters.}
}
\value{
Module bundling the video encoder and decoder. Its
  forward(z) is \code{decode(z)}, returning pixels for a latent;
  \code{$encode()} and \code{$decode()} are callable separately.
}
\description{
Encoder + decoder + per-channel latent statistics (loaded from the
checkpoint's \code{per_channel_statistics}). The checkpoint's
\code{scaling_factor} is 1.0, so latent (de)normalization is purely
the per-channel affine map.
}
