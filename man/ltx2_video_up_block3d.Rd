% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx2_video_up_block3d}
\alias{ltx2_video_up_block3d}
\title{LTX2 Video Up Block 3D}
\usage{
ltx2_video_up_block3d(
  in_channels,
  out_channels = NULL,
  num_layers = 1L,
  dropout = 0,
  resnet_eps = 1e-06,
  resnet_act_fn = "swish",
  spatio_temporal_scale = TRUE,
  inject_noise = FALSE,
  timestep_conditioning = FALSE,
  upsample_residual = FALSE,
  upscale_factor = 1L,
  spatial_padding_mode = "zeros"
)
}
\arguments{
\item{in_channels}{Integer. Input channels.}

\item{out_channels}{Integer or NULL. Output channels.}

\item{num_layers}{Integer. Number of ResNet layers.}

\item{dropout}{Numeric. Dropout rate.}

\item{resnet_eps}{Numeric. Epsilon for normalization.}

\item{resnet_act_fn}{Character. Activation function.}

\item{spatio_temporal_scale}{Logical. Whether to use upsampling.}

\item{inject_noise}{Logical. Whether to inject noise.}

\item{timestep_conditioning}{Logical. Whether to use timestep conditioning.}

\item{upsample_residual}{Logical. Whether upsampler uses residual.}

\item{upscale_factor}{Integer. Channel upscale factor.}

\item{spatial_padding_mode}{Character. Padding mode.}
}
\description{
Decoder up block with upsampling and ResNet layers.
}
