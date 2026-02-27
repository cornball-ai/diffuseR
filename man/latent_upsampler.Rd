% tinyrox says don't edit this manually, but it can't stop you!
\name{latent_upsampler}
\alias{latent_upsampler}
\title{Latent Upsampler}
\usage{
latent_upsampler(
  in_channels = 128L,
  mid_channels = 1024L,
  num_blocks_per_stage = 4L,
  spatial_scale = 2
)
}
\arguments{
\item{in_channels}{Integer. Input/output latent channels (default 128).}

\item{mid_channels}{Integer. Intermediate channels (default 1024).}

\item{num_blocks_per_stage}{Integer. ResBlocks per stage (default 4).}

\item{spatial_scale}{Numeric. Upscale factor (default 2.0).}
}
\value{
An \code{nn_module}.
}
\description{
Full model: Conv3d initial -> GroupNorm -> SiLU -> 4x ResBlock -> SpatialRationalResampler -> 4x ResBlock -> Conv3d final.
}
\keyword{internal}
