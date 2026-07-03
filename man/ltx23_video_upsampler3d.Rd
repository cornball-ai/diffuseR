% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_video_upsampler3d}
\alias{ltx23_video_upsampler3d}
\title{Pixel-shuffle 3D upsampler}
\usage{
ltx23_video_upsampler3d(in_channels, stride = c(1L, 1L, 1L), residual = FALSE,
                        upscale_factor = 1L, spatial_padding_mode = "zeros")
}
\arguments{
\item{in_channels}{Integer.}

\item{stride}{Length-3 integer vector (t, h, w).}

\item{residual}{Logical. Add the rearranged input as a residual.}

\item{upscale_factor}{Integer.}

\item{spatial_padding_mode}{Character.}
}
\description{
Conv followed by channel-to-space/time rearrangement, with an optional
channel-repeat residual and an upscale factor that divides the conv
output channels.
}
