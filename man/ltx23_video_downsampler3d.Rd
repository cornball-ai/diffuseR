% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_video_downsampler3d}
\alias{ltx23_video_downsampler3d}
\title{Pixel-unshuffle 3D downsampler}
\usage{
ltx23_video_downsampler3d(in_channels, out_channels, stride = c(1L, 1L, 1L),
                          spatial_padding_mode = "zeros")
}
\arguments{
\item{stride}{Length-3 integer vector (t, h, w).}

\item{spatial_padding_mode}{Character.}

\item{in_channels,out_channels}{Integers.}
}
\description{
Conv followed by space/time-to-channel rearrangement, plus a grouped
channel-mean residual of the same rearrangement.
}
