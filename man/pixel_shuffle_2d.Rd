% tinyrox says don't edit this manually, but it can't stop you!
\name{pixel_shuffle_2d}
\alias{pixel_shuffle_2d}
\title{2D PixelShuffle (channel -> spatial)}
\usage{
pixel_shuffle_2d(upscale_factor = 2L)
}
\arguments{
\item{upscale_factor}{Integer. Upscale factor (default 2).}
}
\value{
An \code{nn_module}.
}
\description{
Rearranges channels into spatial dimensions:
\code{[B, C*r*r, H, W] -> [B, C, H*r, W*r]}
}
\keyword{internal}
