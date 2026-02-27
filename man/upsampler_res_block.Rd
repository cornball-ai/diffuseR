% tinyrox says don't edit this manually, but it can't stop you!
\name{upsampler_res_block}
\alias{upsampler_res_block}
\title{Residual Block (Conv3d)}
\usage{
upsampler_res_block(channels, mid_channels = NULL)
}
\arguments{
\item{channels}{Integer. Input/output channels.}

\item{mid_channels}{Integer or NULL. Mid channels (default: same as channels).}
}
\value{
An \code{nn_module}.
}
\description{
Two Conv3d layers with GroupNorm and SiLU, plus skip connection.
}
\keyword{internal}
