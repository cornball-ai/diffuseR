% tinyrox says don't edit this manually, but it can't stop you!
\name{VAEDownBlock}
\alias{VAEDownBlock}
\title{VAE Down Block}
\usage{
VAEDownBlock(
  in_channels,
  out_channels,
  num_resnets = 2,
  add_downsample = TRUE,
  norm_groups = 32
)
}
\arguments{
\item{in_channels}{Input channels}

\item{out_channels}{Output channels}

\item{num_resnets}{Number of resnet blocks}

\item{add_downsample}{Whether to add the downsampler}

\item{norm_groups}{Group norm groups}
}
\description{
diffusers DownEncoderBlock2D: resnets, then an optional stride-2
downsample conv. The downsample pads (0, 1, 0, 1) and convolves with
padding 0 (Downsample2D with \code{downsample_padding = 0}).
}
\keyword{internal}
