% tinyrox says don't edit this manually, but it can't stop you!
\name{motion_adapter}
\alias{motion_adapter}
\title{AnimateDiff motion adapter}
\usage{
motion_adapter(
  block_out_channels = c(320L, 640L, 1280L, 1280L),
  layers_per_block = 2L,
  use_mid_block = TRUE,
  num_heads = 8L,
  max_len = 32L
)
}
\arguments{
\item{block_out_channels}{UNet block widths.}

\item{layers_per_block}{Motion modules per down block.}

\item{use_mid_block}{Whether the mid block has a motion module (v2
does, v3 does not).}

\item{num_heads}{Attention heads per motion module.}

\item{max_len}{Longest frame window the positions cover.}
}
\value{
An \code{nn_module}.
}
\description{
The motion modules for every block of an SD UNet, laid out as
diffusers' \code{MotionAdapter} so its parameter names match the
converted checkpoints: \code{down_blocks.<i>.motion_modules.<j>},
\code{mid_block.motion_modules.0} and \code{up_blocks.<i>.motion_modules.<j>}.
Up blocks carry one more module than down blocks, as the UNet's up
blocks carry one more resnet.
}
