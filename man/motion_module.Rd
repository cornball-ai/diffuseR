% tinyrox says don't edit this manually, but it can't stop you!
\name{motion_module}
\alias{motion_module}
\title{AnimateDiff motion modules}
\description{
Temporal transformers that turn the SD 1.5 image UNet into a video
model: after every resnet (and spatial attention) the UNet runs one
motion module, which attends across frames at each latent position.
Ported from diffusers \code{models/unets/unet_motion_model.py}
(\code{AnimateDiffTransformer3D}, \code{MotionModules},
\code{MotionAdapter}, \code{UNetMotionModel}) and the double
self-attention \code{BasicTransformerBlock} in
\code{models/attention.py}, Apache-2.0.
}
\keyword{internal}
