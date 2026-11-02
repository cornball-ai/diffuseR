% tinyrox says don't edit this manually, but it can't stop you!
\name{unet_motion}
\alias{unet_motion}
\title{SD UNet with AnimateDiff motion modules}
\usage{
unet_motion(unet, adapter)
}
\arguments{
\item{unet}{A \code{\link{unet_native}}.}

\item{adapter}{A \code{\link{motion_adapter}} with the same block widths.}
}
\value{
An \code{nn_module}.
}
\description{
Wraps a native SD 1.5 \code{\link{unet_native}} and a
\code{\link{motion_adapter}}. The forward takes latents
\code{[B, 4, F, H, W]}; frames run through the image layers as a batch
of \code{B * F} and through each motion module along the frame axis.
The text context is either \code{[B, 77, D]} (repeated per frame) or
\code{[B * F, 77, D]}, one row per frame, which is how prompt travel
conditions each frame on its own prompt.
}
