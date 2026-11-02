% tinyrox says don't edit this manually, but it can't stop you!
\name{animatediff_load_pipeline}
\alias{animatediff_load_pipeline}
\title{Load the AnimateDiff SD 1.5 pipeline}
\usage{
animatediff_load_pipeline(
  motion = "v3",
  model_dir = NULL,
  device = "cuda",
  dtype = NULL,
  verbose = TRUE
)
}
\arguments{
\item{motion}{Motion module: "v3", "v2", or a path to a motion
safetensors file.}

\item{model_dir}{SD 1.5 diffusers directory; NULL uses
\code{\link{download_sd15}}.}

\item{device}{Compute device for the UNet and VAE.}

\item{dtype}{UNet and motion module dtype; NULL means float16 on CUDA,
float32 on CPU.}

\item{verbose}{Logical.}
}
\value{
A list with \code{unet} (a \code{\link{unet_motion}}),
  \code{decoder}, \code{text_encoder}, \code{device}, \code{dtype} and
  \code{max_frames}.
}
\description{
Load the AnimateDiff SD 1.5 pipeline
}
