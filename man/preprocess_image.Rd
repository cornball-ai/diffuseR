\name{preprocess_image}
\alias{preprocess_image}
\title{Preprocess image for Stable Diffusion}
\description{Preprocess image for Stable Diffusion}
\usage{
preprocess_image(input, device, width, height)
}
\arguments{
  \item{input}{File path to .jpg or .png, or a 3D array}
  \item{device}{Target device for torch ("cpu" or "cuda")}
  \item{width}{Desired width of the output image}
  \item{height}{Desired height of the output image}
}
\value{
Torch tensor of shape c(1, 3, 512, 512), scaled to c(-1, 1)
}
