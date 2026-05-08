% tinyrox says don't edit this manually, but it can't stop you!
\name{preprocess_mask}
\alias{preprocess_mask}
\title{Preprocess a mask for inpainting}
\usage{
preprocess_mask(
  mask_input,
  height,
  width,
  device = "cpu",
  dtype = torch::torch_float32()
)
}
\arguments{
\item{mask_input}{Path to a mask image (.jpg/.jpeg/.png), a matrix, or a 3D array.
White (1) = inpaint region, Black (0) = keep region.}

\item{height}{Target image height in pixels (will be divided by 8 for latent space).}

\item{width}{Target image width in pixels (will be divided by 8 for latent space).}

\item{device}{Target device ("cpu" or "cuda").}

\item{dtype}{Torch dtype for the output tensor.}
}
\value{
Torch tensor of shape [1, 1, height/8, width/8] with values 0 or 1.
}
\description{
Loads, resizes, and binarizes a mask image for use in the inpainting pipeline.
The mask is resized to latent space dimensions (height/8, width/8).
}
