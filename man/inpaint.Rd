% tinyrox says don't edit this manually, but it can't stop you!
\name{inpaint}
\alias{inpaint}
\title{Inpaint an image using a diffusion model}
\usage{
inpaint(input_image, mask_image, prompt, model_name = "sd21", ...)
}
\arguments{
\item{input_image}{Path to the input image (.jpg/.jpeg/.png) or a 3D array.}

\item{mask_image}{Path to the mask image or a matrix/array. White (1) = inpaint,
Black (0) = keep.}

\item{prompt}{Text prompt to guide the inpainting.}

\item{model_name}{Name of the model to use (currently "sd21").}

\item{...}{Additional parameters passed to the model-specific inpainting function.}
}
\value{
A list containing the generated image array and metadata.
}
\description{
Generates a new image by inpainting masked regions of an input image guided
by a text prompt. Unmasked regions are preserved from the original image.
}
