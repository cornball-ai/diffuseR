\name{txt2img}
\alias{txt2img}
\title{Generate an image from a text prompt using a diffusion pipeline}
\description{Generate an image from a text prompt using a diffusion pipeline}
\usage{
txt2img(prompt, model_name, ...)
}
\arguments{
  \item{prompt}{A character string prompt describing the image to generate.}
  \item{model_name}{Name of the model to use (e.g., `"sd21"`).}
  \item{...}{Additional parameters passed to the diffusion process.}
}
\value{
A tensor or image object, depending on implementation.
}
\examples{
\dontrun{
img <- txt2img("a cat wearing sunglasses in space", device = "cuda")
}
}
