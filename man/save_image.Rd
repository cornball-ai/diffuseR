\name{save_image}
\alias{save_image}
\title{Save and Display an Image from a Torch Tensor}
\description{Converts a Torch tensor to a normalized RGB image array, saves it as a PNG file, and optionally displays it in the RStudio Viewer pane using `grid::grid.raster()`.}
\usage{
save_image(img, save_to, normalize)
}
\arguments{
  \item{img}{A numeric with shape `[3, H, W]`.}
  \item{save_to}{File path for the PNG image (default is `"output.png"`).}
  \item{normalize}{Logical; whether to normalize pixel values to `[0, 1]`. Default is `TRUE`.}
}
\value{
Invisibly returns the saved file path.
}
\examples{
\dontrun{
save_image(output_tensor, "sample.png")
}
}
