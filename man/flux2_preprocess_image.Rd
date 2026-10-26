% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_preprocess_image}
\alias{flux2_preprocess_image}
\title{Preprocess a FLUX.2 reference image}
\usage{
flux2_preprocess_image(image, max_area = 1024 * 1024)
}
\arguments{
\item{image}{Path to a PNG/JPEG, or an [H, W, C] array in [0, 1].}

\item{max_area}{Numeric. Pixel area cap.}
}
\value{
Float32 tensor [1, 3, H, W] in [-1, 1].
}
\description{
Mirrors the diffusers klein pipeline: validate (both sides at least
64 px, aspect ratio at most 8), Lanczos-resize to \code{max_area} if
larger (PIL's antialiased Lanczos-3, computed in floating point), floor
both sides to a multiple of 16, center-crop, and scale to [-1, 1].
}
