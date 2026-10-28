% tinyrox says don't edit this manually, but it can't stop you!
\name{flux3_preprocess_image}
\alias{flux3_preprocess_image}
\title{Preprocess a reference image for FLUX 3}
\usage{
flux3_preprocess_image(image, max_area = 4 * 1024 * 1024, max_side = 4096L)
}
\arguments{
\item{image}{Path to a PNG/JPEG, or an [H, W, C] array in [0, 1].}

\item{max_area}{Numeric. Canvas area cap (reference: 4 megapixels).
Lower it to bound memory; the reference tokens are
\code{canvas area / 256}.}

\item{max_side}{Integer. Canvas side cap.}
}
\value{
\code{list(image, crop)}: a float32 tensor [1, 3, H, W] in
  [-1, 1] at the canvas size, and the (left, top, right, bottom) box
  of the image inside the canvas (0-based, right/bottom exclusive),
  used to crop the generated image back.
}
\description{
The single-reference ("fill") path of the reference pipeline
(diffusers Flux3ImageProcessor, Apache-2.0, following BFL's
\code{reference_canvas}): the image is kept whole. It is downscaled,
keeping its aspect ratio, until its canvas (its size rounded up to the
16 px token grid, sides clamped to [64, \code{max_side}]) fits
\code{max_area}; backing the scale off by 0.5\% per try when the
rounding overshoots. The resize is PIL's antialiased Lanczos-3 in
floating point. The image is then centered on the canvas and the
border filled with its reflection (numpy \code{"reflect"}, or
\code{"symmetric"} when a side is at most 16 px).
}
