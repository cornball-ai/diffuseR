% tinyrox says don't edit this manually, but it can't stop you!
\name{flux3_prepare_image_ids}
\alias{flux3_prepare_image_ids}
\title{FLUX 3 image token position ids}
\usage{
flux3_prepare_image_ids(
  height,
  width,
  l_coord = 0,
  rope_scale = 0.5,
  device = "cpu"
)
}
\arguments{
\item{height}{Integer. Token grid rows (pixels / 16).}

\item{width}{Integer. Token grid columns (pixels / 16).}

\item{l_coord}{Numeric. L coordinate: 0 for the generated image,
\code{-i} for the i-th reference image.}

\item{rope_scale}{Numeric. Scale of the H/W axes (0.5).}

\item{device}{Character. Device.}
}
\value{
Float tensor [height * width, 4], row-major over (h, w).
}
\description{
FLUX 3 image token position ids
}
