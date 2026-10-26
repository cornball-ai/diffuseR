% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_prepare_image_ids}
\alias{flux2_prepare_image_ids}
\title{FLUX.2 position ids for reference images}
\usage{
flux2_prepare_image_ids(sizes, device = "cpu")
}
\arguments{
\item{sizes}{List of \code{c(height, width)} packed grid sizes, one
per reference image.}

\item{device}{Device for the resulting tensor.}
}
\value{
Float tensor [sum(height * width), 4].
}
\description{
(T, H, W, L) coordinates for reference-image tokens, concatenated in
order: the i-th reference (1-based) takes T = 10 * i, which keeps it
apart from the generated image's T = 0. Reference:
Flux2KleinPipeline._prepare_image_ids (scale 10).
}
