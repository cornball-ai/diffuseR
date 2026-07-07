% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_prepare_latent_ids}
\alias{flux2_prepare_latent_ids}
\title{Build FLUX.2 latent position ids}
\usage{
flux2_prepare_latent_ids(height, width, device = "cpu")
}
\arguments{
\item{height}{Integer. Packed grid height (pixel height / 16).}

\item{width}{Integer. Packed grid width (pixel width / 16).}

\item{device}{Device for the resulting tensor.}
}
\value{
Float tensor [height * width, 4].
}
\description{
Columns (T, H, W, L) with H and W carrying the packed-grid position
(row-major: H varies slowest), T = L = 0. Reference:
Flux2KleinPipeline._prepare_latent_ids.
}
