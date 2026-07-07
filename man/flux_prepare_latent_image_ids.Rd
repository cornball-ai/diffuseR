% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_prepare_latent_image_ids}
\alias{flux_prepare_latent_image_ids}
\title{Build FLUX latent image position ids}
\usage{
flux_prepare_latent_image_ids(height, width, device = "cpu")
}
\arguments{
\item{height}{Integer. Packed grid height (latent height / 2).}

\item{width}{Integer. Packed grid width (latent width / 2).}

\item{device}{Device for the resulting tensor.}
}
\value{
Float tensor of shape [height * width, 3].
}
\description{
Position ids over the packed latent grid (latent height/2 x width/2).
Channel 1 is always zero, channel 2 holds the row index, channel 3 the
column index. Reference: FluxPipeline._prepare_latent_image_ids.
}
