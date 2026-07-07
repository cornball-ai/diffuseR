% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_prepare_text_ids}
\alias{flux2_prepare_text_ids}
\title{Build FLUX.2 text position ids}
\usage{
flux2_prepare_text_ids(len, device = "cpu")
}
\arguments{
\item{len}{Integer. Text sequence length.}

\item{device}{Device for the resulting tensor.}
}
\value{
Float tensor [len, 4].
}
\description{
Columns (T, H, W, L) with only L varying: 0..len-1. Reference:
Flux2KleinPipeline._prepare_text_ids.
}
