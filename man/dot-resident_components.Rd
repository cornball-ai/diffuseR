% tinyrox says don't edit this manually, but it can't stop you!
\name{.resident_components}
\alias{.resident_components}
\title{Every nn_module field of a pipeline, by name}
\usage{
.resident_components(pipeline)
}
\arguments{
\item{pipeline}{A loaded diffuseR pipeline.}
}
\value{
A named list of the pipeline's \code{nn_module} fields,
  possibly empty.
}
\description{
Discovery beats a hard-coded list: the families disagree on which
components exist (FLUX.1 has two text encoders, LTX adds a video VAE,
an audio VAE and a vocoder), and a field added later is picked up
without touching this file.
}
\keyword{internal}
