% tinyrox says don't edit this manually, but it can't stop you!
\name{.ltx23_release_vae_traces}
\alias{.ltx23_release_vae_traces}
\title{Release all cached decode traces}
\usage{
.ltx23_release_vae_traces()
}
\value{
Invisibly, NULL.
}
\description{
Traces hold references to the weight tensors they captured; drop
them when a component leaves the GPU so its memory actually frees.
}
\keyword{internal}
