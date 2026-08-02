% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_unload}
\alias{resident_unload}
\title{Drop a resident handle entirely}
\usage{
resident_unload(res)
}
\arguments{
\item{res}{A \code{diffuseR_resident} handle.}
}
\value{
Invisibly the handle, with state "unloaded".
}
\description{
Releases the GPU copy if any, drops the pipeline and the pinned host
storage, and marks the handle unloaded. Terminal: nothing but
\code{\link{resident_status}} works afterwards.
}
