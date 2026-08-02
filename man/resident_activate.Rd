% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_activate}
\alias{resident_activate}
\title{Bring a resident pipeline onto the GPU}
\usage{
resident_activate(res)
}
\arguments{
\item{res}{A \code{diffuseR_resident} handle.}
}
\value{
Invisibly the handle, with state "active".
}
\description{
Copies every pinned component to the handle's bound device by DMA and
verifies the result tensor-by-tensor. A failure rolls back to the
pinned host state; a rollback that cannot itself be verified leaves
the handle broken.
}
