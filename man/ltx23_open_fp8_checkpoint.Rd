% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_open_fp8_checkpoint}
\alias{ltx23_open_fp8_checkpoint}
\title{Open an FP8 shard directory as a checkpoint}
\usage{
ltx23_open_fp8_checkpoint(dir)
}
\arguments{
\item{dir}{The fp8 artifact directory (with manifest.json).}
}
\value{
An \code{ltx23_checkpoint}.
}
\description{
Presents the sharded fp8 artifact through the same interface as
\code{\link{ltx23_open_checkpoint}} so the group loaders work
unchanged.
}
