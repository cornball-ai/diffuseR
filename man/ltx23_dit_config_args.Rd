% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_dit_config_args}
\alias{ltx23_dit_config_args}
\title{Transformer arguments from a checkpoint's embedded config}
\usage{
ltx23_dit_config_args(config)
}
\arguments{
\item{config}{The parsed embedded config (\code{ckpt$config}), or NULL.}
}
\value{
Named list of \code{\link{ltx23_transformer}} arguments.
}
\description{
Reads the version-dependent flags out of the \code{transformer} entry
of an LTX single-file checkpoint's embedded config. Every other
dimension is shared by the 2.3 and 2.5 22B checkpoints and stays at
the \code{\link{ltx23_transformer}} defaults. Keys absent from the
config (any 2.3 checkpoint) are left out, so the defaults apply.
}
