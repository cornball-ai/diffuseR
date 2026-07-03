% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_open_checkpoint}
\alias{ltx23_open_checkpoint}
\title{Open an LTX-2.3 checkpoint}
\usage{
ltx23_open_checkpoint(path, require_version = "2.3")
}
\arguments{
\item{path}{Path to the checkpoint .safetensors file.}

\item{require_version}{Character. Required \code{model_version} prefix
(default "2.3"). Set to NULL to skip the check.}
}
\value{
An object of class \code{ltx23_checkpoint}: a list with
  \code{handle} (safetensors reader), \code{keys}, \code{version},
  \code{config} (parsed component configs, or NULL), and \code{path}.
}
\description{
Opens a single-file LTX checkpoint lazily (header only), validates the
\code{model_version} metadata, and parses the embedded component
configuration.
}
\examples{
\dontrun{
ckpt <- ltx23_open_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")
str(ltx23_split_keys(ckpt$keys), max.level = 1)
}
}
