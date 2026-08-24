% tinyrox says don't edit this manually, but it can't stop you!
\name{.devices_for_pipeline}
\alias{.devices_for_pipeline}
\title{Device configuration for an already-built pipeline}
\usage{
.devices_for_pipeline(model_name, devices, unet_dtype_str = NULL)
}
\arguments{
\item{model_name}{A character string naming the model, e.g. "sdxl".}

\item{devices}{A device string or named list of component devices.}

\item{unet_dtype_str}{A character string naming the UNet dtype, or NULL.}
}
\value{
The same shape \code{\link{models2devices}} returns:
  \code{devices}, \code{unet_dtype}, \code{device_cpu}, \code{device_cuda}.
}
\description{
The half of \code{\link{models2devices}} that does not touch the disk.
}
\details{
\code{models2devices()} ends by calling \code{download_model()}, which
resolves the TorchScript \code{.pt} files for the model and stops with
"Missing model files" when they are absent. That is correct when it is
about to load them, and wrong when the caller already holds a pipeline:
a native safetensors pipeline never reads a \code{.pt}, so verifying them
makes a working generation depend on files it does not use. Passing
\code{download_models = FALSE} does not avoid it -- the check runs either
way and only the downloading is suppressed.

So callers with a pipeline in hand take this path and get the same four
fields without the file check.

}
\keyword{internal}
