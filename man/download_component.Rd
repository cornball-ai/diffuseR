% tinyrox says don't edit this manually, but it can't stop you!
\name{download_component}
\alias{download_component}
\title{Download a single TorchScript model component}
\usage{
download_component(
  model_name = "sd21",
  component,
  device = "cpu",
  overwrite = FALSE,
  show_progress = TRUE
)
}
\arguments{
\item{model_name}{Character string, the name of the model (e.g., \code{"sd21"}).}

\item{component}{Character string, the component to download (e.g., \code{"unet"}, \code{"decoder"}).}

\item{device}{Character string, the device type (e.g., \code{"cpu"} or \code{"cuda"}).}

\item{overwrite}{Logical; if \code{TRUE}, force re-download even if cached.}

\item{show_progress}{Logical; if \code{TRUE} (default), displays progress during download.}
}
\value{
The local file path to the downloaded component (character string).
}
\description{
Downloads a specific model component file (e.g., UNet, decoder, text encoder)
using \code{hfhub::hub_download()} from the cornball-ai dataset repos.
}
\examples{
\dontrun{
path <- download_component("sd21", "text_encoder", "cpu")
}
}
