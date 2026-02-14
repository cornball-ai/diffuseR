% tinyrox says don't edit this manually, but it can't stop you!
\name{hf_download_pt}
\alias{hf_download_pt}
\title{Download a single TorchScript component via hfhub}
\usage{
hf_download_pt(model_name, filename, download = TRUE)
}
\arguments{
\item{model_name}{Model name (e.g., \code{"sd21"}, \code{"sdxl"}).}

\item{filename}{Filename within the repo (e.g., \code{"unet-cpu.pt"}).}

\item{download}{If \code{TRUE}, download from HuggingFace when not cached.}
}
\value{
The local file path (character string).
}
\description{
Internal helper that wraps \code{hfhub::hub_download()} with the cornball-ai
dataset repo naming convention. Falls back to legacy \code{R_user_dir()} paths
if the file exists there but not yet in the hfhub cache.
}
\keyword{internal}
