% tinyrox says don't edit this manually, but it can't stop you!
\name{models2devices}
\alias{models2devices}
\title{models2devices}
\usage{
models2devices(
  model_name,
  devices = "cpu",
  unet_dtype_str = NULL,
  download_models = FALSE
)
}
\arguments{
\item{model_name}{A character string representing the name of the model to be used.}

\item{devices}{A character string or a named list specifying the devices for different components of the model.}

\item{unet_dtype_str}{A character string specifying the data type for the UNet model.}

\item{download_models}{Logical indicating whether to download models if they are not found.}
}
\value{
A list containing the device configuration, UNet data type, and CPU/CUDA devices.
}
\description{
This function sets up the model directory, device configuration, and data types for diffusion models.
It checks the validity of the model name and devices, detects model type, and downloads the model if necessary.
}
