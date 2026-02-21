% tinyrox says don't edit this manually, but it can't stop you!
\name{download_model}
\alias{download_model}
\title{Download TorchScript model files for Stable Diffusion}
\usage{
download_model(
  model_name = "sd21",
  devices = list(unet = "cpu", decoder = "cpu", text_encoder = "cpu"),
  unet_dtype_str = NULL,
  overwrite = FALSE,
  show_progress = TRUE,
  download_models = FALSE
)
}
\arguments{
\item{model_name}{Name of the model (e.g., "sd21" for stable-diffusion-2-1)}

\item{devices}{Either a single device string or a named list with elements 'unet', 'decoder', 'text_encoder'; optionally 'encoder'}

\item{unet_dtype_str}{Optional: "float16" or "float32" (only applies if unet uses CUDA)}

\item{overwrite}{If TRUE, force re-download of model files}

\item{show_progress}{Show download progress messages}

\item{download_models}{If TRUE, download the model files from HuggingFace}
}
\value{
A named list of full file paths, keyed by component name.
}
\description{
Downloads the required model files (e.g., UNet, decoder, text encoder)
for a given Stable Diffusion model using \code{hfhub::hub_download()}.
Files are cached by hfhub (typically \code{~/.cache/huggingface/hub/}).
Legacy files in the old \code{R_user_dir()} location are also recognized.
}
\examples{
\dontrun{
paths <- download_model("sd21")
}

}
