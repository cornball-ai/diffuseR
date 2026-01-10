\name{txt2img_sdxl}
\alias{txt2img_sdxl}
\title{Generate an image from a text prompt using a diffusion pipeline}
\description{Generate an image from a text prompt using a diffusion pipeline}
\usage{
txt2img_sdxl(prompt, negative_prompt, img_dim, pipeline, devices, unet_dtype_str, download_models, scheduler, timesteps, initial_latents, num_inference_steps, guidance_scale, seed, save_file, filename, metadata_path, ...)
}
\arguments{
  \item{prompt}{A character string prompt describing the image to generate.}
  \item{negative_prompt}{Optional negative prompt to guide the generation.}
  \item{img_dim}{Dimension of the output image (e.g., 512 for 512x512).}
  \item{pipeline}{Optional A pre-loaded diffusion pipeline. If `NULL`, it will be loaded based on the model name and devices.}
  \item{devices}{A named list of devices for each model component (e.g., `list(unet = "cuda", decoder = "cpu", text_encoder = "cpu")`).}
  \item{unet_dtype_str}{Optional A character for dtype of the unet component (typically "float16" for cuda and "float32" for cpu; float32 is available for cuda).}
  \item{download_models}{Logical indicating whether to download the model files if they are not found.}
  \item{scheduler}{Scheduler to use (e.g., `"ddim"`, `"euler"`).}
  \item{timesteps}{Optional A vector of timesteps to use.}
  \item{initial_latents}{Optional initial latents for the diffusion process.}
  \item{num_inference_steps}{Number of inference steps to run.}
  \item{guidance_scale}{Scale for classifier-free guidance (typically 7.5).}
  \item{seed}{Optional seed for reproducibility.}
  \item{save_file}{Logical indicating whether to save the generated image.}
  \item{filename}{Optional filename for saving the image. If `NULL`, a default name is generated.}
  \item{metadata_path}{Optional file path to save metadata.}
  \item{...}{Additional parameters passed to the diffusion process.}
}
\value{
An image array and metadata
}
\examples{
\dontrun{
img <- txt2img("a cat wearing sunglasses in space", device = "cuda")
}
}
