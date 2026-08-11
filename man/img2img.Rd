% tinyrox says don't edit this manually, but it can't stop you!
\name{img2img}
\alias{img2img}
\title{Image-to-Image Generation with Stable Diffusion}
\usage{
img2img(
  input_image,
  prompt,
  negative_prompt = NULL,
  img_dim = 512,
  model_name = c("sd21", "sdxl"),
  pipeline = NULL,
  devices = "auto",
  unet_dtype_str = "float16",
  download_models = FALSE,
  scheduler = "ddim",
  num_inference_steps = 50,
  strength = 0.8,
  guidance_scale = 7.5,
  seed = NULL,
  save_file = TRUE,
  filename = NULL,
  metadata_path = NULL,
  use_native_decoder = FALSE,
  use_native_text_encoder = FALSE,
  use_native_unet = FALSE,
  ...
)
}
\arguments{
\item{input_image}{Path to the input image or a tensor representing the image.}

\item{prompt}{Text prompt to guide the image generation.}

\item{negative_prompt}{Optional negative prompt to guide the image generation.`}

\item{img_dim}{Dimension of the output image (default: 512).}

\item{model_name}{Name of the Stable Diffusion model to use (default: "sd21").}

\item{pipeline}{Optional pre-loaded pipeline. If `NULL`, it will be loaded based on `model_name`.}

\item{devices}{A named list of devices for each model component (e.g., `list(unet = "cuda", decoder = "cpu", text_encoder = "cpu", encoder = "cpu")`).}

\item{unet_dtype_str}{Optional A character for dtype of the unet component (typically "torch_float16" for cuda and "torch_float32" for cpu).}

\item{download_models}{Logical indicating whether to download models if not found (default: FALSE).}

\item{scheduler}{Scheduler to use for the diffusion process (default: "ddim").}

\item{num_inference_steps}{Number of diffusion steps (default: 50).}

\item{strength}{Strength of the image-to-image transformation (default: 0.8).}

\item{guidance_scale}{Scale for classifier-free guidance (default: 7.5).}

\item{seed}{Random seed for reproducibility (default: NULL).}

\item{save_file}{Logical indicating whether to save the generated image.}

\item{filename}{Optional filename for saving the image. If `NULL`, a default name is generated.}

\item{metadata_path}{Path to save metadata (default: NULL).}

\item{use_native_decoder}{Logical; if TRUE, uses native R torch decoder instead of TorchScript.
Native decoder has better GPU compatibility (especially Blackwell).}

\item{use_native_text_encoder}{Logical; if TRUE, uses native R torch text encoder instead of TorchScript.
Native text encoder has better GPU compatibility (especially Blackwell).}

\item{use_native_unet}{Logical; if TRUE, uses native R torch UNet instead of TorchScript.
Native UNet has better GPU compatibility (especially Blackwell).}

\item{...}{Additional arguments for future use.}
}
\value{
An image array and metadata
}
\description{
This function generates an image based on an input image and a text prompt using the Stable Diffusion model.
It allows for various configurations such as model name, device, scheduler, and more.
}
