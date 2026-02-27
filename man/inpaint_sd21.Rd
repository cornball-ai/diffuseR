% tinyrox says don't edit this manually, but it can't stop you!
\name{inpaint_sd21}
\alias{inpaint_sd21}
\title{Inpaint an image using Stable Diffusion 2.1}
\usage{
inpaint_sd21(
  input_image,
  mask_image,
  prompt,
  negative_prompt = NULL,
  img_dim = 512,
  pipeline = NULL,
  devices = "auto",
  unet_dtype_str = NULL,
  download_models = FALSE,
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
\item{input_image}{Path to the input image (.jpg/.jpeg/.png) or a 3D array.}

\item{mask_image}{Path to the mask image or a matrix/array. White (1) = inpaint,
Black (0) = keep.}

\item{prompt}{Text prompt to guide the inpainting.}

\item{negative_prompt}{Optional negative prompt.}

\item{img_dim}{Dimension of the output image (default: 512).}

\item{pipeline}{Optional pre-loaded pipeline. If NULL, loaded automatically.}

\item{devices}{A named list of devices for each model component, or "auto".}

\item{unet_dtype_str}{Data type for the UNet (e.g., "float16", "float32").}

\item{download_models}{Logical indicating whether to download models if not found.}

\item{num_inference_steps}{Number of diffusion steps (default: 50).}

\item{strength}{Strength of the transformation (default: 0.8). Higher values
change the masked region more.}

\item{guidance_scale}{Scale for classifier-free guidance (default: 7.5).}

\item{seed}{Random seed for reproducibility.}

\item{save_file}{Logical indicating whether to save the generated image.}

\item{filename}{Optional filename for saving the image.}

\item{metadata_path}{Path to save metadata.}

\item{use_native_decoder}{Logical; if TRUE, uses native R torch decoder.}

\item{use_native_text_encoder}{Logical; if TRUE, uses native R torch text encoder.}

\item{use_native_unet}{Logical; if TRUE, uses native R torch UNet.}

\item{...}{Additional arguments.}
}
\value{
A list containing the generated image array and metadata.
}
\description{
Generates a new image by inpainting masked regions of an input image guided
by a text prompt. Uses the standard SD 2.1 pipeline with mask blending at
each denoising step.
}
