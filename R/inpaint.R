#' Inpaint an image using a diffusion model
#'
#' Generates a new image by inpainting masked regions of an input image guided
#' by a text prompt. Unmasked regions are preserved from the original image.
#'
#' @param input_image Path to the input image (.jpg/.jpeg/.png) or a 3D array.
#' @param mask_image Path to the mask image or a matrix/array. White (1) = inpaint,
#'   Black (0) = keep.
#' @param prompt Text prompt to guide the inpainting.
#' @param model_name Name of the model to use (currently "sd21").
#' @param ... Additional parameters passed to the model-specific inpainting function.
#'
#' @return A list containing the generated image array and metadata.
#' @export
inpaint <- function(
  input_image,
  mask_image,
  prompt,
  model_name = "sd21",
  ...
) {
  switch(model_name,
    "sd21" = inpaint_sd21(input_image, mask_image, prompt, ...),
    stop("Unsupported model for inpainting: ", model_name)
  )
}

#' Preprocess a mask for inpainting
#'
#' Loads, resizes, and binarizes a mask image for use in the inpainting pipeline.
#' The mask is resized to latent space dimensions (height/8, width/8).
#'
#' @param mask_input Path to a mask image (.jpg/.jpeg/.png), a matrix, or a 3D array.
#'   White (1) = inpaint region, Black (0) = keep region.
#' @param height Target image height in pixels (will be divided by 8 for latent space).
#' @param width Target image width in pixels (will be divided by 8 for latent space).
#' @param device Target device ("cpu" or "cuda").
#' @param dtype Torch dtype for the output tensor.
#'
#' @return Torch tensor of shape [1, 1, height/8, width/8] with values 0 or 1.
#' @export
preprocess_mask <- function(
  mask_input,
  height,
  width,
  device = "cpu",
  dtype = torch::torch_float32()
) {
  # Load mask image
  if (is.character(mask_input)) {
    if (grepl("\\.jpg$|\\.jpeg$", mask_input, ignore.case = TRUE)) {
      mask <- jpeg::readJPEG(mask_input)
    } else if (grepl("\\.png$", mask_input, ignore.case = TRUE)) {
      mask <- png::readPNG(mask_input)
    } else {
      stop("Unsupported mask format: only .jpg/.jpeg/.png allowed")
    }
  } else if (is.matrix(mask_input) || is.array(mask_input)) {
    mask <- mask_input
  } else {
    stop("mask_input must be a file path, matrix, or array")
  }

  # Convert to single-channel if multi-channel (use first channel or average)
  if (length(dim(mask)) == 3) {
    # If RGBA, drop alpha
    if (dim(mask)[3] == 4) {
      mask <- mask[,, 1:3]
    }
    # Average RGB channels to get single-channel mask
    mask <- apply(mask, c(1, 2), mean)
  }

  # Convert to tensor [1, 1, H, W]
  mask_tensor <- torch::torch_tensor(mask)$unsqueeze(1)$unsqueeze(1)

  # Resize to latent dimensions (H/8, W/8)
  latent_h <- as.integer(height / 8)
  latent_w <- as.integer(width / 8)
  mask_tensor <- torch::nnf_interpolate(
    mask_tensor,
    size = c(latent_h, latent_w),
    mode = "nearest"
  )

  # Binarize: threshold at 0.5 (white=1=inpaint, black=0=keep)
  mask_tensor <- (mask_tensor > 0.5)$to(dtype = dtype, device = torch::torch_device(device))

  mask_tensor
}
