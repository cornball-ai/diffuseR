

#' Preprocess image for Stable Diffusion
#'
#' @param input File path to .jpg or .png, or a 3D array
#' @param device Target device for torch ("cpu" or "cuda")
#' @param width Desired width of the output image
#' @param height Desired height of the output image
#'
#' @return Torch tensor of shape c(1, 3, 512, 512), scaled to c(-1, 1)
#' @export
preprocess_image <- function(
  input,
  device = "cpu",
  width = 512,
  height = 512
) {
  # Load JPEG/PNG if path
  if (is.character(input)) {
    if (grepl("\\.jpg$|\\.jpeg$", input, ignore.case = TRUE)) {
      img <- jpeg::readJPEG(input)
    } else if (grepl("\\.png$", input, ignore.case = TRUE)) {
      img <- png::readPNG(input)
    } else {
      stop("Unsupported image format: only .jpg/.jpeg/.png allowed")
    }
  } else if (is.array(input)) {
    img <- input
  } else {
    stop("Input must be a file path or 3D array")
  }

  # Handle alpha channel if present (RGBA -> RGB)
  if (length(dim(img)) == 3 && dim(img)[3] == 4) {
    img <- img[,, 1:3]# Drop alpha channel
  }

  # Handle grayscale images (add channels dimension)
  if (length(dim(img)) == 2) {
    img <- array(img, dim = c(dim(img), 1))
    # Optionally repeat to make it RGB:
    img <- array(rep(img, 3), dim = c(dim(img)[1], dim(img)[2], 3))
  }

  # Convert to torch tensor
  img_tensor <- torch::torch_tensor(img)

  # Add batch dimension and rearrange to [batch, channel, height, width]
  if (length(dim(img_tensor)) == 3) {
    # [H, W, C] -> [1, C, H, W]
    img_tensor <- img_tensor$permute(c(3, 1, 2)) $unsqueeze(1)
  } else if (length(dim(img_tensor)) == 4) {
    # Assume [B, H, W, C] -> [B, C, H, W]
    img_tensor <- img_tensor$permute(c(1, 4, 2, 3))
  }

  # Use torch's interpolation for resizing
  img_tensor <- torch::nnf_interpolate(img_tensor,
    size = c(height, width),
    mode = "bilinear",
    align_corners = FALSE) # Usually FALSE for stable diffusion

  # Convert to float and normalize to [-1, 1]
  img_tensor <- img_tensor$to(dtype = torch::torch_float()) $to(device = device)
  img_tensor <- (img_tensor * 2) - 1

  return(img_tensor)
}

