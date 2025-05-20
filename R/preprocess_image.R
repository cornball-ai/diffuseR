

# For existing image arrays
resize_for_torch <- function(img_array, width, height) {
  # Convert to torch tensor
  img_tensor <- torch::torch_tensor(img_array)
  
  # Add batch dimension if needed and rearrange to [batch, channel, height, width]
  if(length(dim(img_tensor)) == 3) {
    # [H, W, C] -> [1, C, H, W]
    img_tensor <- img_tensor$permute(c(3, 1, 2))$unsqueeze(1)
  }
  
  # Use torch's built-in interpolation
  resized <- torch::nnf_interpolate(img_tensor, 
                             size = c(height, width),
                             mode = "bilinear",
                             align_corners = TRUE)
  
  # Convert back to R array in [H, W, C] format
  result <- resized$squeeze(1)$permute(c(2, 3, 1))
  # plot
  
  return(result)
}


#' Preprocess image for Stable Diffusion
#'
#' @param input File path to .jpg or .png, or a 3D array
#' @param device Target device for torch ("cpu" or "cuda")
#'
#' @return Torch tensor of shape [1, 3, 512, 512], scaled to [-1, 1]
#' @export
preprocess_image <- function(input, device = "cpu", width = 512, height = 512) {
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
  
  # Resize to 512x512 (nearest neighbor, base R)
  img_resized <- resize_for_torch(img_array = img,
                                  width = width,
                                  height = height) # resize_image_array(img, width = 512, height = 512)
  
  # Reorder [H, W, C] → [C, H, W], scale [0,1] → [-1,1]
  # permute using torch
  img_tensor <- torch::torch_tensor(img_resized$permute(c(3, 1, 2)))
  img_tensor <- img_tensor$to(dtype = torch::torch_float())$to(device = device)
  img_tensor <- (img_tensor * 2) - 1
  img_tensor <- img_tensor$unsqueeze(1)  # [1, 3, 512, 512]
  
  return(img_tensor)
}
