#' Save and Display an Image from a Torch Tensor
#'
#' Converts a Torch tensor to a normalized RGB image array, saves it as a PNG file,
#' and optionally displays it in the RStudio Viewer pane using `grid::grid.raster()`.
#'
#' @param img A numeric with shape `[3, H, W]`.
#' @param save_to File path for the PNG image (default is `"output.png"`).
#' @param normalize Logical; whether to normalize pixel values to `[0, 1]`. Default is `TRUE`.
#'
#' @return Invisibly returns the saved file path.
#' @export
#'
#' @examples
#' \dontrun{
#' save_image(output_tensor, "sample.png")
#' }
save_image <- function(img, save_to = "output.png", normalize = TRUE) {
  # img_array <- tensor2image(img, normalize = normalize)
  dims <- dim(img)
  
  grDevices::png(filename = save_to, width = dims[2], height = dims[1])
  grid::grid.raster(img)
  grDevices::dev.off()
  
  if (interactive()) {
    grid::grid.raster(img)
  }
  
  cat("Image saved to", save_to, "\n")
  invisible(save_to)
}
