#' Generate an image from a text prompt using a diffusion pipeline
#'
#' @param prompt A character string prompt describing the image to generate.
#' @param model_name Name of the model to use (e.g., `"sd21"`).
#' @param ... Additional parameters passed to the diffusion process.
#'
#' @return A tensor or image object, depending on implementation.
#' @export
#'
#' @examples
#' \dontrun{
#' img <- txt2img("a cat wearing sunglasses in space", device = "cuda")
#' }
txt2img <- function(prompt,
                    model_name = "sdxl",
                    ...) {
  switch(model_name,
         "sd15" = txt2img_sd15(prompt, ...),
         "sd21" = txt2img_sd21(prompt, ...),
         "sdxl" = txt2img_sdxl(prompt, ...),
         "sd3" = txt2img_sd3(prompt, ...),
         stop("Unsupported model: ", model_name)
  )
}
