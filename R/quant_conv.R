# quant_conv.R

#' Quant Conv
#'
#' This function applies a quantized convolution operation to an input tensor.
#' It is typically used in the context of image processing, particularly in
#' generative models like Stable Diffusion.
#'
#' @param x Input tensor to be processed.
#' @param dtype Data type for the tensor (e.g., "torch_float16" or "torch_float32").
#' @param device Device on which the tensor is located (e.g., "cpu" or "cuda").
#' @return Processed tensor after applying the quantized convolution.
#' @export
quant_conv <- function(
  x,
  dtype,
  device
) {
  params_path <- system.file("quant_conv/", package = "diffuseR")
  qc_weights <- as.matrix(utils::read.csv(paste0(params_path,
        "/quant_conv_weights.csv"),
      header = FALSE))
  qc_bias <- as.numeric(utils::read.csv(paste0(params_path,
        "/quant_conv_bias.csv"),
      header = FALSE) [[1]])

  # Convert to torch tensors and reshape weights for conv2d
  qc_weights_tensor <- torch::torch_tensor(qc_weights) $view(c(8, 8, 1, 1))
  qc_bias_tensor <- torch::torch_tensor(qc_bias)

  conv <- torch::nnf_conv2d(x, weight = qc_weights_tensor, bias = qc_bias_tensor)
  conv$to(dtype = dtype, device = torch::torch_device(device))
}

