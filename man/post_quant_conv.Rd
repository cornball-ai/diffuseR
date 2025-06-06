% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/post_quant_conv.R
\name{post_quant_conv}
\alias{post_quant_conv}
\title{Post Quant Conv}
\usage{
post_quant_conv(x, dtype, device)
}
\arguments{
\item{x}{Input tensor to be processed.}

\item{dtype}{Data type for the tensor (e.g., "torch_float16" or "torch_float32").}

\item{device}{Device on which the tensor is located (e.g., "cpu" or "cuda").}
}
\value{
Processed tensor after applying the quantized convolution.
}
\description{
This function applies a quantized convolution operation to an input tensor.
It is typically used in the context of image post  processing, particularly in
generative models like Stable Diffusion XL.
}
