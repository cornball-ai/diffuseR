% tinyrox says don't edit this manually, but it can't stop you!
\name{linear_to_int4}
\alias{linear_to_int4}
\title{Create INT4 Linear from Standard Linear}
\usage{
linear_to_int4(linear_module, device = "cuda", dtype = torch::torch_float16())
}
\arguments{
\item{linear_module}{nn_linear module to convert.}

\item{device}{Character. Target device for INT4 weights.}

\item{dtype}{torch_dtype. Target dtype for dequantized operations.}
}
\value{
int4_linear module with quantized weights.
}
\description{
Converts a standard nn_linear layer to an INT4 linear layer.
}
