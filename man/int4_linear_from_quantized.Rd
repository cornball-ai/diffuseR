% tinyrox says don't edit this manually, but it can't stop you!
\name{int4_linear_from_quantized}
\alias{int4_linear_from_quantized}
\title{Create INT4 Linear from Pre-quantized Weights}
\usage{
int4_linear_from_quantized(
  q_weight,
  q_bias = NULL,
  bias_tensor = NULL,
  device = "cuda",
  dtype = torch::torch_float16()
)
}
\arguments{
\item{q_weight}{List with packed, scales, orig_shape from load_int4_weights().}

\item{q_bias}{Optional. Quantized bias (or NULL for no bias).}

\item{bias_tensor}{Optional. Float tensor for bias (if not quantized).}

\item{device}{Character. Target device.}

\item{dtype}{torch_dtype. Target dtype for operations.}
}
\value{
int4_linear module with loaded weights.
}
\description{
Creates an INT4 linear layer from pre-quantized weight data.
}
