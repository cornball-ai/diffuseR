% tinyrox says don't edit this manually, but it can't stop you!
\name{load_int4_into_model}
\alias{load_int4_into_model}
\title{Load INT4 Weights into Model}
\usage{
load_int4_into_model(
  model,
  int4_weights,
  device = "cuda",
  dtype = torch::torch_float16(),
  verbose = TRUE
)
}
\arguments{
\item{model}{nn_module. The model to convert.}

\item{int4_weights}{List of quantized weights from `load_int4_weights()`.}

\item{device}{Character. Target device for INT4 weights.}

\item{dtype}{torch_dtype. Target dtype for dequantized operations.}

\item{verbose}{Logical. Print progress.}
}
\value{
The model with linear layers replaced by INT4 versions.
}
\description{
Replaces linear layers in a model with INT4 versions and loads quantized weights.
This is the main entry point for running large models with INT4 quantization.
}
\details{
This function:
1. Identifies linear layers by matching parameter names ending in ".weight"
2. Creates INT4Linear layers with matching dimensions
3. Loads quantized weights and biases
4. Replaces the original layers in the model

The INT4 weights stay compressed on GPU (~10GB for a 40GB model).
During forward(), each layer dequantizes on-the-fly, keeping memory usage low.
}
