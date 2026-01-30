% tinyrox says don't edit this manually, but it can't stop you!
\name{make_linear}
\alias{make_linear}
\title{Create Linear Layer (Standard or INT4)}
\usage{
make_linear(in_features, out_features, bias = TRUE)
}
\arguments{
\item{in_features}{Integer. Input dimension.}

\item{out_features}{Integer. Output dimension.}

\item{bias}{Logical. Include bias term.}
}
\value{
nn_linear or int4_linear module.
}
\description{
Factory function that creates either a standard nn_linear or INT4 linear layer
based on package options. Use this instead of torch::nn_linear() in model code.
}
\details{
Behavior controlled by options:
- `diffuseR.use_int4`: If TRUE, create INT4 layer (default FALSE)
- `diffuseR.int4_device`: Device for INT4 layers (default "cuda")
- `diffuseR.int4_dtype`: Dtype for INT4 operations (default torch_float16())
}
