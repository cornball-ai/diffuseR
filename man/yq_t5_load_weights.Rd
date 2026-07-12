% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_t5_load_weights}
\alias{yq_t5_load_weights}
\title{Load T5 encoder weights into an anvl pytree}
\usage{
yq_t5_load_weights(path, num_layers = 24L, device = "cpu")
}
\arguments{
\item{path}{safetensors file with the T5 encoder state_dict.}

\item{num_layers}{Integer. Encoder blocks to load.}

\item{device}{Character. Target device.}
}
\value{
List \code{list(embed = <[vocab, d_model] R matrix>,
  rel_bias = <[num_buckets, num_heads] R matrix>, layers = <per-layer
  weight lists>, final_ln = <AnvlArray>)}.
}
\description{
Reads a T5 state_dict from a safetensors file (F32/F16/BF16), keys
mirroring \code{\link{t5_encoder}}'s module tree. Linear weights are
transposed to \code{[in, out]} for \code{\link[yunque]{yq_linear}};
the shared embedding and the relative-position bias embedding stay R
matrices for host-side gather. Strict census: every requested key must
exist.
}
