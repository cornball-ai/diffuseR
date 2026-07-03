% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_ada_layer_norm_single}
\alias{ltx23_ada_layer_norm_single}
\title{Adaptive layer norm single (adaLN-single)}
\usage{
ltx23_ada_layer_norm_single(embedding_dim, num_mod_params = 6L)
}
\arguments{
\item{embedding_dim}{Integer. Model dimension.}

\item{num_mod_params}{Integer. Number of modulation parameter vectors.}
}
\value{
Module whose forward returns
  \code{list(mod_params [N, num_mod_params * dim], embedded_timestep [N, dim])}.
}
\description{
Embeds a timestep/sigma and projects it to a configurable number of
modulation parameter vectors.
}
