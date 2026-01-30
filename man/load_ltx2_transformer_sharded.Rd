% tinyrox says don't edit this manually, but it can't stop you!
\name{load_ltx2_transformer_sharded}
\alias{load_ltx2_transformer_sharded}
\title{Load sharded transformer weights}
\usage{
load_ltx2_transformer_sharded(
  transformer,
  weights_dir,
  index_path,
  verbose = TRUE
)
}
\arguments{
\item{transformer}{LTX2 transformer module}

\item{weights_dir}{Directory containing sharded safetensors}

\item{index_path}{Path to index.json}

\item{verbose}{Print progress}
}
\description{
Load sharded transformer weights
}
\keyword{internal}
