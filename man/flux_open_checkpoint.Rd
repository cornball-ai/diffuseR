% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_open_checkpoint}
\alias{flux_open_checkpoint}
\title{Open a FLUX transformer checkpoint directory}
\usage{
flux_open_checkpoint(transformer_dir)
}
\arguments{
\item{transformer_dir}{Directory containing \code{config.json} and the
\code{diffusion_pytorch_model*.safetensors} file(s).}
}
\value{
An object of class \code{ltx23_checkpoint} (shared checkpoint
  interface): list with \code{handle$get_tensor}, \code{keys},
  \code{config}, and \code{path}.
}
\description{
Opens a diffusers-layout transformer directory lazily (headers only).
Sharded checkpoints are resolved through the index.json weight map;
single-file checkpoints are opened directly. The transformer
\code{config.json} is attached as \code{$config}.
}
