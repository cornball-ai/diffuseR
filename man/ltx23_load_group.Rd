% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_load_group}
\alias{ltx23_load_group}
\title{Stream a checkpoint key group into a module}
\usage{
ltx23_load_group(ckpt, keys, module, map_key = identity, verbose = TRUE,
                 gc_every = 50L)
}
\arguments{
\item{ckpt}{An \code{ltx23_checkpoint}.}

\item{keys}{Character vector of checkpoint keys to load (one group
from \code{\link{ltx23_split_keys}}).}

\item{module}{A torch nn_module to populate.}

\item{map_key}{Function mapping a checkpoint key to the module's
parameter/buffer name, or NA to skip the key deliberately.}

\item{verbose}{Logical. Report progress and coverage.}

\item{gc_every}{Integer. Run \code{gc()} after this many tensors.}
}
\value{
Invisibly, a list with \code{unmapped} (checkpoint keys that
  found no destination), \code{skipped} (keys the mapper declined),
  and \code{unfilled} (module parameters/buffers never written).
  A perfectly loaded group has zero \code{unmapped} and zero
  \code{unfilled}.
}
\description{
Reads tensors one at a time from an open checkpoint and copies them
into the matching parameters/buffers of \code{module}. Destination
names are derived by \code{map_key}; \code{$copy_()} handles any
dtype/device conversion, so the module may already live on its target
device in its target dtype.
}
