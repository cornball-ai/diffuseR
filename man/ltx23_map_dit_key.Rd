% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_map_dit_key}
\alias{ltx23_map_dit_key}
\title{Map an official DiT checkpoint key to the R module name}
\usage{
ltx23_map_dit_key(key)
}
\arguments{
\item{key}{Character. Checkpoint key (with or without the
\code{model.diffusion_model.} prefix).}
}
\value{
Character. Module parameter/buffer name.
}
\description{
Applies the official-to-diffusers renames for the LTX-2.3 transformer
(cf. diffusers scripts/convert_ltx2_to_diffusers.py). Our module tree
matches the diffusers names, so this is the full mapping.
}
