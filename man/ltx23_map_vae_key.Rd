% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_map_vae_key}
\alias{ltx23_map_vae_key}
\title{Map an official VAE checkpoint key to the R module name}
\usage{
ltx23_map_vae_key(key)
}
\arguments{
\item{key}{Character. Checkpoint key (with or without "vae." prefix).}
}
\value{
Character. Module parameter/buffer name.
}
\description{
The official checkpoint stores the encoder/decoder as flat block lists
(down_blocks.0-8 / up_blocks.0-8) where downsamplers/upsamplers and
the mid block are separate entries; diffusers (and this port) nest
them. Index mapping per diffusers convert_ltx2_to_diffusers.py.
}
