% tinyrox says don't edit this manually, but it can't stop you!
\name{zimage_unpatchify}
\alias{zimage_unpatchify}
\title{Unpatchify Z-Image tokens back to a latent image}
\usage{
zimage_unpatchify(tokens, size, patch_size = 2L, f_patch_size = 1L,
                  out_channels = 16L)
}
\arguments{
\item{tokens}{Tensor of shape [S, pF * p * p * C] with the image
tokens first.}

\item{size}{Integer vector c(F, H, W) of the target latent size.}

\item{patch_size}{Integer spatial patch size. Default 2.}

\item{f_patch_size}{Integer temporal patch size. Default 1.}

\item{out_channels}{Integer number of latent channels. Default 16.}
}
\value{
Tensor of shape [C, F, H, W].
}
\description{
Takes the first F/pF * H/p * W/p tokens (the image span of the
unified sequence) and reassembles [C, F, H, W], matching unpatchify.
}
