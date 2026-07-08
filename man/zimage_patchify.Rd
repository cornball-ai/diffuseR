% tinyrox says don't edit this manually, but it can't stop you!
\name{zimage_patchify}
\alias{zimage_patchify}
\title{Patchify a latent image to Z-Image tokens}
\usage{
zimage_patchify(image, patch_size = 2L, f_patch_size = 1L)
}
\arguments{
\item{image}{Tensor of shape [C, F, H, W].}

\item{patch_size}{Integer spatial patch size. Default 2.}

\item{f_patch_size}{Integer temporal patch size. Default 1.}
}
\value{
Tensor of shape [num_tokens, patch_dim].
}
\description{
(C, F, H, W) -> [F/pF * H/p * W/p, pF * p * p * C], matching
_patchify_image. No padding is applied here.
}
