% tinyrox says don't edit this manually, but it can't stop you!
\name{zimage_img_pos_ids}
\alias{zimage_img_pos_ids}
\title{Build Z-Image latent image position ids}
\usage{
zimage_img_pos_ids(h_tokens, w_tokens, start0, f_tokens = 1L, device = "cpu")
}
\arguments{
\item{h_tokens}{Integer. Token grid height (latent height / patch).}

\item{w_tokens}{Integer. Token grid width (latent width / patch).}

\item{start0}{Integer. First-axis start, cap_padded_len + 1.}

\item{f_tokens}{Integer. Token grid frames; 1 for txt2img.}

\item{device}{Device for the resulting tensor.}
}
\value{
Float tensor of shape [padded token count, 3].
}
\description{
Image tokens use axis 1 for the frame index offset past the caption
(start0 = cap_padded_len + 1), axis 2 for the token row and axis 3 for
the token column. Trailing pad tokens (token count not a multiple of
32) sit at (0, 0, 0). Reference: patchify_and_embed / _pad_with_ids.
}
