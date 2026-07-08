% tinyrox says don't edit this manually, but it can't stop you!
\name{zimage_cap_pos_ids}
\alias{zimage_cap_pos_ids}
\title{Build Z-Image caption position ids}
\usage{
zimage_cap_pos_ids(cap_padded_len, device = "cpu")
}
\arguments{
\item{cap_padded_len}{Integer. Caption length after padding to a
multiple of 32.}

\item{device}{Device for the resulting tensor.}
}
\value{
Float tensor of shape [cap_padded_len, 3].
}
\description{
Caption tokens ramp 1..cap_padded_len on the first axis (axes 2 and 3
zero). The reference builds the grid over the already-padded length,
so pad tokens continue the ramp rather than sitting at the origin
(the (0,0,0) pad ids it also emits are truncated away in
_prepare_sequence and never reach RoPE).
}
