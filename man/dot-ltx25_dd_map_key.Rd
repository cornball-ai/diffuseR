% tinyrox says don't edit this manually, but it can't stop you!
\name{.ltx25_dd_map_key}
\alias{.ltx25_dd_map_key}
\title{Map an LTX-2.5 diffusion decoder checkpoint key}
\usage{
.ltx25_dd_map_key(key)
}
\arguments{
\item{key}{Checkpoint key (\code{decoder.*} or
\code{per_channel_statistics.*}, optionally under \code{vae.}).}
}
\value{
The module key, or \code{NA} for keys the decoder does not use
  (the encoder, and \code{type_emb}, which tags keyframe latents in
  Lightricks' keyframe-aware decode).
}
\description{
Map an LTX-2.5 diffusion decoder checkpoint key
}
\keyword{internal}
