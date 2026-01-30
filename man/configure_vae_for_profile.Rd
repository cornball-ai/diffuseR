% tinyrox says don't edit this manually, but it can't stop you!
\name{configure_vae_for_profile}
\alias{configure_vae_for_profile}
\title{Configure VAE for Memory Profile}
\usage{
configure_vae_for_profile(vae, profile)
}
\arguments{
\item{vae}{The LTX2 VAE module.}

\item{profile}{Memory profile from `ltx2_memory_profile()`.}
}
\value{
The VAE (modified in place).
}
\description{
Sets VAE tiling parameters based on memory profile.
}
\examples{
\dontrun{
profile <- ltx2_memory_profile(vram_gb = 8)
vae <- load_ltx2_vae(...)
configure_vae_for_profile(vae, profile)
}
}
