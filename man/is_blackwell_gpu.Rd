% tinyrox says don't edit this manually, but it can't stop you!
\name{is_blackwell_gpu}
\alias{is_blackwell_gpu}
\title{Check if GPU is Blackwell Architecture}
\usage{
is_blackwell_gpu()
}
\value{
Logical. TRUE if Blackwell GPU detected.
}
\description{
Blackwell GPUs (RTX 50xx) may need special handling.
}
\examples{
# Soft probe: FALSE on any machine without a Blackwell card, and on
# machines where torch has no lantern binaries.
is_blackwell_gpu()
}
