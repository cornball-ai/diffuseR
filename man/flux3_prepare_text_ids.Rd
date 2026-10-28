% tinyrox says don't edit this manually, but it can't stop you!
\name{flux3_prepare_text_ids}
\alias{flux3_prepare_text_ids}
\title{FLUX 3 text token position ids}
\usage{
flux3_prepare_text_ids(len, device = "cpu")
}
\arguments{
\item{len}{Integer. Text sequence length (padded).}

\item{device}{Character. Device.}
}
\value{
Float tensor [len, 4]: (0, 0, 0, 0..len-1).
}
\description{
FLUX 3 text token position ids
}
