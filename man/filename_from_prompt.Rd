% tinyrox says don't edit this manually, but it can't stop you!
\name{filename_from_prompt}
\alias{filename_from_prompt}
\title{Generate a filename from a prompt}
\usage{
filename_from_prompt(prompt, datetime = TRUE)
}
\arguments{
\item{prompt}{A character string representing the prompt.}

\item{datetime}{Logical indicating whether to prepend the current date and time to the filename. Default is TRUE.}
}
\value{
A character string representing the generated filename.
}
\description{
This function generates a filename from a prompt by removing all non-alphanumeric characters and replacing them with underscores. The filename is limited to 50 characters. If `datetime` is set to TRUE, the current date and time are prepended to the filename.
}
\examples{
filename_from_prompt("A beautiful sunset over the mountains")
filename_from_prompt("A beautiful sunset over the mountains", datetime = FALSE)
}
