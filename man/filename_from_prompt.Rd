\name{filename_from_prompt}
\alias{filename_from_prompt}
\title{filename_from_prompt}
\description{filename_from_prompt}
\usage{
filename_from_prompt(prompt, datetime)
}
\arguments{
  \item{prompt}{A character string representing the prompt.}
  \item{datetime}{Logical indicating whether to prepend the current date and time to the filename. Default is TRUE.}
}
\value{
A character string representing the generated filename.
}
\examples{
filename_from_prompt("A beautiful sunset over the mountains")
filename_from_prompt("A beautiful sunset over the mountains", datetime = FALSE)
}
