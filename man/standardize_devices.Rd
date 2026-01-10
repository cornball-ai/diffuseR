\name{standardize_devices}
\alias{standardize_devices}
\title{Standardize devices configuration}
\description{This function standardizes the device configuration for model components.
It checks if the devices parameter is a single string or a named list, and fills in missing components with reasonable defaults.}
\usage{
standardize_devices(devices, required_components)
}
\arguments{
  \item{devices}{A character string or a named list specifying the devices for model components.}
  \item{required_components}{A character vector of required components for the model.}
}
\value{
A named list of devices for each required component.
}
