# diffuseR

[![CRAN status](https://www.r-pkg.org/badges/version/diffuseR)](https://CRAN.R-project.org/package=diffuseR)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)

## Overview

`diffuseR` is a functional R implementation of diffusion models, inspired by Hugging Face's Python `diffusers` library. The package provides a simple, idiomatic R interface to state-of-the-art generative AI models for image generation and manipulation.

```r
# Simple text-to-image generation
library(diffuseR)
image <- txt2img("A serene landscape with mountains and a lake at sunset")
plot(image)
```

## Installation

You can install the development version of diffuseR from GitHub:

```r
# install.packages("devtools")
devtools::install_github("yourusername/diffuseR")
```

## Features

- **Text-to-Image Generation**: Create images from textual descriptions
- **Multiple Models**: Support for Stable Diffusion 2.1 and more
- **Scheduler Options**: DDIM, DPM-Solver+ (more coming soon)
- **Device Support**: Works on both CPU and GPU (via TorchScript)
- **R-native Interface**: Functional programming approach that feels natural in R

## Quick Start

### Basic Usage

```r
library(diffuseR)

# Generate an image from text
cat_img <- txt2img(
  prompt = "a photorealistic cat wearing sunglasses",
  steps = 30,
  save_to = "cat.png"
)

# Display the image in R
plot(cat_img)
```

### Advanced Usage with Pipelines

```r
# Load and configure a pipeline once
sd <- load_pipeline("stable-diffusion-2-1", device = "cuda")

# Change the scheduler
sd$scheduler <- create_scheduler("dpm_solver_plus", num_inference_steps = 20)

# Generate multiple images with the same configuration
mountain_img <- generate(sd, prompt = "mountains at sunset")
ocean_img <- generate(sd, prompt = "waves crashing on the shore")

# Save images
save_image(mountain_img, "mountain.png")
save_image(ocean_img, "ocean.png")
```

## Supported Models

Currently supported models:

- Stable Diffusion 2.1
- More coming soon!

## Roadmap

Future plans for diffuseR include:

- [ ] Image-to-image generation
- [ ] Inpainting support
- [ ] Additional schedulers (K-LMS, PLMS, Euler ancestral)
- [ ] Model fine-tuning capabilities
- [ ] ControlNet and other extensions

## How It Works

diffuseR uses TorchScript models exported from PyTorch implementations, wrapped in a functional R interface. This approach balances performance and ease of use while feeling natural to R users.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache 2. License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the original diffusers library
- Stability AI for Stable Diffusion
- The R and torch communities for their excellent tooling