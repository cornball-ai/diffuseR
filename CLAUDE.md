# diffuseR

Native R package for Stable Diffusion image generation.

## Key Functions

| Function | Purpose |
|----------|---------|
| `txt2img_sdxl()` | Generate image from text (SDXL) |
| `txt2img_sd21()` | Generate image from text (SD 2.1) |
| `img2img()` | Image-to-image generation |
| `load_pipeline()` | Pre-load models for reuse |
| `models2devices()` | Configure device placement |

## Usage

```r
library(diffuseR)
torch::local_no_grad()

devices <- list(unet = "cuda", decoder = "cpu",
                text_encoder = "cpu", encoder = "cpu")

m2d <- models2devices("sdxl", devices, unet_dtype_str = "float16",
                      download_models = TRUE)
pipeline <- load_pipeline("sdxl", m2d, unet_dtype_str = "float16")

txt2img_sdxl("A sunset over mountains", devices = devices,
             pipeline = pipeline, filename = "output.png")
```

## Known Issues / TODOs

### Blackwell GPU Compatibility (CRITICAL)

**Problem**: TorchScript models (.pt files) fail on Blackwell GPUs (RTX 50xx) when all components run on CUDA.

**Current workaround**: Only unet on CUDA, rest on CPU (see usage above).

**TODO**: Replace TorchScript approach with proper torch code that loads models directly. TorchScript is brittle across GPU generations; native torch loading from safetensors/HuggingFace would be more portable and maintainable.

### Model Files (Current - to be replaced)

Models stored in `~/.local/share/R/diffuseR/{model_name}/`:
- `unet-{device}-{dtype}.pt` (TorchScript)
- `decoder-{device}.pt` (TorchScript)
- `text_encoder-{device}.pt` (TorchScript)
- `encoder-{device}.pt` (TorchScript)

Downloaded from: `huggingface.co/datasets/cornball-ai/sdxl-R`
