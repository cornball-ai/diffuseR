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
torch::with_no_grad({
  # Auto-detect optimal device configuration (requires gpuctl)
  txt2img_sdxl("A sunset over mountains", filename = "output.png")
})
```

With explicit device configuration:
```r
devices <- list(unet = "cuda", decoder = "cpu",
                text_encoder = "cpu", encoder = "cpu")
txt2img_sdxl("A sunset over mountains", devices = devices)
```

## GPU Poor Support

diffuseR integrates with gpuctl for automatic "GPU poor" support:

```r
# Auto-detect VRAM and select optimal strategy
devices <- auto_devices("sdxl")  # Uses gpuctl if available

# Strategies:
# - full_gpu: All on CUDA (16GB+ VRAM)
# - unet_gpu: Unet on CUDA, rest CPU (8GB+ VRAM) - forced on Blackwell
# - cpu_only: All on CPU
devices <- auto_devices("sdxl", strategy = "unet_gpu")
```

The `txt2img_*` and `img2img` functions default to `devices = "auto"`, which:
1. Uses gpuctl to detect VRAM and GPU architecture
2. Selects optimal strategy (full_gpu, unet_gpu, or cpu_only)
3. Forces unet_gpu on Blackwell GPUs (TorchScript workaround)

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

## Roadmap

### Native Torch Implementation (Priority: High)
- [ ] **Replace TorchScript with native torch**: Load models directly from safetensors/HuggingFace
  - Eliminates GPU architecture compatibility issues (Blackwell, future GPUs)
  - More maintainable than pre-exported .pt files
  - Reference: Python diffusers library implementation

### gpuctl Integration
- [x] **Auto-device configuration**: `auto_devices()` integrates with gpuctl
  - Queries available VRAM via `gpuctl::gpu_detect()`
  - Auto-selects optimal devices based on model requirements
  - Handles Blackwell workaround automatically

### Model Support
- [ ] Add FLUX model support
- [ ] Add SD3 model support
- [ ] ControlNet integration
