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

# Strategies (SDXL thresholds):
# - full_gpu: All on CUDA (10GB+ VRAM)
# - unet_gpu: Unet on CUDA, rest CPU (6GB+ VRAM) - forced on Blackwell
# - cpu_only: All on CPU (<6GB VRAM)
devices <- auto_devices("sdxl", strategy = "unet_gpu")
```

The `txt2img_*` and `img2img` functions default to `devices = "auto"`, which:
1. Uses gpuctl to detect VRAM and GPU architecture
2. Selects optimal strategy (full_gpu, unet_gpu, or cpu_only)
3. Forces unet_gpu on Blackwell GPUs (TorchScript workaround)

## GPU-Poor Execution Plan (TODO)

Profile-based memory optimization for constrained GPUs. Inspired by mmgp/Wan2GP approach.

### API

```r
txt2img_sdxl("A cat", profile = "auto")  # Default: auto-detect via gpuctl
txt2img_sdxl("A cat", profile = "gpu_poor")  # Force low-memory mode
txt2img_sdxl("A cat", profile = "full_gpu", vram_debug = TRUE)  # Debug VRAM usage
```

### Profiles

| Profile | VRAM | Devices | CFG Mode | Cleanup | Use Case |
|---------|------|---------|----------|---------|----------|
| `full_gpu` | 16GB+ | All CUDA | batched | none | RTX 4090, 5080, 5090 |
| `balanced` | 10-12GB | UNet+decoder CUDA | batched | phase | RTX 3080, 4070 Ti |
| `gpu_poor` | 6-8GB | UNet CUDA only | sequential | phase | RTX 3060, 4060 |
| `extreme` | <6GB | UNet CUDA only | sequential | step | GTX 1660, laptops |
| `cpu_only` | 0 | All CPU | sequential | none | No GPU / testing |

### Profile Details

**full_gpu** (16GB+)
- All components on CUDA
- Batched CFG (uncond+cond in single forward pass)
- No cleanup overhead
- Fastest execution

**balanced** (10-12GB)
- UNet and decoder on CUDA, text encoders on CPU
- Batched CFG
- Phase cleanup between denoise and decode
- Good speed/memory tradeoff

**gpu_poor** (6-8GB)
- Only UNet on CUDA, everything else CPU
- Sequential CFG (halves peak activation memory)
- Phase cleanup + UNet→CPU swap before decode
- Slower but fits in limited VRAM

**extreme** (<6GB)
- Only UNet on CUDA
- Sequential CFG
- Step-level cleanup (gc + cuda_empty_cache each step)
- Slowest but minimum peak memory

### Implementation Components

#### 1. Profile Resolution

```r
resolve_profile <- function(profile = "auto", model = "sdxl") {
  if (profile == "auto") {
    vram <- gpuctl::gpu_detect()$vram_gb
    profile <- if (vram >= 16) "full_gpu"
               else if (vram >= 10) "balanced"
               else if (vram >= 6) "gpu_poor"
               else if (vram > 0) "extreme"
               else "cpu_only"
  }
  get_profile_config(profile, model)
}
```

#### 2. Sequential CFG Mode

Run uncond and cond UNet passes separately instead of batched:

```r
if (cfg_mode == "sequential") {
  noise_pred_uncond <- unet(latents, t, empty_prompt, ...)
  noise_pred_cond <- unet(latents, t, prompt, ...)
  noise_pred <- noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
  rm(noise_pred_uncond, noise_pred_cond)  # Free immediately
}
```

Halves peak activation memory during UNet forward pass.

#### 3. Phase Cleanup

Clean up between denoise and decode phases:

```r
if (cleanup %in% c("phase", "step")) {
  # Swap UNet to CPU before decode
  pipeline$unet$to(device = "cpu")
  rm(noise_pred, timestep, ...)
  gc()
  torch::cuda_empty_cache()
}
```

#### 4. Step Cleanup (extreme mode)

Most aggressive - cleanup after each denoising step:

```r
for (i in seq_along(timesteps)) {
  # ... denoising step ...
  if (cleanup == "step") {
    rm(noise_pred, timestep)
    gc()
    torch::cuda_empty_cache()
  }
}
```

#### 5. Function Isolation

Extract denoise and decode into separate functions so tensors go out of scope:

```r
run_denoise_sdxl <- function(latents, ...) {
  # Denoise loop
  # Returns ONLY latents (all other tensors die when function exits)
  latents
}

run_decode_sdxl <- function(latents, decoder, device) {
  # Decode latents to image
  # Returns R array (all torch tensors die when function exits)
  img_array
}
```

#### 6. VRAM Debug

```r
vram_report <- function(label) {
  allocated <- torch::cuda_memory_allocated() / 1024^3
  reserved <- torch::cuda_memory_reserved() / 1024^3
  message(sprintf("[%s] VRAM: %.2f GB allocated, %.2f GB reserved",
                  label, allocated, reserved))
}
```

### Future Optimizations (from mmgp)

Not implemented, but worth considering:

- **Pinned CPU memory**: `$pin_memory()` for faster CPU↔GPU transfers
- **Async prefetch**: Load next component while current one computes (needs torch streams)
- **Resident set policy**: Keep hot components on GPU, evict cold ones (useful for batch generation)

## Native Torch Migration (Complete)

Replaced TorchScript with native R torch modules for Blackwell GPU compatibility.

### Component Status

| Component | Status | Notes |
|-----------|--------|-------|
| VAE Decoder | ✅ Complete | `use_native_decoder = TRUE` |
| Text Encoder | ✅ Complete | `use_native_text_encoder = TRUE`, auto-detects architecture |
| Text Encoder 2 | ✅ Complete | SDXL's OpenCLIP ViT-bigG |
| UNet (SD21) | ✅ Complete | `use_native_unet = TRUE`, 686 parameters |
| UNet (SDXL) | ✅ Complete | `use_native_unet = TRUE`, 1680 parameters, fixed timestep_embedding |

### Usage with Native Components

```r
# Full native pipeline for SD21 (works on Blackwell)
txt2img_sd21("A cat wearing a hat",
             use_native_decoder = TRUE,
             use_native_text_encoder = TRUE,
             use_native_unet = TRUE)

# Full native pipeline for SDXL (works on Blackwell)
txt2img_sdxl("A sunset over mountains",
             use_native_decoder = TRUE,
             use_native_text_encoder = TRUE,
             use_native_unet = TRUE)
```

### SDXL Native UNet Fix (January 2026)

The native SDXL UNet initially had ~12% mean error due to incorrect `timestep_embedding()`:

**Root cause:** Model-specific parameters in shared utility function:
- SDXL uses `flip_sin_to_cos=TRUE` (cos before sin), SD21 uses FALSE
- SDXL uses `downscale_freq_shift=0` (divide by half_dim), SD21 uses 1

**Resolution:** Added config parameters to `timestep_embedding()` with correct defaults.
Final output error: <0.1% (within float16 tolerance).

### Architecture Auto-Detection

The native text encoder auto-detects model architecture:

| Model | embed_dim | layers | heads | Final LN |
|-------|-----------|--------|-------|----------|
| SD21 (OpenCLIP ViT-H) | 1024 | 23 | 16 | Yes |
| SDXL text_encoder (CLIP ViT-L) | 768 | 12 | 12 | No |
| SDXL text_encoder2 (OpenCLIP ViT-bigG) | 1280 | 32 | 20 | No |

See `TORCHSCRIPT_MIGRATION.md` for detailed migration progress.

## Critical: Inference Memory Management

**ALWAYS use `torch::with_no_grad()` for inference.** Without it, PyTorch builds computation graphs that consume massive amounts of VRAM.

The `txt2img_*` functions now wrap the denoising loop in `with_no_grad()`. If writing custom inference code:

```r
# CORRECT - no gradient tracking
torch::with_no_grad({
  output <- unet(latents, timestep, prompt_embed)
})

# WRONG - will OOM on large models
output <- unet(latents, timestep, prompt_embed)
```

## Known Issues / TODOs

### Blackwell GPU Compatibility

**Status**: Resolved with native modules.

All components now have native R torch implementations that work on Blackwell GPUs (RTX 50xx):
- Use `use_native_unet = TRUE`, `use_native_decoder = TRUE`, `use_native_text_encoder = TRUE`

### Model Files

**Current (TorchScript - being phased out):**

Models stored in `~/.local/share/R/diffuseR/{model_name}/`:
- `unet-{device}-{dtype}.pt` (TorchScript)
- `decoder-{device}.pt` (TorchScript)
- `text_encoder-{device}.pt` (TorchScript)
- `encoder-{device}.pt` (TorchScript)

Downloaded from: `huggingface.co/datasets/cornball-ai/sdxl-R`

**Future (safetensors):**

Native torch modules will load weights directly from HuggingFace safetensors format:
- No device-specific files needed
- Direct loading from HuggingFace model repos
- Smaller downloads (weights only, no traced graphs)

```r
# Future API (TODO)
model <- load_from_hf("stabilityai/stable-diffusion-xl-base-1.0")
```

See cornyverse CLAUDE.md for safetensors package setup (use cornball-ai fork until PR merged).

## Roadmap

### Native Torch Implementation (Complete)
- [x] **VAE Decoder**: Native implementation complete
- [x] **Text Encoder**: Native implementation with auto-detection
- [x] **Text Encoder 2**: OpenCLIP ViT-bigG for SDXL
- [x] **UNet SD21**: 4 blocks, 686 parameters
- [x] **UNet SDXL**: 3 blocks, variable transformer depths (0, 2, 10), 1680 parameters

### gpuctl Integration
- [x] **Auto-device configuration**: `auto_devices()` integrates with gpuctl
  - Queries available VRAM via `gpuctl::gpu_detect()`
  - Auto-selects optimal devices based on model requirements
  - Handles Blackwell workaround automatically

### Model Support
- [ ] Add FLUX model support
- [ ] Add SD3 model support
- [ ] ControlNet integration

### LTX-2 Video Generation (In Progress)
- [x] FlowMatch scheduler (validated against Python)
- [x] RoPE positional embeddings (validated against Python)
- [x] LTX2 Video VAE (3D causal convolutions) - see learnings below
- [x] DiT transformer (audio-video) - see learnings below
- [x] Text encoder integration (connectors + flexible backends)
- [x] GPU-poor optimizations (wan2GP style memory profiles)
- [x] Pipeline integration (txt2vid_ltx2)
- [x] Video output utilities (save_video)
- [x] Weight loading from HuggingFace safetensors

#### LTX-2 Weight Loading

Load LTX-2 model weights from HuggingFace safetensors:

```r
# Load VAE (2.44 GB)
vae <- load_ltx2_vae(
  weights_path = "~/.cache/huggingface/hub/models--Lightricks--LTX-2/vae/diffusion_pytorch_model.safetensors",
  config_path = "~/.cache/huggingface/hub/models--Lightricks--LTX-2/vae/config.json",
  device = "cuda",
  dtype = "float16"
)

# Load transformer (37.8 GB, sharded across 8 files)
transformer <- load_ltx2_transformer(
  weights_dir = "~/.cache/huggingface/hub/models--Lightricks--LTX-2/transformer",
  device = "cpu",  # Start on CPU, offload to GPU layer-by-layer
  dtype = "float16"
)

# Load text connectors (2.86 GB)
connectors <- load_ltx2_connectors(
  weights_path = "~/.cache/huggingface/hub/models--Lightricks--LTX-2/connectors/diffusion_pytorch_model.safetensors",
  config_path = "~/.cache/huggingface/hub/models--Lightricks--LTX-2/connectors/config.json"
)
```

**Model sizes:**
| Component | Size | Notes |
|-----------|------|-------|
| VAE | 2.44 GB | Single safetensors file |
| Transformer | 37.8 GB | Sharded across 8 files |
| Connectors | 2.86 GB | Single safetensors file |
| Total (19B) | 43.3 GB | Full precision |
| Total FP8 | 27.1 GB | Quantized |

#### LTX2 VAE Implementation Learnings

**Temporal dimension constraint for causal downsampling:**
For LTX2's causal 3D convolutions with stride S downsampling, the temporal dimension T must satisfy:
```
(T + S - 1) % S == 0
```
This means T % S == 1, so **T must be odd for stride=2**.

Example valid sequences for 2 spatiotemporal downsampling stages:
- T=5 → (5+1)/2=3 → (3+1)/2=2 ✓
- T=9 → (9+1)/2=5 → (5+1)/2=3 ✓
- T=4 → (4+1)/2=2.5 ✗ (fails unflatten)

**R torch `unflatten` is 1-indexed:**
```r
# Python: x.unflatten(2, (-1, stride))  # dim 2 is 0-indexed
# R: x$unflatten(3, c(-1, stride))      # dim 3 is 1-indexed
```

**LTX2 decoder channel flow:**
In LTX2 up blocks, `in_channels == out_channels` always. The upsampler handles channel reduction via pixel shuffle. Test inputs must match this pattern.

#### LTX2 DiT Transformer Learnings

**cross_attention_dim must equal inner_dim:**
In the LTX2 transformer, `caption_projection` projects text embeddings from `caption_channels` to `inner_dim`. The transformer blocks then expect `encoder_hidden_states` to have dimension `cross_attention_dim`. These must be equal:
```r
# In model config:
inner_dim = num_attention_heads * attention_head_dim  # e.g., 32 * 128 = 4096
cross_attention_dim = 4096  # Must equal inner_dim!
```

**R torch lacks nnf_scaled_dot_product_attention:**
Manual scaled dot-product attention is required:
```r
scale <- 1.0 / sqrt(head_dim)
attn_weights <- torch::torch_matmul(query, key$transpose(-2L, -1L)) * scale
if (!is.null(attention_mask)) attn_weights <- attn_weights + attention_mask
attn_weights <- torch::nnf_softmax(attn_weights, dim = -1L)
hidden_states <- torch::torch_matmul(attn_weights, value)
```

**Avoid function name collisions across files:**
The `apply_interleaved_rotary_emb` function in `rope.R` expects `freqs$cos_freqs`, while a similar function in `dit_ltx2_modules.R` expects `freqs[[1]]`. Name collision caused segfaults - renamed to `apply_interleaved_rotary_emb_list` in dit module.

#### LTX2 Text Encoder Learnings

**Architecture: Gemma3 + Connectors:**
LTX-2 uses Gemma3 as the text encoder, with separate connector transformers for video and audio streams:
```
Gemma3 → pack_text_embeds → text_proj_in → video_connector → video_embeds
                                        → audio_connector → audio_embeds
```

**Attention mask broadcasting:**
When using 2D attention masks [B, S], they must be expanded to [B, 1, 1, S] for broadcasting with attention weights [B, H, S, S]:
```r
if (attention_mask$ndim == 2L) {
  attention_mask <- attention_mask$unsqueeze(2L)$unsqueeze(2L)
}
```

**Flexible text encoding backends:**
The `encode_text_ltx2()` function supports multiple backends:
- `"gemma3"`: Native R torch Gemma3 encoder (no Python dependency)
- `"precomputed"`: Load from file (cached embeddings)
- `"api"`: HTTP request to external service (Gemma container)
- `"random"`: Random embeddings for testing

#### Native Gemma3 Text Encoder

Full Gemma3 12B implementation in R torch (`R/gemma3_text_encoder.R`):

```r
# Load tokenizer and model
tokenizer <- gemma3_tokenizer("/path/to/LTX-2/tokenizer")
model <- load_gemma3_text_encoder("/path/to/LTX-2/text_encoder",
                                   device = "cuda", dtype = "float16")

# Encode prompts
result <- encode_with_gemma3("A robot dancing", model = model, tokenizer = tokenizer)
# Returns: list(prompt_embeds, prompt_attention_mask)

# Or use in pipeline
txt2vid_ltx2("A robot dancing", text_backend = "gemma3",
             model_path = "/path/to/LTX-2/text_encoder")
```

**Architecture:**
- 48 hidden layers, hidden_size = 3840
- 16 attention heads, 8 KV heads (GQA)
- Sliding window attention (1024 tokens) on 5/6 layers
- RoPE with 8x scaling for 128K context

#### Native BPE Tokenizer

Pure R BPE tokenizer (`R/tokenizer_bpe.R`) - no Python/reticulate dependency:

```r
# Load from HuggingFace tokenizer.json format
tok <- bpe_tokenizer("/path/to/tokenizer")

# Encode text
result <- encode_bpe(tok, c("hello world", "testing"),
                      max_length = 128L,
                      padding = "max_length",
                      return_tensors = "pt")
# Returns: list(input_ids, attention_mask) as torch tensors

# Decode back
text <- decode_bpe(tok, result$input_ids)
```

**Features:**
- HuggingFace tokenizer.json format support
- UTF-8 character handling
- Left/right padding, truncation
- Torch tensor output
- SentencePiece-style space markers (▁)

## R torch API Quirks

Important differences between R torch and Python PyTorch:

### `torch_arange` is inclusive
R's `torch_arange` includes the end value, unlike Python's `torch.arange`:
```r
# R: 0, 1, 2, 3, 4 (5 elements)
torch::torch_arange(start = 0, end = 4)

# To match Python behavior (0, 1, 2, 3):
torch::torch_arange(start = 0, end = 3)  # or end = n - 1
```

### `$flatten()` requires named arguments
```r
# Wrong - positional args don't work
x$flatten(2, 4)

# Correct - use named args
x$flatten(start_dim = 2, end_dim = 4)

# Or use the function form
torch::torch_flatten(x, start_dim = 2, end_dim = 4)
```

### `$repeat()` needs backticks
```r
# `repeat` is a reserved word in R
x$`repeat`(c(2, 1, 1, 1))
```

### Tensor slicing is 1-indexed
```r
# R: first element is index 1
x[1, , ]  # First batch element

# Python: first element is index 0
# x[0, :, :]
```

### Method chaining with dots
R torch methods use `$` not `.`:
```r
x$unsqueeze(1)$to(device = "cuda")$contiguous()
```

### Device specification
```r
# R uses character strings
device <- "cuda"  # or "cpu"
torch::torch_tensor(c(1, 2, 3), device = device)

# For torch_device objects
torch::torch_device("cuda")
```

### No `with torch.no_grad()` context manager
```r
# Use torch::with_no_grad() function
torch::with_no_grad({
  # inference code here
})

# Or torch::local_no_grad() but be careful with scope
# (only disables gradients within calling function)
```

### R scalar promotion breaks float16

When mixing R scalars with tensors, R promotes to float64, then the operation promotes the tensor:

```r
# WRONG - promotes to float32
x <- some_float16_tensor
y <- x * (1 + scale)  # scale is float16, but (1 + scale) becomes float64 in R

# CORRECT - use tensor methods
y <- x * scale$add(1)  # Preserves float16
```

Same issue with scalar multiplication:
```r
# WRONG - scalar promotes dtype
noise_pred <- noise_uncond + guidance_scale * (noise_cond - noise_uncond)

# CORRECT - use tensor method
noise_pred <- noise_uncond + (noise_cond - noise_uncond)$mul(guidance_scale)
```

And with FlowMatch/Euler step:
```r
# WRONG - R numeric creates float64
dt <- sigma_next - sigma  # Float64
latents <- latents + dt * noise_pred  # Promotes to float32

# CORRECT - create tensor with explicit dtype
dt <- torch::torch_tensor(sigma_next - sigma, dtype = latent_dtype, device = device)
latents <- latents + dt * noise_pred  # Stays float16
```

### Converting tensors to R arrays

Use `as.array()` to convert torch tensors to R arrays. The `$numpy()` method is a Python-ism that doesn't work in R torch:

```r
# CORRECT - R idiom
result <- with_no_grad({
  x <- some_model(input)
  x$cpu()
})
arr <- as.array(result)  # Works

# WRONG - Python-ism, doesn't exist in R torch
arr <- result$numpy()  # Error: could not find function "fn"
```
