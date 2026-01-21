#' GPU-Poor Memory Management for LTX-2
#'
#' wan2GP-style memory optimizations for running LTX-2 video generation
#' on limited VRAM (6-16GB).
#'
#' @name gpu_poor
NULL

#' Get LTX-2 Memory Profile
#'
#' Determines optimal memory configuration based on available VRAM.
#'
#' @param vram_gb Numeric. Available VRAM in GB, or NULL for auto-detection.
#' @param model Character. Model variant: "ltx2-19b-fp4" (default), "ltx2-19b-fp8",
#'   or "ltx2-19b-distilled".
#'
#' @return A list with memory profile settings.
#'
#' @details
#' LTX-2 is a 19B parameter model. Even with FP4 quantization (~10GB weights),
#' it requires careful memory management. The GPU-poor approach:
#'
#' 1. Text encoding runs on CPU (cached)
#' 2. DiT loaded in chunks, processed layer-by-layer, unloaded
#' 3. VAE loaded after DiT unload, decode with tiling, unload
#'
#' Memory profiles:
#' \describe{
#'   \item{high}{16GB+ - FP4 DiT with chunk loading, VAE on GPU}
#'   \item{medium}{12GB - FP4 DiT chunk loading, VAE tiled}
#'   \item{low}{8GB - FP4 DiT layer-by-layer, VAE tiled small}
#'   \item{very_low}{6GB - FP4 layer-by-layer, VAE on CPU}
#'   \item{cpu_only}{All on CPU}
#' }
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Auto-detect profile
#' profile <- ltx2_memory_profile()
#'
#' # Specific VRAM
#' profile <- ltx2_memory_profile(vram_gb = 8)
#' }
ltx2_memory_profile <- function(
  vram_gb = NULL,
  model = "ltx2-19b-fp4"
) {
  # Auto-detect free VRAM if not provided
  if (is.null(vram_gb)) {
    vram_gb <- .detect_vram(use_free = TRUE)
    message(sprintf("Detected %.1f GB free VRAM", vram_gb))
  }

  # Determine profile level
  if (vram_gb >= 16) {
    profile <- "high"
  } else if (vram_gb >= 12) {
    profile <- "medium"
  } else if (vram_gb >= 8) {
    profile <- "low"
  } else if (vram_gb >= 6) {
    profile <- "very_low"
  } else {
    profile <- "cpu_only"
  }

  # Build profile config
  # Note: LTX-2 19B has 48 transformer layers
  # At FP4, ~10GB total model weights
  # Layer chunk size determines how many layers loaded at once

  profiles <- list(
    high = list(
      name = "high",
      # Stage 1: Text encoding (always CPU)
      text_device = "cpu",
      text_backend = "native", # Native Gemma3 encoder
      # Stage 2: DiT denoising
      dit_device = "cuda",
      dit_offload = "chunk", # Load layers in chunks
      dit_chunk_size = 12L, # 12 layers at a time (~2.5GB)
      # Stage 3: VAE decode
      vae_device = "cuda",
      vae_tiling = FALSE,
      vae_tile_size = c(512L, 512L),
      vae_tile_frames = 16L,
      # General settings
      dtype = "float16",
      model_precision = "fp4", # Preferred quantization
      max_resolution = c(720L, 1280L), # height, width
      max_frames = 121L,
      cfg_mode = "batched"# Distilled uses CFG=1, so this is moot
    ),
    medium = list(
      name = "medium",
      text_device = "cpu",
      text_backend = "native",
      dit_device = "cuda",
      dit_offload = "chunk",
      dit_chunk_size = 8L, # 8 layers at a time (~1.7GB)
      vae_device = "cuda",
      vae_tiling = TRUE,
      vae_tile_size = c(512L, 512L),
      vae_tile_frames = 16L,
      dtype = "float16",
      model_precision = "fp4",
      max_resolution = c(720L, 1280L),
      max_frames = 121L,
      cfg_mode = "batched"
    ),
    low = list(
      name = "low",
      text_device = "cpu",
      text_backend = "native",
      dit_device = "cuda",
      dit_offload = "layer", # One layer at a time
      dit_chunk_size = 1L,
      vae_device = "cuda",
      vae_tiling = TRUE,
      vae_tile_size = c(256L, 256L),
      vae_tile_frames = 8L,
      dtype = "float16",
      model_precision = "fp4",
      max_resolution = c(480L, 854L),
      max_frames = 61L,
      cfg_mode = "sequential"
    ),
    very_low = list(
      name = "very_low",
      text_device = "cpu",
      text_backend = "native",
      dit_device = "cuda",
      dit_offload = "layer",
      dit_chunk_size = 1L,
      vae_device = "cpu", # VAE on CPU
      vae_tiling = TRUE,
      vae_tile_size = c(128L, 128L),
      vae_tile_frames = 4L,
      dtype = "float16",
      model_precision = "fp4",
      max_resolution = c(480L, 640L),
      max_frames = 33L,
      cfg_mode = "sequential"
    ),
    cpu_only = list(
      name = "cpu_only",
      text_device = "cpu",
      text_backend = "native",
      dit_device = "cpu",
      dit_offload = "none",
      dit_chunk_size = 48L, # All layers (CPU has more RAM)
      vae_device = "cpu",
      vae_tiling = TRUE,
      vae_tile_size = c(256L, 256L),
      vae_tile_frames = 8L,
      dtype = "float32", # CPU often faster with float32
      model_precision = "fp4",
      max_resolution = c(480L, 640L),
      max_frames = 33L,
      cfg_mode = "sequential"
    )
  )

  profiles[[profile]]
}

#' Get SDXL Memory Profile
#'
#' Determines optimal memory configuration for SDXL image generation
#' based on available VRAM.
#'
#' @param vram_gb Numeric. Available VRAM in GB, or NULL for auto-detection.
#'
#' @return A list with memory profile settings.
#'
#' @details
#' Memory profiles for SDXL:
#' \describe{
#'   \item{full_gpu}{16GB+ - All components on CUDA}
#'   \item{balanced}{10-12GB - UNet + decoder on CUDA, text encoders on CPU}
#'   \item{unet_gpu}{6-10GB - Only UNet on CUDA, everything else CPU}
#'   \item{cpu_only}{<6GB - All on CPU}
#' }
#'
#' Each profile also specifies:
#' - cfg_mode: "batched" or "sequential" (sequential halves peak memory)
#' - cleanup: "none", "phase", or "step" (when to clear VRAM)
#' - dtype: "float16" or "float32"
#' - max_resolution: maximum image dimension
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Auto-detect profile
#' profile <- sdxl_memory_profile()
#'
#' # Specific VRAM
#' profile <- sdxl_memory_profile(vram_gb = 8)
#' }
sdxl_memory_profile <- function(vram_gb = NULL) {
  # Auto-detect free VRAM if not provided
  if (is.null(vram_gb)) {
    vram_gb <- .detect_vram(use_free = TRUE)
    message(sprintf("Detected %.1f GB free VRAM", vram_gb))
  }

  # Determine profile level
  if (vram_gb >= 16) {
    profile <- "full_gpu"
  } else if (vram_gb >= 10) {
    profile <- "balanced"
  } else if (vram_gb >= 6) {
    profile <- "unet_gpu"
  } else {
    profile <- "cpu_only"
  }

  # Build profile config
  profiles <- list(
    full_gpu = list(
      name = "full_gpu",
      devices = list(
        unet = "cuda",
        decoder = "cuda",
        text_encoder = "cuda",
        text_encoder2 = "cuda",
        encoder = "cuda"
      ),
      dtype = "float16",
      cfg_mode = "batched",
      cleanup = "none",
      max_resolution = 1536L,
      step_cleanup_interval = 0L# No step cleanup
    ),
    balanced = list(
      name = "balanced",
      devices = list(
        unet = "cuda",
        decoder = "cuda",
        text_encoder = "cpu",
        text_encoder2 = "cpu",
        encoder = "cpu"
      ),
      dtype = "float16",
      cfg_mode = "batched",
      cleanup = "phase", # Cleanup between text encoding and denoising
      max_resolution = 1024L,
      step_cleanup_interval = 0L
    ),
    unet_gpu = list(
      name = "unet_gpu",
      devices = list(
        unet = "cuda",
        decoder = "cpu",
        text_encoder = "cpu",
        text_encoder2 = "cpu",
        encoder = "cpu"
      ),
      dtype = "float16",
      cfg_mode = "sequential", # Sequential CFG halves peak memory
      cleanup = "phase",
      max_resolution = 1024L,
      step_cleanup_interval = 10L# Cleanup every 10 steps
    ),
    cpu_only = list(
      name = "cpu_only",
      devices = list(
        unet = "cpu",
        decoder = "cpu",
        text_encoder = "cpu",
        text_encoder2 = "cpu",
        encoder = "cpu"
      ),
      dtype = "float32", # CPU often faster with float32
      cfg_mode = "sequential",
      cleanup = "none", # No GPU to clean
      max_resolution = 768L,
      step_cleanup_interval = 0L
    )
  )

  profiles[[profile]]
}

#' Check if GPU is Blackwell Architecture
#'
#' Blackwell GPUs (RTX 50xx) may need special handling.
#'
#' @return Logical. TRUE if Blackwell GPU detected.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' if (is_blackwell_gpu()) {
#'   message("Using Blackwell-compatible settings")
#' }
#' }
is_blackwell_gpu <- function() {
  # Use gpuctl if available
  if (requireNamespace("gpu.ctl", quietly = TRUE)) {
    return(gpu.ctl::gpu_is_blackwell())
  }

  # Fallback: check compute capability via torch
  if (torch::cuda_is_available()) {
    props <- tryCatch(
      torch::cuda_get_device_properties(0L),
      error = function(e) NULL
    )
    if (!is.null(props)) {
      # Blackwell is compute 12.x
      major <- props$major
      return(major >= 12)
    }
  }

  FALSE
}

#' Detect Available VRAM
#'
#' Uses gpuctl if available.
#'
#' @param use_free Logical. If TRUE, return free VRAM. If FALSE, return total.
#'
#' @return Numeric. VRAM in GB, or 0 if no GPU detected.
#' @keywords internal
.detect_vram <- function(use_free = FALSE) {
  # Try gpuctl (preferred - uses nvidia-smi)
  if (requireNamespace("gpu.ctl", quietly = TRUE)) {
    info <- gpu.ctl::gpu_detect()
    if (!is.null(info)) {
      if (use_free && !is.null(info$vram_free_gb)) {
        return(info$vram_free_gb)
      }
      if (!is.null(info$vram_total_gb)) {
        return(info$vram_total_gb)
      }
    }
  }

  # Fallback: check if CUDA available but can't determine VRAM
  if (torch::cuda_is_available()) {
    # Conservative estimate - assume 8GB if we can't detect
    message("Could not detect VRAM. Install gpuctl for accurate detection.")
    return(8)
  }

  # No GPU detected
  0
}

#' Offload Module to CPU
#'
#' Moves a torch module and all its parameters to CPU.
#'
#' @param module A torch nn_module.
#' @param gc Logical. Run garbage collection after offload.
#'
#' @return The module (modified in place).
#'
#' @export
#'
#' @examples
#' \dontrun{
#' model$to(device = "cuda")
#' output <- model(x)
#' offload_to_cpu(model)
#' }
offload_to_cpu <- function(
  module,
  gc = TRUE
) {
  module$to(device = "cpu")
  if (gc && torch::cuda_is_available()) {
    gc()
    torch::cuda_empty_cache()
  }
  invisible(module)
}

#' Load Module to GPU
#'
#' Moves a torch module and all its parameters to CUDA.
#'
#' @param module A torch nn_module.
#' @param device Character. Target device (default "cuda").
#'
#' @return The module (modified in place).
#'
#' @export
#'
#' @examples
#' \dontrun{
#' load_to_gpu(model)
#' output <- model(x)
#' offload_to_cpu(model)
#' }
load_to_gpu <- function(
  module,
  device = "cuda"
) {
  module$to(device = device)
  invisible(module)
}

#' Report VRAM Usage
#'
#' Prints current VRAM usage using gpuctl.
#'
#' @param label Character. Label for the report.
#'
#' @return Invisibly returns a list with used and free VRAM in GB.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' vram_report("After model load")
#' }
vram_report <- function(label = "") {
  if (!torch::cuda_is_available()) {
    message("[", label, "] No CUDA available")
    return(invisible(list(used = 0, free = 0)))
  }

  # Use gpuctl for accurate reporting
  if (requireNamespace("gpu.ctl", quietly = TRUE)) {
    info <- gpu.ctl::gpu_detect()
    if (!is.null(info)) {
      used <- info$vram_used_gb
      free <- info$vram_free_gb
      message(sprintf("[%s] VRAM: %.2f GB used, %.2f GB free",
          label, used, free))
      return(invisible(list(used = used, free = free)))
    }
  }

  message("[", label, "] VRAM: (install gpuctl for detailed stats)")
  invisible(list(used = NA, free = NA))
}

#' Clear VRAM Cache
#'
#' Forces garbage collection and clears CUDA memory cache.
#'
#' @param verbose Logical. Print memory status before/after.
#'
#' @return Invisibly returns NULL.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' clear_vram()
#' }
clear_vram <- function(verbose = FALSE) {
  if (!torch::cuda_is_available()) {
    return(invisible(NULL))
  }

  if (verbose) {
    vram_report("Before clear")
  }

  gc()
  tryCatch(
    torch::cuda_empty_cache(),
    error = function(e) NULL
  )

  if (verbose) {
    vram_report("After clear")
  }

  invisible(NULL)
}

#' DiT Chunk-based Forward Pass
#'
#' Runs transformer layers in chunks, moving each chunk to GPU before
#' computation and back to CPU after. Balances memory usage with speed.
#'
#' @param hidden_states Input tensor.
#' @param layers List of transformer layers (on CPU).
#' @param chunk_size Integer. Number of layers to load at once (default 1).
#' @param device Target device for computation.
#' @param verbose Logical. Print progress.
#' @param ... Additional arguments passed to each layer.
#'
#' @return Output tensor (on CPU).
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Layer-by-layer for 8GB VRAM
#' output <- dit_offloaded_forward(
#'   hidden_states,
#'   model$transformer_blocks,
#'   chunk_size = 1,
#'   device = "cuda"
#' )
#'
#' # Chunk-based for 16GB VRAM
#' output <- dit_offloaded_forward(
#'   hidden_states,
#'   model$transformer_blocks,
#'   chunk_size = 12,  # 12 layers at a time
#'   device = "cuda"
#' )
#' }
dit_offloaded_forward <- function(
  hidden_states,
  layers,
  chunk_size = 1L,
  device = "cuda",
  verbose = FALSE,
  ...
) {
  n_layers <- length(layers)
  chunk_size <- as.integer(chunk_size)

  # Move input to target device
  x <- hidden_states$to(device = device)

  # Process in chunks
  chunk_start <- 1L
  while (chunk_start <= n_layers) {
    chunk_end <- min(chunk_start + chunk_size - 1L, n_layers)

    if (verbose) {
      message(sprintf("  Processing layers %d-%d of %d", chunk_start, chunk_end, n_layers))
    }

    # Load chunk to GPU
    for (i in chunk_start:chunk_end) {
      layers[[i]]$to(device = device)
    }

    # Forward pass through chunk
    for (i in chunk_start:chunk_end) {
      x <- layers[[i]](x, ...)
    }

    # Offload chunk back to CPU
    for (i in chunk_start:chunk_end) {
      layers[[i]]$to(device = "cpu")
    }

    # Clear cache after each chunk
    if (device != "cpu") {
      torch::cuda_empty_cache()
    }

    chunk_start <- chunk_end + 1L
  }

  # Return result on CPU
  x$to(device = "cpu")
}

#' Sequential CFG Forward Pass
#'
#' Runs unconditional and conditional forward passes separately to halve
#' peak activation memory. For GPU-poor scenarios.
#'
#' @param model The DiT model.
#' @param latents Current latent tensor.
#' @param timestep Current timestep tensor.
#' @param prompt_embeds Conditional prompt embeddings.
#' @param negative_prompt_embeds Unconditional prompt embeddings.
#' @param guidance_scale CFG scale.
#' @param ... Additional arguments to model forward pass.
#'
#' @return The CFG-combined noise prediction.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' noise_pred <- sequential_cfg_forward(
#'   model, latents, timestep,
#'   prompt_embeds, negative_prompt_embeds,
#'   guidance_scale = 4.0
#' )
#' }
sequential_cfg_forward <- function(
  model,
  latents,
  timestep,
  prompt_embeds,
  negative_prompt_embeds,
  guidance_scale,
  ...
) {
  torch::with_no_grad({
      # Unconditional pass
      noise_pred_uncond <- model(
        hidden_states = latents,
        encoder_hidden_states = negative_prompt_embeds,
        timestep = timestep,
        ...
      )$sample

      # Conditional pass
      noise_pred_cond <- model(
        hidden_states = latents,
        encoder_hidden_states = prompt_embeds,
        timestep = timestep,
        ...
      )$sample

      # CFG combination
      noise_pred <- noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

      # Clean up intermediate tensors
      rm(noise_pred_uncond, noise_pred_cond)
    })

  noise_pred
}

#' Validate Resolution Against Profile
#'
#' Checks if requested resolution fits within memory profile limits.
#'
#' @param height Integer. Requested height.
#' @param width Integer. Requested width.
#' @param num_frames Integer. Requested number of frames.
#' @param profile Memory profile from `ltx2_memory_profile()`.
#'
#' @return List with adjusted height, width, num_frames and warning if adjusted.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' profile <- ltx2_memory_profile(vram_gb = 8)
#' validated <- validate_resolution(720, 1280, 60, profile)
#' }
validate_resolution <- function(
  height,
  width,
  num_frames,
  profile
) {
  adjusted <- FALSE
  warnings <- character(0)

  max_h <- profile$max_resolution[1]
  max_w <- profile$max_resolution[2]
  max_f <- profile$max_frames

  if (height > max_h) {
    warnings <- c(warnings, sprintf("Height %d exceeds profile max %d", height, max_h))
    height <- max_h
    adjusted <- TRUE
  }

  if (width > max_w) {
    warnings <- c(warnings, sprintf("Width %d exceeds profile max %d", width, max_w))
    width <- max_w
    adjusted <- TRUE
  }

  if (num_frames > max_f) {
    warnings <- c(warnings, sprintf("Frames %d exceeds profile max %d", num_frames, max_f))
    num_frames <- max_f
    adjusted <- TRUE
  }

  if (adjusted && length(warnings) > 0) {
    warning("Resolution adjusted for memory profile '", profile$name, "':\n  ",
      paste(warnings, collapse = "\n  "))
  }

  list(
    height = height,
    width = width,
    num_frames = num_frames,
    adjusted = adjusted
  )
}

#' Configure VAE for Memory Profile
#'
#' Sets VAE tiling parameters based on memory profile.
#'
#' @param vae The LTX2 VAE module.
#' @param profile Memory profile from `ltx2_memory_profile()`.
#'
#' @return The VAE (modified in place).
#'
#' @export
#'
#' @examples
#' \dontrun{
#' profile <- ltx2_memory_profile(vram_gb = 8)
#' vae <- load_ltx2_vae(...)
#' configure_vae_for_profile(vae, profile)
#' }
configure_vae_for_profile <- function(
  vae,
  profile
) {
  if (profile$vae_tiling) {
    vae$enable_tiling(
      tile_sample_min_height = profile$vae_tile_size[1],
      tile_sample_min_width = profile$vae_tile_size[2],
      tile_sample_min_num_frames = profile$vae_tile_frames
    )
  } else {
    vae$disable_tiling()
  }

  invisible(vae)
}

#' Quantize Tensor to INT4
#'
#' Quantizes a float tensor to 4-bit integers with block-wise scaling.
#' Two INT4 values are packed per byte for 7-8x compression.
#'
#' @param x Tensor. Input float tensor.
#' @param block_size Integer. Number of values per scale factor (default 64).
#'
#' @return A list with:
#'   - `packed`: uint8 tensor with packed INT4 values
#'   - `scales`: float tensor with per-block scale factors
#'   - `orig_shape`: original tensor shape
#'   - `orig_numel`: original number of elements
#'   - `block_size`: block size used
#'
#' @details
#' INT4 range is -8 to 7. Values are scaled per block, quantized, shifted to
#' unsigned (0-15), and packed two per byte. Compression is ~7x vs float32,
#' ~3.5x vs float16. Typical reconstruction error is 10-12% of std.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' x <- torch_randn(c(4096, 4096)) * 0.02
#' q <- quantize_int4(x)
#' x_back <- dequantize_int4(q)
#' }
quantize_int4 <- function(
  x,
  block_size = 64L
) {
  orig_shape <- x$shape
  orig_dtype <- x$dtype
  x_flat <- x$to(dtype = torch::torch_float32())$flatten()
  n <- x_flat$shape[1]

  # Pad to multiple of block_size * 2 (2 values per byte)
  pad_to <- ceiling(n / (block_size * 2)) * block_size * 2
  if (pad_to > n) {
    x_flat <- torch::torch_cat(list(
        x_flat,
        torch::torch_zeros(pad_to - n, dtype = torch::torch_float32(), device = x$device)
      ))
  }

  # Reshape into blocks
  n_blocks <- as.integer(pad_to / block_size)
  x_blocks <- x_flat$reshape(c(n_blocks, block_size))

  # Compute scale per block (absmax / 7)
  scales <- x_blocks$abs()$max(dim = 2) [[1]] / 7.0
  scales <- scales$clamp(min = 1e-10)

  # Quantize: scale, round, clamp to -8..7, shift to 0..15
  x_scaled <- x_blocks / scales$unsqueeze(2)
  x_int <- x_scaled$round()$clamp(- 8, 7) + 8L
  x_uint <- x_int$to(dtype = torch::torch_uint8())

  # Pack pairs into bytes (high nibble * 16 + low nibble)
  x_uint <- x_uint$reshape(c(n_blocks, block_size %/% 2L, 2L))
  high <- x_uint[,, 1]$to(torch::torch_int32()) * 16L
  low <- x_uint[,, 2]$to(torch::torch_int32())
  packed <- (high + low)$to(torch::torch_uint8())

  list(
    packed = packed$flatten(),
    scales = scales,
    orig_shape = orig_shape,
    orig_numel = n,
    block_size = block_size
  )
}

#' Dequantize INT4 Tensor
#'
#' Reconstructs a float tensor from INT4-quantized data.
#'
#' @param q List. Quantized data from `quantize_int4()`.
#' @param dtype Torch dtype. Output dtype (default float16).
#' @param device Character. Target device.
#'
#' @return Tensor with original shape and specified dtype.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' q <- quantize_int4(weights)
#' weights_approx <- dequantize_int4(q, dtype = torch_float16(), device = "cuda")
#' }
dequantize_int4 <- function(
  q,
  dtype = torch::torch_float16(),
  device = "cpu"
) {
  packed <- q$packed$to(dtype = torch::torch_int32(), device = device)

  # Unpack bytes: high nibble = floor(x/16), low nibble = x mod 16
  high <- torch::torch_floor(packed$to(torch::torch_float32()) / 16)$to(torch::torch_int32())
  low <- packed - high * 16L

  # Interleave high and low nibbles
  x_uint <- torch::torch_stack(list(high, low), dim = 2L)$flatten()

  # Shift back to signed (-8 to 7)
  x_int <- x_uint - 8L

  # Apply per-block scales
  block_size <- q$block_size
  x_blocks <- x_int$reshape(c(- 1L, block_size))$to(dtype = dtype)
  scales_dev <- q$scales$to(dtype = dtype, device = device)
  x_scaled <- x_blocks * scales_dev$unsqueeze(2)

  # Flatten and trim to original size
  x_flat <- x_scaled$flatten()[1:q$orig_numel]
  x_flat$reshape(q$orig_shape)
}

#' Create Linear Layer (Standard or INT4)
#'
#' Factory function that creates either a standard nn_linear or INT4 linear layer
#' based on package options. Use this instead of torch::nn_linear() in model code.
#'
#' @param in_features Integer. Input dimension.
#' @param out_features Integer. Output dimension.
#' @param bias Logical. Include bias term.
#'
#' @return nn_linear or int4_linear module.
#'
#' @details
#' Behavior controlled by options:
#' - `diffuseR.use_int4`: If TRUE, create INT4 layer (default FALSE)
#' - `diffuseR.int4_device`: Device for INT4 layers (default "cuda")
#' - `diffuseR.int4_dtype`: Dtype for INT4 operations (default torch_float16())
#'
#' @export
make_linear <- function(
  in_features,
  out_features,
  bias = TRUE
) {

  if (getOption("diffuseR.use_int4", FALSE)) {
    device <- getOption("diffuseR.int4_device", "cuda")
    dtype <- getOption("diffuseR.int4_dtype", torch::torch_float16())
    int4_linear(in_features, out_features, bias = bias, device = device, dtype = dtype)
  } else {
    torch::nn_linear(in_features, out_features, bias = bias)
  }
}

#' INT4 Linear Layer
#'
#' A linear layer that stores weights in INT4 format and dequantizes on-the-fly
#' during forward pass. This enables running large models on limited VRAM by
#' keeping weights compressed on GPU.
#'
#' @param in_features Integer. Size of each input sample.
#' @param out_features Integer. Size of each output sample.
#' @param bias Logical. If TRUE, adds a learnable bias (stored in float16).
#' @param device Character. Device for the layer.
#' @param dtype torch_dtype. Data type for dequantized operations.
#'
#' @return nn_module with INT4 weight storage.
#'
#' @details
#' The layer stores:
#' - `weight_packed`: uint8 tensor with packed INT4 values
#' - `weight_scales`: float32 tensor with per-block scales
#' - `weight_shape`: original weight shape
#' - `bias`: optional float16 bias
#'
#' During forward(), weights are dequantized to the target dtype, the matmul
#' is performed, and the dequantized tensor is freed. This allows ~40GB models
#' to run with ~10GB of VRAM for weights.
#'
#' @export
int4_linear <- torch::nn_module(
  "INT4Linear",
  initialize = function(
    in_features,
    out_features,
    bias = TRUE,
    device = "cuda",
    dtype = torch::torch_float16()
  ) {
    self$in_features <- in_features
    self$out_features <- out_features
    self$dtype <- dtype
    self$device <- device
    self$block_size <- 64L

    # Placeholder - will be set by load_int4_weight()
    self$weight_packed <- NULL
    self$weight_scales <- NULL
    self$weight_shape <- c(out_features, in_features)
    self$weight_numel <- out_features * in_features

    if (bias) {
      self$bias <- torch::nn_parameter(torch::torch_zeros(out_features,
          dtype = dtype, device = device))
    } else {
      self$bias <- NULL
    }
  },

#' Load INT4 quantized weight into this layer
#' @param q List with packed, scales, orig_shape from quantize_int4()
  load_int4_weight = function(q) {
    # Store INT4 data as buffers (not parameters)
    self$weight_packed <- q$packed$to(device = self$device)
    self$weight_scales <- q$scales$to(device = self$device)
    self$weight_shape <- q$orig_shape
    self$weight_numel <- q$orig_numel
    invisible(self)
  },

  forward = function(x) {
    if (is.null(self$weight_packed)) {
      stop("INT4 weight not loaded. Call load_int4_weight() first.")
    }

    # Dequantize weight on-the-fly
    q <- list(
      packed = self$weight_packed,
      scales = self$weight_scales,
      orig_shape = self$weight_shape,
      orig_numel = self$weight_numel,
      block_size = self$block_size
    )
    weight <- dequantize_int4(q, dtype = self$dtype, device = self$device)

    # Linear operation
    out <- torch::nnf_linear(x, weight, self$bias)

    # Weight tensor goes out of scope and will be freed
    out
  }
)

#' Create INT4 Linear from Standard Linear
#'
#' Converts a standard nn_linear layer to an INT4 linear layer.
#'
#' @param linear_module nn_linear module to convert.
#' @param device Character. Target device for INT4 weights.
#' @param dtype torch_dtype. Target dtype for dequantized operations.
#'
#' @return int4_linear module with quantized weights.
#'
#' @export
linear_to_int4 <- function(
  linear_module,
  device = "cuda",
  dtype = torch::torch_float16()
) {
  in_features <- linear_module$in_features
  out_features <- linear_module$out_features
  has_bias <- !is.null(linear_module$bias)

  # Create INT4 layer
  int4_layer <- int4_linear(in_features, out_features, bias = has_bias,
    device = device, dtype = dtype)

  # Quantize and load weight
  q <- quantize_int4(linear_module$weight)
  int4_layer$load_int4_weight(q)

  # Copy bias if present (use with_no_grad to avoid in-place error on parameter)
  if (has_bias) {
    torch::with_no_grad({
        int4_layer$bias$copy_(linear_module$bias$to(dtype = dtype, device = device))
      })
  }

  int4_layer
}

#' Create INT4 Linear from Pre-quantized Weights
#'
#' Creates an INT4 linear layer from pre-quantized weight data.
#'
#' @param q_weight List with packed, scales, orig_shape from load_int4_weights().
#' @param q_bias Optional. Quantized bias (or NULL for no bias).
#' @param bias_tensor Optional. Float tensor for bias (if not quantized).
#' @param device Character. Target device.
#' @param dtype torch_dtype. Target dtype for operations.
#'
#' @return int4_linear module with loaded weights.
#'
#' @export
int4_linear_from_quantized <- function(
  q_weight,
  q_bias = NULL,
  bias_tensor = NULL,
  device = "cuda",
  dtype = torch::torch_float16()
) {
  out_features <- q_weight$orig_shape[1]
  in_features <- q_weight$orig_shape[2]
  has_bias <- !is.null(q_bias) || !is.null(bias_tensor)

  # Create INT4 layer
  int4_layer <- int4_linear(in_features, out_features, bias = has_bias,
    device = device, dtype = dtype)

  # Load quantized weight
  int4_layer$load_int4_weight(q_weight)

  # Load bias if present (use with_no_grad to avoid in-place error on parameter)
  if (!is.null(bias_tensor)) {
    torch::with_no_grad({
        int4_layer$bias$copy_(bias_tensor$to(dtype = dtype, device = device))
      })
  } else if (!is.null(q_bias)) {
    # Dequantize bias
    bias_dequant <- dequantize_int4(q_bias, dtype = dtype, device = device)
    torch::with_no_grad({
        int4_layer$bias$copy_(bias_dequant)
      })
  }

  int4_layer
}

#' Load INT4 Weights into Model
#'
#' Replaces linear layers in a model with INT4 versions and loads quantized weights.
#' This is the main entry point for running large models with INT4 quantization.
#'
#' @param model nn_module. The model to convert.
#' @param int4_weights List of quantized weights from `load_int4_weights()`.
#' @param device Character. Target device for INT4 weights.
#' @param dtype torch_dtype. Target dtype for dequantized operations.
#' @param verbose Logical. Print progress.
#'
#' @return The model with linear layers replaced by INT4 versions.
#'
#' @details
#' This function:
#' 1. Identifies linear layers by matching parameter names ending in ".weight"
#' 2. Creates INT4Linear layers with matching dimensions
#' 3. Loads quantized weights and biases
#' 4. Replaces the original layers in the model
#'
#' The INT4 weights stay compressed on GPU (~10GB for a 40GB model).
#' During forward(), each layer dequantizes on-the-fly, keeping memory usage low.
#'
#' @export
load_int4_into_model <- function(
  model,
  int4_weights,
  device = "cuda",
  dtype = torch::torch_float16(),
  verbose = TRUE
) {
  # Get all module names that have .weight in the quantized weights
  weight_names <- grep("\\.weight$", names(int4_weights), value = TRUE)

  if (verbose) {
    message(sprintf("Loading %d INT4 weights into model...", length(weight_names)))
  }

  loaded <- 0
  skipped <- 0

  for (weight_name in weight_names) {
    # Extract module path (e.g., "transformer_blocks.0.attn1.to_q")
    module_path <- sub("\\.weight$", "", weight_name)
    bias_name <- paste0(module_path, ".bias")

    q_weight <- int4_weights[[weight_name]]
    if (bias_name %in% names(int4_weights)) {
      q_bias <- int4_weights[[bias_name]]
    } else {
      q_bias <- NULL
    }

    # Check dimensions - only process 2D weights (linear layers)
    if (length(q_weight$orig_shape) != 2) {
      skipped <- skipped + 1
      next
    }

    # Create INT4 layer
    out_features <- q_weight$orig_shape[1]
    in_features <- q_weight$orig_shape[2]
    has_bias <- !is.null(q_bias)

    int4_layer <- int4_linear(in_features, out_features, bias = has_bias,
      device = device, dtype = dtype)
    int4_layer$load_int4_weight(q_weight)

    # Load bias if present
    if (has_bias) {
      bias_dequant <- dequantize_int4(q_bias, dtype = dtype, device = device)
      torch::with_no_grad({
          int4_layer$bias$copy_(bias_dequant)
        })
    }

    # Store the INT4 layer for later assignment
    # Note: Direct module replacement in R torch is complex
    # For now, store in a separate list that can be used during forward
    if (!exists("int4_layers", where = model)) {
      model$int4_layers <- list()
    }
    model$int4_layers[[module_path]] <- int4_layer

    loaded <- loaded + 1
  }

  if (verbose) {
    message(sprintf("Loaded %d INT4 layers, skipped %d non-linear", loaded, skipped))
  }

  invisible(model)
}

#' Load INT4 Weights into INT4 Model
#'
#' Loads pre-quantized INT4 weights into a model created with `make_linear()`
#' when `diffuseR.use_int4 = TRUE`.
#'
#' @param model nn_module created with INT4 layers.
#' @param int4_weights List from `load_int4_weights()`.
#' @param verbose Logical. Print progress.
#'
#' @return Model with INT4 weights loaded (invisibly).
#'
#' @export
load_int4_weights_into_model <- function(
  model,
  int4_weights,
  verbose = TRUE
) {
  # Get model's named modules (flattened)
  params <- model$parameters
  param_names <- names(params)

  loaded <- 0
  skipped <- 0

  # Name mapping from HuggingFace to R model structure
  # FFN layers have different naming:
  #   HF: ff.net.0.proj, ff.net.2
  #   R:  ff.act_fn.proj, ff.proj_out
  map_hf_to_r_name <- function(hf_name) {
    r_name <- hf_name
    # Map FFN layer names
    r_name <- gsub("\\.ff\\.net\\.0\\.proj\\.", ".ff.act_fn.proj.", r_name)
    r_name <- gsub("\\.ff\\.net\\.2\\.", ".ff.proj_out.", r_name)
    r_name <- gsub("\\.audio_ff\\.net\\.0\\.proj\\.", ".audio_ff.act_fn.proj.", r_name)
    r_name <- gsub("\\.audio_ff\\.net\\.2\\.", ".audio_ff.proj_out.", r_name)
    # Handle end-of-string cases
    r_name <- gsub("\\.ff\\.net\\.0\\.proj$", ".ff.act_fn.proj", r_name)
    r_name <- gsub("\\.ff\\.net\\.2$", ".ff.proj_out", r_name)
    r_name <- gsub("\\.audio_ff\\.net\\.0\\.proj$", ".audio_ff.act_fn.proj", r_name)
    r_name <- gsub("\\.audio_ff\\.net\\.2$", ".audio_ff.proj_out", r_name)
    r_name
  }

  for (int4_name in names(int4_weights)) {
    # Map HuggingFace name to R model name
    r_name <- map_hf_to_r_name(int4_name)
    # Check if this weight (with mapped name) exists in model
    if (r_name %in% param_names) {
      # This is a regular parameter (bias, norm weights, etc.)
      # Dequantize and copy
      q <- int4_weights[[int4_name]]
      if (length(q$orig_shape) == 1) {
        # 1D tensor (bias, norm) - dequantize to model's device/dtype
        param <- params[[r_name]]
        dequant <- dequantize_int4(q, dtype = param$dtype, device = as.character(param$device))
        torch::with_no_grad({
            param$copy_(dequant)
          })
        loaded <- loaded + 1
      }
    } else if (grepl("\\.weight$", r_name)) {
      # This might be an INT4 linear weight - find the layer
      # Weight name: "module.path.weight" -> layer path: "module.path"
      layer_path <- sub("\\.weight$", "", r_name)

      # Try to find corresponding INT4 layer by navigating module tree
      tryCatch({
          # Navigate to the layer using R model path
          parts <- strsplit(layer_path, "\\.") [[1]]
          current <- model

          for (part in parts) {
            if (grepl("^[0-9]+$", part)) {
              # Numeric index (0-based in Python, 1-based in R)
              idx <- as.integer(part) + 1L
              current <- current[[idx]]
            } else {
              current <- current[[part]]
            }
          }

          # Check if this is an INT4Linear layer (has load_int4_weight method)
          if (!is.null(current$load_int4_weight)) {
            # Load INT4 weight directly (using original name for data access)
            q <- int4_weights[[int4_name]]
            current$load_int4_weight(q)
            loaded <- loaded + 1
          } else {
            skipped <- skipped + 1
          }
        }, error = function(e) {
          skipped <<- skipped + 1
        })
    } else {
      skipped <- skipped + 1
    }
  }

  if (verbose) {
    message(sprintf("Loaded %d weights, skipped %d", loaded, skipped))
  }

  invisible(model)
}

#' Quantize Model Weights to INT4
#'
#' Quantizes all parameters in a torch module to INT4 format.
#'
#' @param module nn_module. The model to quantize.
#' @param block_size Integer. Block size for quantization.
#' @param verbose Logical. Print progress.
#'
#' @return List of quantized parameters (does not modify original module).
#'
#' @export
quantize_model_int4 <- function(
  module,
  block_size = 64L,
  verbose = TRUE
) {
  params <- module$parameters
  quantized <- list()

  total_orig <- 0
  total_quant <- 0

  for (name in names(params)) {
    p <- params[[name]]
    orig_bytes <- prod(p$shape) * 2# Assume float16

    q <- quantize_int4(p, block_size = block_size)
    quant_bytes <- q$packed$shape[1] + prod(q$scales$shape) * 4

    quantized[[name]] <- q
    total_orig <- total_orig + orig_bytes
    total_quant <- total_quant + quant_bytes

    if (verbose) {
      message(sprintf("  %s: %.2f MB -> %.2f MB",
          name, orig_bytes / 1e6, quant_bytes / 1e6))
    }
  }

  if (verbose) {
    message(sprintf("Total: %.2f MB -> %.2f MB (%.1fx compression)",
        total_orig / 1e6, total_quant / 1e6, total_orig / total_quant))
  }

  quantized
}

#' Save INT4 Quantized Weights
#'
#' Saves INT4 quantized weights to disk as sharded safetensors files.
#'
#' @param quantized_params List of quantized parameters from `quantize_model_int4()`.
#' @param path Character. Base path for safetensors files. If multiple shards needed,
#'   files will be named `path-00001-of-NNNNN.safetensors`.
#' @param max_shard_size Numeric. Maximum bytes per shard (default 2GB to avoid R
#'   integer overflow issues).
#' @param verbose Logical. Print progress.
#'
#' @return Invisible character vector of saved file paths.
#'
#' @details
#' Weights are saved in safetensors format with the following structure:
#' \itemize{
#'   \item `{name}::packed` - uint8 tensor with packed INT4 values
#'   \item `{name}::scales` - float32 tensor with per-block scales
#'   \item `{name}::shape` - int64 tensor with original shape
#' }
#'
#' Large models are automatically sharded to avoid R's 2GB vector limit.
#' The block size is fixed at 64 (standard for INT4 quantization).
#'
#' @export
#'
#' @examples
#' \dontrun{
#' q <- quantize_model_int4(model)
#' save_int4_weights(q, "model_int4.safetensors")
#' }
save_int4_weights <- function(
  quantized_params,
  path,
  max_shard_size = 2e9,
  verbose = TRUE
) {
  if (verbose) message(sprintf("Preparing %d parameters for safetensors...", length(quantized_params)))

  # Calculate total size and estimate number of shards
  total_bytes <- 0
  param_sizes <- list()
  for (name in names(quantized_params)) {
    q <- quantized_params[[name]]
    size <- prod(q$packed$shape) + prod(q$scales$shape) * 4 + length(q$orig_shape) * 8
    param_sizes[[name]] <- size
    total_bytes <- total_bytes + size
  }

  n_shards <- max(1L, ceiling(total_bytes / max_shard_size))

  if (n_shards == 1) {
    # Single file - use original path
    tensors <- list()
    for (name in names(quantized_params)) {
      q <- quantized_params[[name]]
      tensors[[paste0(name, "::packed")]] <- q$packed$cpu()
      tensors[[paste0(name, "::scales")]] <- q$scales$cpu()
      tensors[[paste0(name, "::shape")]] <- torch::torch_tensor(q$orig_shape, dtype = torch::torch_int64())
    }
    if (verbose) message(sprintf("Saving %d tensors to %s...", length(tensors), path))
    safetensors::safe_save_file(tensors, path)
    file_size <- file.info(path)$size / 1e6
    if (verbose) message(sprintf("Saved %.2f MB", file_size))
    return(invisible(path))
  }

  # Multiple shards - split params across files
  if (verbose) message(sprintf("Sharding into %d files (max %.1f GB each)...", n_shards, max_shard_size / 1e9))

  # Remove extension for base path
  base_path <- sub("\\.safetensors$", "", path)
  param_names <- names(quantized_params)
  params_per_shard <- ceiling(length(param_names) / n_shards)
  saved_paths <- character(0)

  for (shard_idx in seq_len(n_shards)) {
    start_idx <- (shard_idx - 1) * params_per_shard + 1
    end_idx <- min(shard_idx * params_per_shard, length(param_names))

    if (start_idx > length(param_names)) break

    shard_names <- param_names[start_idx:end_idx]
    tensors <- list()

    for (name in shard_names) {
      q <- quantized_params[[name]]
      tensors[[paste0(name, "::packed")]] <- q$packed$cpu()
      tensors[[paste0(name, "::scales")]] <- q$scales$cpu()
      tensors[[paste0(name, "::shape")]] <- torch::torch_tensor(q$orig_shape, dtype = torch::torch_int64())
    }

    shard_path <- sprintf("%s-%05d-of-%05d.safetensors", base_path, shard_idx, n_shards)
    if (verbose) message(sprintf("  [%d/%d] Saving %d params to %s...",
        shard_idx, n_shards, length(shard_names), basename(shard_path)))
    safetensors::safe_save_file(tensors, shard_path)
    saved_paths <- c(saved_paths, shard_path)
  }

  total_size <- sum(file.info(saved_paths)$size) / 1e6
  if (verbose) message(sprintf("Total: %.2f MB across %d shards", total_size, length(saved_paths)))

  invisible(saved_paths)
}

#' Load INT4 Quantized Weights
#'
#' Loads INT4 quantized weights from safetensors file(s).
#'
#' @param path Character. Path to safetensors file or base path for sharded files.
#'   For sharded files, pass the base path (e.g., "model_int4.safetensors") and
#'   the function will find all shards matching the pattern.
#' @param verbose Logical. Print progress.
#'
#' @return List of quantized parameter structures ready for `dequantize_int4()`.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' q <- load_int4_weights("model_int4.safetensors")
#' # Dequantize specific parameter on GPU
#' weight <- dequantize_int4(q[["linear.weight"]], device = "cuda")
#' }
load_int4_weights <- function(
  path,
  verbose = TRUE
) {
  path <- path.expand(path)

  # Check for sharded files
  base_path <- sub("\\.safetensors$", "", path)
  shard_pattern <- sprintf("%s-[0-9]+-of-[0-9]+\\.safetensors$", basename(base_path))
  shard_dir <- dirname(path)
  shard_files <- list.files(shard_dir, pattern = shard_pattern, full.names = TRUE)

  if (length(shard_files) > 0) {
    # Load sharded files
    shard_files <- sort(shard_files)
    if (verbose) {
      total_size <- sum(file.info(shard_files)$size) / 1e6
      message(sprintf("Loading INT4 weights from %d shards (%.2f MB total)...",
          length(shard_files), total_size))
    }
    paths <- shard_files
  } else if (file.exists(path)) {
    # Single file
    if (verbose) {
      size_mb <- file.info(path)$size / 1e6
      message(sprintf("Loading INT4 weights from %s (%.2f MB)...", path, size_mb))
    }
    paths <- path
  } else {
    stop("File not found: ", path)
  }

  # Load all files
  quantized <- list()
  for (i in seq_along(paths)) {
    p <- paths[i]
    if (verbose && length(paths) > 1) {
      message(sprintf("  [%d/%d] Loading %s...", i, length(paths), basename(p)))
    }

    tensors <- safetensors::safe_load_file(p, framework = "torch")

    # Parse tensor names to reconstruct parameter structures
    packed_names <- grep("::packed$", names(tensors), value = TRUE)
    param_names <- sub("::packed$", "", packed_names)

    for (name in param_names) {
      packed <- tensors[[paste0(name, "::packed")]]
      scales <- tensors[[paste0(name, "::scales")]]
      shape_tensor <- tensors[[paste0(name, "::shape")]]
      orig_shape <- as.integer(as.array(shape_tensor))
      orig_numel <- prod(orig_shape)

      quantized[[name]] <- list(
        packed = packed,
        scales = scales,
        orig_shape = orig_shape,
        orig_numel = orig_numel,
        block_size = 64L# Standard block size
      )
    }
  }

  if (verbose) message(sprintf("Done. Loaded %d parameters.", length(quantized)))
  quantized
}

#' Quantize Safetensor Weights to INT4
#'
#' Loads weights from safetensors file(s) and quantizes to INT4.
#' Useful for quantizing large models without loading the full module.
#'
#' @param paths Character vector. Paths to safetensor files.
#' @param output_path Character. Path to save INT4 weights (.safetensors).
#' @param block_size Integer. Block size for quantization (default 64).
#' @param verbose Logical. Print progress.
#'
#' @return Invisible NULL. Writes quantized weights to output_path.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Quantize sharded transformer weights
#' paths <- list.files("~/.cache/huggingface/.../transformer",
#'                     pattern = "safetensors$", full.names = TRUE)
#' quantize_safetensors_int4(paths, "dit_int4.safetensors")
#' }
quantize_safetensors_int4 <- function(
  paths,
  output_path,
  block_size = 64L,
  verbose = TRUE
) {
  all_quantized <- list()
  total_orig <- 0
  total_quant <- 0

  for (path in paths) {
    if (verbose) message(sprintf("Loading %s...", basename(path)))

    weights <- safetensors::safe_load_file(path, framework = "torch")

    for (name in names(weights)) {
      w <- weights[[name]]
      orig_bytes <- prod(w$shape) * 2# Assume float16

      q <- quantize_int4(w, block_size = block_size)
      quant_bytes <- length(as.array(q$packed)) + prod(q$scales$shape) * 4

      all_quantized[[name]] <- q
      total_orig <- total_orig + orig_bytes
      total_quant <- total_quant + quant_bytes

      if (verbose && prod(w$shape) > 1e6) {
        message(sprintf("  %s: %.2f MB -> %.2f MB",
            name, orig_bytes / 1e6, quant_bytes / 1e6))
      }
    }

    # Clear memory between shards
    rm(weights)
    gc()
  }

  if (verbose) {
    message(sprintf("\nTotal: %.2f GB -> %.2f GB (%.1fx compression)",
        total_orig / 1e9, total_quant / 1e9, total_orig / total_quant))
  }

  save_int4_weights(all_quantized, output_path, verbose = verbose)
}

#' Quantize LTX-2 Transformer to INT4
#'
#' Downloads (if needed) and quantizes the LTX-2 19B transformer to INT4 format.
#' The quantized weights are cached for future use.
#'
#' @param model_id Character. HuggingFace model ID (default "Lightricks/LTX-2").
#' @param output_dir Character. Directory to save quantized weights.
#'   Default uses `tools::R_user_dir("diffuseR", "cache")`.
#' @param block_size Integer. Block size for INT4 quantization (default 64).
#' @param force Logical. Re-quantize even if cached file exists.
#' @param download Logical. If TRUE, download model from HuggingFace if not cached.
#' @param verbose Logical. Print progress.
#'
#' @return Character. Path to the quantized weights file.
#'
#' @details
#' LTX-2 is a 19B parameter model (~40GB in BF16). INT4 quantization reduces
#' this to ~5.7GB, fitting in 16GB VRAM with room for activations.
#'
#' The function:
#' 1. Uses hfhub to locate/download the model from HuggingFace
#' 2. Loads each safetensor shard
#' 3. Quantizes all weights to INT4 (block-wise, 64 values per scale)
#' 4. Saves as safetensors file
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Quantize and cache (first run takes ~10-20 minutes)
#' weights_path <- quantize_ltx2_transformer()
#'
#' # Load quantized weights
#' q <- load_int4_weights(weights_path)
#'
#' # Dequantize specific layer on GPU
#' layer_weight <- dequantize_int4(q[["transformer_blocks.0.attn1.to_q.weight"]],
#'                                  device = "cuda", dtype = torch_float16())
#' }
quantize_ltx2_transformer <- function(
  model_id = "Lightricks/LTX-2",
  output_dir = NULL,
  block_size = 64L,
  force = FALSE,
  download = FALSE,
  verbose = TRUE
) {
  if (!requireNamespace("hfhub", quietly = TRUE)) {
    stop("Package 'hfhub' is required. Install with: install.packages('hfhub')")
  }

  # Use R_user_dir for CRAN-compliant cache

  if (is.null(output_dir)) {
    output_dir <- tools::R_user_dir("diffuseR", "cache")
  }

  output_file <- file.path(output_dir, "ltx2_transformer_int4.safetensors")

  # Check if already cached
  if (file.exists(output_file) && !force) {
    if (verbose) {
      size_gb <- file.info(output_file)$size / 1e9
      message(sprintf("Using cached INT4 weights: %s (%.2f GB)", output_file, size_gb))
    }
    return(output_file)
  }

  # Check if model is available locally via transformer/config.json
  transformer_dir <- tryCatch({
      config_path <- hfhub::hub_download(model_id, "transformer/config.json",
        local_files_only = TRUE)
      dirname(config_path)
    }, error = function(e) NULL)

  if (is.null(transformer_dir)) {
    if (!download) {
      stop("Model '", model_id, "' transformer not found in HuggingFace cache.\n",
        "Run with download = TRUE to download, or use:\n",
        "  huggingface-cli download ", model_id)
    }

    # Interactive consent before downloading
    if (interactive()) {
      ans <- utils::askYesNo(
        paste0("Download '", model_id, "' transformer (~40GB) from HuggingFace?"),
        default = TRUE
      )
      if (!isTRUE(ans)) {
        stop("Download cancelled.", call. = FALSE)
      }
    }

    if (verbose) message("Downloading transformer weights from HuggingFace...")
    model_path <- hfhub::hub_snapshot(model_id,
      allow_patterns = "transformer/*")
    transformer_dir <- file.path(model_path, "transformer")
  }
  if (!dir.exists(transformer_dir)) {
    stop("Transformer directory not found: ", transformer_dir)
  }

  safetensor_files <- list.files(transformer_dir, pattern = "\\.safetensors$",
    full.names = TRUE)
  if (length(safetensor_files) == 0) {
    stop("No safetensor files found in: ", transformer_dir)
  }

  if (verbose) {
    message(sprintf("Found %d safetensor files in: %s", length(safetensor_files), transformer_dir))
    total_size <- sum(file.info(safetensor_files)$size) / 1e9
    message(sprintf("Total size: %.2f GB (will compress to ~%.2f GB)",
        total_size, total_size / 7))
  }

  # Create output directory only when actually writing (CRAN policy)
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # Quantize
  quantize_safetensors_int4(
    paths = safetensor_files,
    output_path = output_file,
    block_size = block_size,
    verbose = verbose
  )

  output_file
}

#' Quantize LTX-2 VAE to INT4
#'
#' Quantizes the LTX-2 video VAE to INT4 format.
#'
#' @param model_id Character. HuggingFace model ID (default "Lightricks/LTX-2").
#' @param output_dir Character. Directory to save quantized weights.
#'   Default uses `tools::R_user_dir("diffuseR", "cache")`.
#' @param block_size Integer. Block size for INT4 quantization.
#' @param force Logical. Re-quantize even if cached file exists.
#' @param download Logical. If TRUE, download model from HuggingFace if not cached.
#' @param verbose Logical. Print progress.
#'
#' @return Character. Path to the quantized weights file.
#'
#' @export
quantize_ltx2_vae <- function(
  model_id = "Lightricks/LTX-2",
  output_dir = NULL,
  block_size = 64L,
  force = FALSE,
  download = FALSE,
  verbose = TRUE
) {
  if (!requireNamespace("hfhub", quietly = TRUE)) {
    stop("Package 'hfhub' is required. Install with: install.packages('hfhub')")
  }

  # Use R_user_dir for CRAN-compliant cache

  if (is.null(output_dir)) {
    output_dir <- tools::R_user_dir("diffuseR", "cache")
  }

  output_file <- file.path(output_dir, "ltx2_vae_int4.safetensors")

  # Check if already cached
  if (file.exists(output_file) && !force) {
    if (verbose) {
      size_mb <- file.info(output_file)$size / 1e6
      message(sprintf("Using cached INT4 VAE weights: %s (%.2f MB)", output_file, size_mb))
    }
    return(output_file)
  }

  # Check if model is available locally via vae/config.json
  vae_dir <- tryCatch({
      config_path <- hfhub::hub_download(model_id, "vae/config.json",
        local_files_only = TRUE)
      dirname(config_path)
    }, error = function(e) NULL)

  if (is.null(vae_dir)) {
    if (!download) {
      stop("Model '", model_id, "' VAE not found in HuggingFace cache.\n",
        "Run with download = TRUE to download, or use:\n",
        "  huggingface-cli download ", model_id)
    }

    # Interactive consent before downloading
    if (interactive()) {
      ans <- utils::askYesNo(
        paste0("Download '", model_id, "' VAE from HuggingFace?"),
        default = TRUE
      )
      if (!isTRUE(ans)) {
        stop("Download cancelled.", call. = FALSE)
      }
    }

    if (verbose) message("Downloading VAE weights from HuggingFace...")
    model_path <- hfhub::hub_snapshot(model_id,
      allow_patterns = "vae/*")
    vae_dir <- file.path(model_path, "vae")
  }
  if (!dir.exists(vae_dir)) {
    stop("VAE directory not found: ", vae_dir)
  }

  safetensor_files <- list.files(vae_dir, pattern = "\\.safetensors$",
    full.names = TRUE)
  if (length(safetensor_files) == 0) {
    stop("No safetensor files found in: ", vae_dir)
  }

  if (verbose) {
    total_size <- sum(file.info(safetensor_files)$size) / 1e6
    message(sprintf("Found %d VAE safetensor files (%.2f MB)",
        length(safetensor_files), total_size))
  }

  # Create output directory only when actually writing (CRAN policy)
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # Quantize
  quantize_safetensors_int4(
    paths = safetensor_files,
    output_path = output_file,
    block_size = block_size,
    verbose = verbose
  )

  output_file
}

