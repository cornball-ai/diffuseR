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
#' @param model Character. Model variant: "ltx2-2b" (default) or "ltx2-2b-distilled".
#'
#' @return A list with memory profile settings.
#'
#' @details
#' Memory profiles for LTX-2:
#' \describe{
#'   \item{high}{16GB+ - DiT and VAE on GPU, text via API}
#'   \item{medium}{12GB - DiT on GPU, VAE tiled, text via API}
#'   \item{low}{8GB - DiT offloaded layer-by-layer, VAE tiled small, text via API}
#'   \item{very_low}{6GB - Maximum offloading, minimum tiles}
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
ltx2_memory_profile <- function(vram_gb = NULL, model = "ltx2-2b") {
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
  profiles <- list(
    high = list(
      name = "high",
      dit_device = "cuda",
      dit_offload = FALSE,
      vae_device = "cuda",
      vae_tiling = FALSE,
      vae_tile_size = c(512L, 512L),
      vae_tile_frames = 16L,
      text_backend = "api",
      dtype = "float16",
      max_resolution = c(1080L, 1920L),  # height, width
      max_frames = 121L,
      cfg_mode = "batched"
    ),
    medium = list(
      name = "medium",
      dit_device = "cuda",
      dit_offload = FALSE,
      vae_device = "cuda",
      vae_tiling = TRUE,
      vae_tile_size = c(512L, 512L),
      vae_tile_frames = 16L,
      text_backend = "api",
      dtype = "float16",
      max_resolution = c(720L, 1280L),
      max_frames = 121L,
      cfg_mode = "batched"
    ),
    low = list(
      name = "low",
      dit_device = "cuda",
      dit_offload = TRUE,  # Layer-by-layer offloading
      vae_device = "cuda",
      vae_tiling = TRUE,
      vae_tile_size = c(256L, 256L),  # Smaller tiles
      vae_tile_frames = 8L,
      text_backend = "api",
      dtype = "float16",
      max_resolution = c(480L, 854L),
      max_frames = 61L,
      cfg_mode = "sequential"  # Run uncond/cond separately
    ),
    very_low = list(
      name = "very_low",
      dit_device = "cuda",
      dit_offload = TRUE,
      vae_device = "cpu",  # VAE decode on CPU
      vae_tiling = TRUE,
      vae_tile_size = c(128L, 128L),
      vae_tile_frames = 4L,
      text_backend = "api",
      dtype = "float16",
      max_resolution = c(480L, 640L),
      max_frames = 33L,
      cfg_mode = "sequential"
    ),
    cpu_only = list(
      name = "cpu_only",
      dit_device = "cpu",
      dit_offload = FALSE,
      vae_device = "cpu",
      vae_tiling = TRUE,
      vae_tile_size = c(256L, 256L),
      vae_tile_frames = 8L,
      text_backend = "api",
      dtype = "float32",  # CPU often faster with float32
      max_resolution = c(480L, 640L),
      max_frames = 33L,
      cfg_mode = "sequential"
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
 if (requireNamespace("gpuctl", quietly = TRUE)) {
    return(gpuctl::gpu_is_blackwell())
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
  if (requireNamespace("gpuctl", quietly = TRUE)) {
    info <- gpuctl::gpu_detect()
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
offload_to_cpu <- function(module, gc = TRUE) {
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
load_to_gpu <- function(module, device = "cuda") {
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
  if (requireNamespace("gpuctl", quietly = TRUE)) {
    info <- gpuctl::gpu_detect()
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


#' DiT Layer-by-Layer Forward Pass
#'
#' Runs transformer layers one at a time, moving each to GPU before
#' computation and back to CPU after. For extreme memory-constrained scenarios.
#'
#' @param hidden_states Input tensor.
#' @param layers List of transformer layers (on CPU).
#' @param device Target device for computation.
#' @param ... Additional arguments passed to each layer.
#'
#' @return Output tensor (on CPU).
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # During low-VRAM inference
#' output <- dit_offloaded_forward(
#'   hidden_states,
#'   model$transformer_blocks,
#'   device = "cuda",
#'   encoder_hidden_states = text_embeds
#' )
#' }
dit_offloaded_forward <- function(hidden_states, layers, device = "cuda", ...) {
  # Move input to target device
  x <- hidden_states$to(device = device)

  for (i in seq_along(layers)) {
    # Load layer to GPU
    layers[[i]]$to(device = device)

    # Forward pass
    x <- layers[[i]](x, ...)

    # Offload layer back to CPU
    layers[[i]]$to(device = "cpu")

    # Clear cache periodically
    if (i %% 4 == 0) {
      torch::cuda_empty_cache()
    }
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
sequential_cfg_forward <- function(model, latents, timestep, prompt_embeds,
                                    negative_prompt_embeds, guidance_scale, ...) {
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
validate_resolution <- function(height, width, num_frames, profile) {
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
configure_vae_for_profile <- function(vae, profile) {
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
