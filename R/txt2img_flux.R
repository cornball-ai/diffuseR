#' FLUX.1 Text-to-Image Pipeline
#'
#' FLUX latent packing helpers and (in later phases) the schnell
#' text-to-image pipeline. Ported from the diffusers reference
#' implementation (Apache-2.0, src/diffusers/pipelines/flux/
#' pipeline_flux.py).
#'
#' @name txt2img_flux
NULL

#' Pack FLUX latents into a patch sequence
#'
#' Packs a [B, C, H, W] latent into 2x2 patches, giving a sequence
#' [B, (H/2) * (W/2), C * 4]. Reference: FluxPipeline._pack_latents.
#'
#' @param latents Tensor of shape [B, C, H, W]; H and W must be even.
#'
#' @return Tensor of shape [B, (H/2) * (W/2), C * 4].
#'
#' @export
flux_pack_latents <- function(latents) {
    shape <- latents$shape
    b <- shape[1]
    ch <- shape[2]
    h <- shape[3]
    w <- shape[4]
    latents <- latents$view(c(b, ch, h %/% 2L, 2L, w %/% 2L, 2L))
    # Python permute (0, 2, 4, 1, 3, 5), 1-indexed here
    latents <- latents$permute(c(1L, 3L, 5L, 2L, 4L, 6L))
    latents$reshape(c(b, (h %/% 2L) * (w %/% 2L), ch * 4L))
}

#' Unpack a FLUX patch sequence back into latents
#'
#' Inverse of \code{flux_pack_latents}. Height and width are the target
#' image dimensions in pixels; the latent grid is derived via the VAE
#' scale factor and the 2x2 patch size. Reference:
#' FluxPipeline._unpack_latents.
#'
#' @param latents Tensor of shape [B, S, C_packed].
#' @param height,width Integers. Image height/width in pixels.
#' @param vae_scale_factor Integer. Spatial downsampling of the VAE (8).
#'
#' @return Tensor of shape [B, C_packed / 4, height / 8, width / 8].
#'
#' @export
flux_unpack_latents <- function(latents, height, width, vae_scale_factor = 8L) {
    shape <- latents$shape
    b <- shape[1]
    ch <- shape[3]
    h <- 2L * (as.integer(height) %/% (vae_scale_factor * 2L))
    w <- 2L * (as.integer(width) %/% (vae_scale_factor * 2L))
    latents <- latents$view(c(b, h %/% 2L, w %/% 2L, ch %/% 4L, 2L, 2L))
    # Python permute (0, 3, 1, 4, 2, 5), 1-indexed here
    latents <- latents$permute(c(1L, 4L, 2L, 5L, 3L, 6L))
    latents$reshape(c(b, ch %/% 4L, h, w))
}

# Resolve a FLUX.1-schnell support file from the HuggingFace cache
.flux1_cached <- function(file) {
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("The hfhub package is required to locate model files.")
    }
    tryCatch(
             hfhub::hub_download(.flux1_repo, file, local_files_only = TRUE),
             error = function(e) {
        stop("Missing ", file, " in the HuggingFace cache; ",
             "run download_flux1() first.", call. = FALSE)
    }
    )
}

#' Load the FLUX.1-schnell pipeline
#'
#' Loads the quantized transformer artifact plus the VAE decoder, CLIP
#' and T5 text encoders, tokenizer, and scheduler config (from the
#' HuggingFace cache populated by \code{\link{download_flux1}}).
#' Components load to the CPU when \code{phase_offload} is on and move
#' to the GPU only for their phase of the generation.
#'
#' @param model_dir Quantized artifact directory (default: the
#'   \code{download_flux1} location for \code{precision}), or a raw
#'   diffusers transformer directory for full-precision loading.
#' @param device Character. Compute device.
#' @param precision "nf4" or "fp8"; NULL picks the
#'   \code{\link{flux_memory_profile}} recommendation.
#' @param text_device Device for the text encoders ("cpu" default; the
#'   T5-XXL runs float32 there).
#' @param attn_chunk Integer or NULL. Attention query-chunk override.
#' @param phase_offload Logical. One GPU tenant per phase.
#' @param verbose Logical.
#'
#' @return A \code{flux_pipeline} list.
#'
#' @export
flux_load_pipeline <- function(model_dir = NULL, device = "cuda",
                               precision = NULL, text_device = "cpu",
                               attn_chunk = NULL, phase_offload = TRUE,
                               verbose = TRUE) {
    profile <- flux_memory_profile()
    if (is.null(precision)) {
        precision <- profile$precision
    }
    if (is.null(attn_chunk)) {
        attn_chunk <- profile$attn_chunk
    }
    if (is.null(model_dir)) {
        model_dir <- file.path(tools::R_user_dir("diffuseR", "data"),
                               paste0("flux1-schnell-", precision))
    }

    ckpt <- if (file.exists(file.path(model_dir, "manifest.json"))) {
        flux_open_quantized(model_dir)
    } else {
        flux_open_checkpoint(model_dir)
    }

    if (!nzchar(Sys.getenv("PYTORCH_CUDA_ALLOC_CONF"))) {
        # Must be set before the first CUDA allocation (see
        # ltx23_load_pipeline for the per-format rationale)
        conf <- if (identical(ckpt$format, "nf4")) {
            "backend:native"
        } else {
            "expandable_segments:True"
        }
        Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = conf)
    }
    if (device == "cuda") {
        footprint <- if (identical(ckpt$format, "nf4")) 8 else 4
        ltx23_tune_gc(footprint_gb = footprint)
    }

    component_device <- if (phase_offload) "cpu" else device

    pipe <- list(
                 format = ckpt$format %||% "full",
                 device = device,
                 text_device = text_device,
                 phase_offload = phase_offload,
                 attn_chunk = if (is.null(attn_chunk)) NULL else as.integer(attn_chunk),
                 config = ckpt$config
    )

    if (verbose) {
        message("Loading transformer (", pipe$format, ")...")
    }
    pipe$transformer <- flux_load_transformer(
                                              ckpt, device = component_device,
                                              pin = device == "cuda",
                                              verbose = verbose
    )

    vae_config <- jsonlite::fromJSON(.flux1_cached("vae/config.json"))
    pipe$vae_scaling_factor <- vae_config$scaling_factor %||% 0.3611
    pipe$vae_shift_factor <- vae_config$shift_factor %||% 0.1159
    if (verbose) {
        message("Loading VAE decoder...")
    }
    dec <- vae_decoder_native(
                              latent_channels = as.integer(vae_config$latent_channels %||% 16L)
    )
    load_decoder_safetensors(
                             dec, .flux1_cached("vae/diffusion_pytorch_model.safetensors"),
                             verbose = verbose
    )
    dec$to(device = component_device)
    dec$eval()
    pipe$decoder <- dec

    if (verbose) {
        message("Loading CLIP text encoder...")
    }
    clip <- text_encoder_native(gelu_type = "quick")
    load_text_encoder_safetensors(
                                  clip, .flux1_cached("text_encoder/model.safetensors"),
                                  verbose = verbose
    )
    clip$to(device = text_device)
    clip$eval()
    pipe$text_encoder <- clip

    if (verbose) {
        message("Loading T5 text encoder...")
    }
    t5_dir <- dirname(.flux1_cached("text_encoder_2/config.json"))
    pipe$text_encoder2 <- load_t5_text_encoder(
                                               t5_dir, device = text_device,
                                               dtype = if (text_device == "cpu") "float32" else "bfloat16",
                                               verbose = verbose
    )
    pipe$tokenizer2 <- unigram_tokenizer(.flux1_cached("tokenizer_2/tokenizer.json"))

    sched_cfg <- tryCatch(
                          jsonlite::fromJSON(.flux1_cached("scheduler/scheduler_config.json")),
                          error = function(e) NULL
    )
    pipe$scheduler_shift <- sched_cfg$shift %||% 1.0

    structure(pipe, class = "flux_pipeline")
}

# Flow-matching Euler loop over the packed latent sequence. Latents stay
# float32; the transformer runs in compute_dtype. schnell is CFG-free:
# one forward per step.
.flux_denoise <- function(transformer, latents, schedule, prompt_embeds,
                          pooled_prompt_embeds, image_rotary_emb,
                          compute_dtype, chunk_size = NULL, verbose = TRUE) {
    timesteps <- as.numeric(schedule$timesteps$cpu())
    n <- length(timesteps)
    pb <- if (verbose) {
        utils::txtProgressBar(min = 0, max = n, style = 3)
    } else {
        NULL
    }
    f32 <- torch::torch_float32()

    torch::with_no_grad({
        for (i in seq_len(n)) {
            t <- timesteps[[i]]
            # The transformer takes sigma-space time (it rescales by 1000
            # internally, matching the reference pipeline's t/1000)
            t_model <- torch::torch_tensor(t / 1000, dtype = compute_dtype,
                                           device = latents$device)$reshape(1L)

            noise_pred <- transformer(
                                      hidden_states = latents$to(dtype = compute_dtype),
                                      encoder_hidden_states = prompt_embeds,
                                      pooled_projections = pooled_prompt_embeds,
                                      timestep = t_model,
                                      image_rotary_emb = image_rotary_emb,
                                      chunk_size = chunk_size
            )

            step <- flowmatch_scheduler_step(
                                             noise_pred$to(dtype = f32), t, latents, schedule
            )
            latents <- step$prev_sample
            schedule <- step$schedule
            rm(noise_pred, step)
            if (!is.null(pb)) {
                utils::setTxtProgressBar(pb, i)
            }
        }
    })
    if (!is.null(pb)) {
        close(pb)
    }
    latents
}

#' Generate an image with FLUX.1-schnell
#'
#' 4-step distilled text-to-image generation (no classifier-free
#' guidance): T5 + CLIP prompt encoding, flow-matching Euler denoising
#' over the packed latent sequence, and 16-channel VAE decode. With
#' phase offloading each component is the sole GPU tenant for its phase.
#'
#' @param prompt Character. The prompt.
#' @param pipeline A \code{flux_pipeline} from
#'   \code{\link{flux_load_pipeline}}; NULL loads one (passing
#'   \code{...} through).
#' @param width,height Integers, divisible by 16.
#' @param num_inference_steps Integer. Denoising steps (schnell: 4).
#' @param max_sequence_length Integer. T5 token length (schnell: 256).
#' @param seed Integer or NULL. Initial latents are drawn on the CPU, so
#'   a seed matches a Python diffusers run with a CPU generator.
#' @param prompt_embeds,pooled_prompt_embeds Optional precomputed text
#'   embeddings (skip the text encoders).
#' @param save_file Logical. Write a PNG.
#' @param filename Output path (default derived from the prompt).
#' @param verbose Logical.
#' @param ... Passed to \code{\link{flux_load_pipeline}} when
#'   \code{pipeline} is NULL.
#'
#' @return Invisibly, \code{list(image, metadata)} where \code{image} is
#'   an [H, W, 3] array in [0, 1].
#'
#' @export
txt2img_flux <- function(prompt, pipeline = NULL, width = 1024L,
                         height = 1024L, num_inference_steps = 4L,
                         max_sequence_length = 256L, seed = NULL,
                         prompt_embeds = NULL, pooled_prompt_embeds = NULL,
                         save_file = TRUE, filename = NULL, verbose = TRUE,
                         ...) {
    if (is.null(pipeline)) {
        pipeline <- flux_load_pipeline(..., verbose = verbose)
    }
    device <- pipeline$device
    width <- as.integer(width)
    height <- as.integer(height)
    if (width %% 16L != 0L || height %% 16L != 0L) {
        stop("width and height must be divisible by 16")
    }

    f32 <- torch::torch_float32()
    compute_dtype <- if (device == "cpu") {
        f32
    } else {
        torch::torch_bfloat16()
    }

    phase_offload <- isTRUE(pipeline$phase_offload) && device != "cpu"
    onload <- function(module) {
        if (phase_offload) {
            module$to(device = device)
        }
        module
    }
    offload <- function(module) {
        if (phase_offload) {
            module$to(device = "cpu")
            clear_vram()
        }
        invisible(module)
    }

    t0 <- Sys.time()

    # --- Phase 1: text encoding ------------------------------------------------
    torch::with_no_grad({
        if (is.null(prompt_embeds)) {
            if (verbose) {
                message("Encoding prompt (T5)...")
            }
            prompt_embeds <- encode_with_t5(
                                            prompt, pipeline$text_encoder2, pipeline$tokenizer2,
                                            max_sequence_length = max_sequence_length
            )
        }
        if (is.null(pooled_prompt_embeds)) {
            tokens <- CLIPTokenizer(prompt)
            clip_device <- pipeline$text_encoder$token_embedding$weight$device
            tokens <- tokens$to(device = clip_device)
            hidden <- pipeline$text_encoder(tokens)
            pooled_prompt_embeds <- clip_pooled_output(hidden, tokens)
        }
    })
    prompt_embeds <- prompt_embeds$to(device = device, dtype = compute_dtype)
    pooled_prompt_embeds <- pooled_prompt_embeds$to(device = device,
                                                    dtype = compute_dtype)
    txt_len <- prompt_embeds$shape[2]

    # --- Phase 2: latents, rotary embeddings, schedule --------------------------
    lat_ch <- as.integer((pipeline$config$in_channels %||% 64L) %/% 4L)
    h_lat <- 2L * (height %/% 16L)
    w_lat <- 2L * (width %/% 16L)
    if (!is.null(seed)) {
        torch::torch_manual_seed(seed)
    }
    # Drawn on the CPU in the unpacked diffusers shape for seed parity
    latents <- torch::torch_randn(c(1L, lat_ch, h_lat, w_lat), dtype = f32)
    latents <- flux_pack_latents(latents)$to(device = device)

    img_ids <- flux_prepare_latent_image_ids(h_lat %/% 2L, w_lat %/% 2L)
    txt_ids <- torch::torch_zeros(txt_len, 3L)
    ids <- torch::torch_cat(list(txt_ids, img_ids), dim = 1L)
    rope <- flux_pos_embed(ids,
                           axes_dim = pipeline$transformer$axes_dims_rope %||% c(16L, 56L, 56L))
    rope <- list(rope[[1]]$to(device = device), rope[[2]]$to(device = device))

    sched <- flowmatch_scheduler_create(
                                        shift = pipeline$scheduler_shift %||% 1.0,
                                        use_dynamic_shifting = FALSE
    )
    n_steps <- as.integer(num_inference_steps)
    sched <- flowmatch_set_timesteps(
                                     sched, n_steps,
                                     sigmas = seq(1, 1 / n_steps, length.out = n_steps)
    )

    # --- Phase 3: denoise --------------------------------------------------------
    transformer <- onload(pipeline$transformer)
    if (verbose) {
        message(sprintf("Denoising: %d steps at %dx%d...", n_steps, width,
                        height))
    }
    latents <- .flux_denoise(
                             transformer, latents, sched, prompt_embeds,
                             pooled_prompt_embeds, rope, compute_dtype,
                             chunk_size = pipeline$attn_chunk, verbose = verbose
    )
    offload(pipeline$transformer)
    ltx23_release_dequant_buffers()

    # --- Phase 4: decode -----------------------------------------------------------
    if (verbose) {
        message("Decoding...")
    }
    latents <- flux_unpack_latents(latents, height, width)
    latents <- latents$div(pipeline$vae_scaling_factor %||% 0.3611)$
    add(pipeline$vae_shift_factor %||% 0.1159)

    decoder <- pipeline$decoder
    if (phase_offload) {
        decoder$to(device = device, dtype = compute_dtype)
    }
    torch::with_no_grad({
        dec_param <- decoder$conv_in$weight
        img <- decoder(latents$to(device = dec_param$device,
                                  dtype = dec_param$dtype))
        img <- img$to(dtype = f32)$cpu()
    })
    offload(decoder)

    img <- img$squeeze(1)$permute(c(2L, 3L, 1L))
    img <- img$add(1)$div(2)$clamp(0, 1)
    img_array <- as.array(img)

    gen_seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    if (verbose) {
        message(sprintf("Generated in %.1f s", gen_seconds))
    }

    if (save_file) {
        if (is.null(filename)) {
            filename <- filename_from_prompt(prompt)
        }
        save_image(img_array, filename)
        if (verbose) {
            message("Saved to ", filename)
        }
    }

    metadata <- list(
                     prompt = prompt, width = width, height = height,
                     steps = n_steps, seed = seed, model = "flux1-schnell",
                     precision = pipeline$format, seconds = gen_seconds
    )
    invisible(list(image = img_array, metadata = metadata))
}
