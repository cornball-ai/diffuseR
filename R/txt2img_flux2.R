#' FLUX.2 Klein Text-to-Image Pipeline
#'
#' Klein text-to-image, ported from the diffusers reference
#' (Apache-2.0, src/diffusers/pipelines/flux2/pipeline_flux2_klein.py).
#' Klein-4B is step-distilled: no classifier-free guidance and no
#' guidance embedding; the FlowMatch schedule uses dynamic shifting with
#' the BFL empirical mu. Latent noise is drawn directly in the packed
#' 128-channel space; only VAE-encoded latents get the BatchNorm
#' normalization (txt2img never encodes).
#'
#' @name txt2img_flux2
NULL

# Resolve a FLUX.2-klein support file from the HuggingFace cache
.flux2_cached <- function(file) {
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("The hfhub package is required to locate model files.")
    }
    tryCatch(
             hfhub::hub_download(.flux2_repo, file, local_files_only = TRUE),
             error = function(e) {
        stop("Missing ", file, " in the HuggingFace cache; ",
             "run download_flux2_klein() first.", call. = FALSE)
    }
    )
}

#' Load the FLUX.2 klein pipeline
#'
#' Loads the quantized transformer artifact plus the FLUX.2 VAE decoder,
#' Qwen3 text encoder, and tokenizer from the HuggingFace cache
#' populated by \code{\link{download_flux2_klein}}. With fp8 precision
#' the ~4 GB transformer stays GPU-resident.
#'
#' @param model_dir Quantized artifact directory (default: the
#'   \code{download_flux2_klein} location for \code{precision}), or a
#'   raw diffusers transformer directory.
#' @param device Character. Compute device.
#' @param precision "auto" (default: reuse an existing artifact, else
#'   fp8 when safetensors supports float8, else nf4), "fp8", or "nf4".
#' @param text_device Device for the Qwen3 encoder (default:
#'   \code{device}; it encodes in its own phase and offloads).
#' @param attn_chunk Integer or NULL. Attention query-chunk override.
#' @param phase_offload Logical. One GPU tenant per phase.
#' @param verbose Logical.
#'
#' @return A \code{flux2_pipeline} list.
#'
#' @export
flux2_load_pipeline <- function(model_dir = NULL, device = "cuda",
                                precision = c("auto", "fp8", "nf4"),
                                text_device = NULL, attn_chunk = NULL,
                                phase_offload = TRUE, verbose = TRUE) {
    precision <- .flux_resolve_precision(match.arg(precision),
        file.path(tools::R_user_dir("diffuseR", "data"), "flux2-klein-4b-"))
    if (is.null(text_device)) {
        if (device == "cuda") {
            text_device <- "cuda"
        } else {
            text_device <- "cpu"
        }
    }
    if (is.null(model_dir)) {
        model_dir <- file.path(tools::R_user_dir("diffuseR", "data"),
                               paste0("flux2-klein-4b-", precision))
    }

    ckpt <- if (file.exists(file.path(model_dir, "manifest.json"))) {
        flux_open_quantized(model_dir)
    } else {
        flux_open_checkpoint(model_dir)
    }

    if (!nzchar(Sys.getenv("PYTORCH_CUDA_ALLOC_CONF"))) {
        # Resident fp8 has an NF4-like stable footprint: the native
        # backend avoids expandable_segments' page-unmap cost on the
        # per-step activation churn (see ltx23_load_pipeline)
        Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "backend:native")
    }
    if (device == "cuda") {
        # Footprint sized to the LARGEST phase (the 8 GB Qwen3 encode),
        # not the transformer: a low footprint puts the allocator's
        # R-gc callback threshold under the working set, and every
        # callback walks the ~300k-object tokenizer heap (measured:
        # 13-20 s forwards at footprint 6 vs sub-second at 12)
        .flux_gc_gates(footprint_gb = 12)
    }

    if (phase_offload) {
        component_device <- "cpu"
    } else {
        component_device <- device
    }

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
        dtype = if (device == "cpu") "float32" else "bfloat16",
        pin = FALSE,
        fp8_resident = FALSE,
        verbose = verbose
    )
    # Resident fp8 happens at onload time (the weights ride to the GPU
    # with the phase and back off after)
    pipe$fp8_resident <- identical(pipe$format, "fp8") && device == "cuda"

    if (verbose) {
        message("Loading FLUX.2 VAE decoder...")
    }
    vae_config <- jsonlite::fromJSON(.flux2_cached("vae/config.json"))
    pipe$vae_bn_eps <- vae_config$batch_norm_eps %||% 1e-4
    pipe$decoder <- load_flux2_vae_decoder(
        .flux2_cached("vae/diffusion_pytorch_model.safetensors"),
        latent_channels = as.integer(vae_config$latent_channels %||% 32L),
        verbose = verbose
    )
    pipe$decoder$to(device = component_device)

    if (verbose) {
        message("Loading Qwen3 text encoder...")
    }
    te_dir <- dirname(.flux2_cached("text_encoder/config.json"))
    pipe$text_encoder <- load_qwen3_text_encoder(
        te_dir, device = if (phase_offload) "cpu" else text_device,
        dtype = if (text_device == "cpu") "float32" else "bfloat16",
        verbose = verbose
    )
    pipe$tokenizer <- qwen_bpe_tokenizer(.flux2_cached("tokenizer/tokenizer.json"))

    structure(pipe, class = "flux2_pipeline")
}

# Flow-matching Euler loop; klein is CFG-free (one forward per step)
.flux2_denoise <- function(transformer, latents, schedule, prompt_embeds,
                           image_rotary_emb, compute_dtype,
                           chunk_size = NULL, verbose = TRUE) {
    timesteps <- as.numeric(schedule$timesteps$cpu())
    n <- length(timesteps)
    pb <- .denoise_progress(n, NULL, verbose)
    f32 <- torch::torch_float32()

    torch::with_no_grad({
        for (i in seq_len(n)) {
            t <- timesteps[[i]]
            t_model <- torch::torch_tensor(t / 1000, dtype = compute_dtype,
                device = latents$device)$reshape(1L)

            noise_pred <- transformer(
                                      hidden_states = latents$to(dtype = compute_dtype),
                                      encoder_hidden_states = prompt_embeds,
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
            pb$tick(i)
        }
    })
    pb$done()
    latents
}

#' Generate an image with FLUX.2 klein
#'
#' Step-distilled text-to-image (klein-4B: 4 steps, no guidance): Qwen3
#' prompt encoding (chat template, mid-stack hidden states), FlowMatch
#' denoising with the empirical dynamic shift, and 32-channel VAE decode
#' through the BatchNorm latent statistics.
#'
#' @param prompt Character. The prompt.
#' @param pipeline A \code{flux2_pipeline} from
#'   \code{\link{flux2_load_pipeline}}; NULL loads one (passing
#'   \code{...} through).
#' @param width,height Integers, divisible by 16.
#' @param num_inference_steps Integer. Denoising steps (klein-4B: 4).
#' @param max_sequence_length Integer. Qwen3 token length (512).
#' @param seed Integer or NULL. Latents are drawn on the CPU in the
#'   packed shape, so a seed matches a Python diffusers run with a CPU
#'   generator.
#' @param prompt_embeds Optional precomputed [B, S, 7680] embeddings.
#' @param save_file Logical. Write a PNG.
#' @param filename Output path (default derived from the prompt).
#' @param verbose Logical, or one of "silent", "progress", "steps".
#'   TRUE = "steps" (full per-phase chatter), FALSE = "silent".
#'   "progress" prints a one-line generation summary plus a denoise
#'   progress bar (interactive) or periodic step ticks (captured logs).
#' @param ... Passed to \code{\link{flux2_load_pipeline}} when
#'   \code{pipeline} is NULL.
#'
#' @return Invisibly, \code{list(image, metadata)} where \code{image} is
#'   an [H, W, 3] array in [0, 1].
#'
#' @export
txt2img_flux2 <- function(prompt, pipeline = NULL, width = 1024L,
                          height = 1024L, num_inference_steps = 4L,
                          max_sequence_length = 512L, seed = NULL,
                          prompt_embeds = NULL, save_file = TRUE,
                          filename = NULL, verbose = TRUE, ...) {
    level <- .verbosity(verbose)
    verbose <- level == "steps"
    if (level != "silent") {
        message(sprintf("FLUX.2 klein: %dx%d, %d steps", as.integer(width),
                        as.integer(height), as.integer(num_inference_steps)))
    }
    if (is.null(pipeline)) {
        pipeline <- flux2_load_pipeline(..., verbose = verbose)
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

    # --- Phase 1: text encoding --------------------------------------------------
    if (is.null(prompt_embeds)) {
        if (verbose) {
            message("Encoding prompt (Qwen3)...")
        }
        onload(pipeline$text_encoder)
        te_device <- pipeline$text_encoder$model$embed_tokens$weight$device
        prompt_embeds <- encode_with_qwen3(prompt, pipeline$text_encoder,
            pipeline$tokenizer, max_sequence_length = max_sequence_length,
            device = te_device)
        offload(pipeline$text_encoder)
    }
    prompt_embeds <- prompt_embeds$to(device = device, dtype = compute_dtype)
    txt_len <- prompt_embeds$shape[2]

    # --- Phase 2: latents, rotary embeddings, schedule -----------------------------
    h2 <- height %/% 16L
    w2 <- width %/% 16L
    if (!is.null(seed)) {
        torch::torch_manual_seed(seed)
    }
    # Noise directly in the packed/patchified space (no BN for txt2img)
    latents <- torch::torch_randn(c(1L, 128L, h2, w2), dtype = f32)
    latents <- flux2_pack_latents(latents)$to(device = device)
    seq_img <- latents$shape[2]

    txt_ids <- flux2_prepare_text_ids(txt_len)
    latent_ids <- flux2_prepare_latent_ids(h2, w2)
    ids <- torch::torch_cat(list(txt_ids, latent_ids), dim = 1L)
    rope <- flux_pos_embed(
                           ids,
                           axes_dim = pipeline$transformer$axes_dims_rope %||% c(32L, 32L, 32L, 32L),
                           theta = pipeline$transformer$rope_theta %||% 2000
    )
    rope <- list(rope[[1]]$to(device = device), rope[[2]]$to(device = device))

    n_steps <- as.integer(num_inference_steps)
    mu <- flux2_empirical_mu(seq_img, n_steps)
    sched <- flowmatch_scheduler_create(
                                        use_dynamic_shifting = TRUE,
                                        time_shift_type = "exponential"
    )
    sched <- flowmatch_set_timesteps(
                                     sched, n_steps, mu = mu,
                                     sigmas = seq(1, 1 / n_steps, length.out = n_steps)
    )

    # --- Phase 3: denoise ------------------------------------------------------------
    transformer <- onload(pipeline$transformer)
    if (isTRUE(pipeline$fp8_resident)) {
        .flux_fp8_to_device(transformer, device)
    }
    if (verbose) {
        message(sprintf("Denoising: %d steps at %dx%d...", n_steps, width,
                        height))
    }
    latents <- .flux2_denoise(
                              transformer, latents, sched, prompt_embeds, rope,
                              compute_dtype, chunk_size = pipeline$attn_chunk,
                              verbose = level
    )
    if (isTRUE(pipeline$fp8_resident) && phase_offload) {
        .flux_fp8_to_device(pipeline$transformer, "cpu")
    }
    offload(pipeline$transformer)
    ltx23_release_dequant_buffers()

    # --- Phase 4: decode ---------------------------------------------------------------
    if (verbose) {
        message("Decoding...")
    }
    latents <- flux2_unpack_latents_with_ids(latents, latent_ids, h2, w2)
    latents <- flux2_bn_normalize(
                                  latents, pipeline$decoder$bn$running_mean,
                                  pipeline$decoder$bn$running_var,
                                  eps = pipeline$vae_bn_eps %||% 1e-4, inverse = TRUE
    )
    latents <- flux2_unpatchify_latents(latents)

    decoder <- pipeline$decoder
    if (phase_offload) {
        decoder$to(device = device, dtype = compute_dtype)
    }
    torch::with_no_grad({
        dec_param <- decoder$post_quant_conv$weight
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
                     steps = n_steps, seed = seed, model = "flux2-klein-4b",
                     precision = pipeline$format, seconds = gen_seconds
    )
    invisible(list(image = img_array, metadata = metadata))
}
