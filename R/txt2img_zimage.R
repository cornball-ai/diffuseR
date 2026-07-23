#' Z-Image-Turbo Text-to-Image Pipeline
#'
#' Z-Image-Turbo text-to-image, ported from the diffusers reference
#' (Apache-2.0, src/diffusers/pipelines/z_image/pipeline_z_image.py).
#' Turbo is guidance-distilled: 8 steps, no classifier-free guidance.
#' The FlowMatch schedule uses the checkpoint's static shift (3.0) on
#' sigmas linspace(1, 1/N, N); the model consumes the REVERSED
#' normalized timestep (1000 - t)/1000 and its output is negated before
#' the Euler step. The VAE is the FLUX.1 16-channel autoencoder
#' verbatim.
#'
#' @name txt2img_zimage
NULL

# Resolve a Z-Image support file from the HuggingFace cache
.zimage_cached <- function(file) {
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("The hfhub package is required to locate model files.")
    }
    tryCatch(
             hfhub::hub_download(.zimage_repo, file, local_files_only = TRUE),
             error = function(e) {
        stop("Missing ", file, " in the HuggingFace cache; ",
             "run download_zimage_turbo() first.", call. = FALSE)
    }
    )
}

#' Load the Z-Image-Turbo pipeline
#'
#' Loads the quantized transformer artifact plus the 16-channel VAE
#' decoder, Qwen3-4B text encoder, and tokenizer from the HuggingFace
#' cache populated by \code{\link{download_zimage_turbo}}. With fp8
#' precision the ~6.3 GB transformer rides to the GPU per phase.
#'
#' @param model_dir Quantized artifact directory (default: the
#'   \code{download_zimage_turbo} location for \code{precision}), or a
#'   raw diffusers transformer directory.
#' @param device Character. Compute device.
#' @param precision "auto" (default: reuse an existing artifact, else
#'   fp8 when safetensors supports float8, else nf4), "fp8", or "nf4".
#' @param text_device Device for the Qwen3 encoder (default:
#'   \code{device}; it encodes in its own phase and offloads).
#' @param attn_chunk Integer or NULL. Attention query-chunk override.
#' @param phase_offload Logical. One GPU tenant per phase.
#' @param pin Logical or NULL. Page-lock the phase-swapped weights for
#'   DMA-rate transfer (see \code{\link{staging}}). NULL (default)
#'   resolves via \code{options(diffuseR.pin_staging)} then the
#'   host-RAM-aware \code{\link{recommend}} decision.
#' @param verbose Logical.
#'
#' @return A \code{zimage_pipeline} list.
#'
#' @export
zimage_load_pipeline <- function(model_dir = NULL, device = "cuda",
                                 precision = c("auto", "fp8", "nf4"),
                                 text_device = NULL, attn_chunk = NULL,
                                 phase_offload = TRUE, pin = NULL,
                                 verbose = TRUE) {
    precision <- .flux_resolve_precision(match.arg(precision),
        file.path(tools::R_user_dir("diffuseR", "data"), "zimage-turbo-"))
    if (is.null(text_device)) {
        if (device == "cuda") {
            text_device <- "cuda"
        } else {
            text_device <- "cpu"
        }
    }
    if (is.null(model_dir)) {
        model_dir <- file.path(tools::R_user_dir("diffuseR", "data"),
                               paste0("zimage-turbo-", precision))
    }

    ckpt <- if (file.exists(file.path(model_dir, "manifest.json"))) {
        flux_open_quantized(model_dir)
    } else {
        flux_open_checkpoint(model_dir)
    }
    pin <- .resolve_pin(pin, "zimage", ckpt$format %||% "full")

    if (!nzchar(Sys.getenv("PYTORCH_CUDA_ALLOC_CONF"))) {
        # Stable resident footprint: the native backend avoids
        # expandable_segments' page-unmap cost on per-step churn
        Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "backend:native")
    }
    if (device == "cuda") {
        # Footprint sized to the largest phase (the 8 GB Qwen3 encode),
        # same reasoning as the FLUX.2 klein pipeline
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
        message("Loading VAE decoder...")
    }
    vae_config <- jsonlite::fromJSON(.zimage_cached("vae/config.json"))
    pipe$vae_scaling_factor <- vae_config$scaling_factor %||% 0.3611
    pipe$vae_shift_factor <- vae_config$shift_factor %||% 0.1159
    dec <- vae_decoder_native(
                              latent_channels = as.integer(vae_config$latent_channels %||% 16L)
    )
    load_decoder_safetensors(
                             dec, .zimage_cached("vae/diffusion_pytorch_model.safetensors"),
                             verbose = verbose
    )
    dec$to(device = component_device)
    if (device == "cuda" && phase_offload) {
        # Cast to compute dtype once so the per-generation onload is
        # device-only and the pinned copy is bf16 (see flux2 loader).
        dec$to(dtype = torch::torch_bfloat16())
    }
    dec$eval()
    pipe$decoder <- dec

    sched_config <- jsonlite::fromJSON(.zimage_cached("scheduler/scheduler_config.json"))
    pipe$sched_shift <- sched_config$shift %||% 3.0

    if (verbose) {
        message("Loading Qwen3 text encoder...")
    }
    te_dir <- dirname(.zimage_cached("text_encoder/config.json"))
    te_config <- jsonlite::fromJSON(file.path(te_dir, "config.json"))
    # hidden_states[-2]: the state after num_hidden_layers - 1 layers
    pipe$te_penult_layer <- as.integer(te_config$num_hidden_layers %||% 36L) - 1L
    pipe$text_encoder <- load_qwen3_text_encoder(
        te_dir, device = if (phase_offload) "cpu" else text_device,
        dtype = if (text_device == "cpu") "float32" else "bfloat16",
        verbose = verbose
    )
    pipe$tokenizer <- qwen_bpe_tokenizer(.zimage_cached("tokenizer/tokenizer.json"))

    components <- c("transformer", "decoder")
    if (!identical(text_device, "cpu")) {
        components <- c(components, "text_encoder")
    }
    pipe$staging <- .flux_build_staging(pipe, pin, phase_offload, device,
                                        components, verbose = verbose)

    structure(pipe, class = "zimage_pipeline")
}

# Qwen3 prompt encoding, Z-Image style: thinking-enabled chat template,
# penultimate hidden state, valid tokens only (right padding -> first n)
.zimage_encode_prompt <- function(prompt, model, tokenizer, penult_layer,
                                  max_sequence_length = 512L, device = NULL) {
    enc <- encode_qwen(tokenizer, prompt, max_length = max_sequence_length,
                       chat_template = TRUE, enable_thinking = TRUE)
    device <- device %||% model$model$embed_tokens$weight$device
    long <- torch::torch_long()
    ids <- torch::torch_tensor(enc$input_ids + 1L, dtype = long,
                               device = device)
    mask <- torch::torch_tensor(enc$attention_mask, dtype = long,
                                device = device)

    states <- torch::with_no_grad(model(ids, attention_mask = mask,
                                        out_layers = penult_layer))
    n_real <- sum(enc$attention_mask[1,])
    states[[1]][1, 1:n_real,] # [L, hidden]
}

# Flow-matching Euler loop; Turbo is CFG-free (one forward per step).
# The model sees the reversed normalized timestep and predicts the
# negated velocity.
.zimage_denoise <- function(transformer, latents, schedule, cap_feats,
                            compute_dtype, chunk_size = NULL, verbose = TRUE) {
    timesteps <- as.numeric(schedule$timesteps$cpu())
    n <- length(timesteps)
    pb <- .denoise_progress(n, NULL, verbose)
    f32 <- torch::torch_float32()

    torch::with_no_grad({
        for (i in seq_len(n)) {
            t <- timesteps[[i]]
            t_model <- torch::torch_tensor((1000 - t) / 1000, dtype = f32,
                device = latents$device)$reshape(1L)

            # [1, 16, H8, W8] -> [16, 1, H8, W8] (frame axis)
            x_in <- latents$squeeze(1L)$unsqueeze(2L)$to(dtype = compute_dtype)
            out <- transformer(x_in, t_model, cap_feats,
                               chunk_size = chunk_size)
            noise_pred <- out$squeeze(2L)$unsqueeze(1L)$to(dtype = f32)$neg()

            step <- flowmatch_scheduler_step(noise_pred, t, latents, schedule)
            latents <- step$prev_sample
            schedule <- step$schedule
            rm(out, noise_pred, step)
            pb$tick(i)
        }
    })
    pb$done()
    latents
}

#' Generate an image with Z-Image-Turbo
#'
#' Guidance-distilled text-to-image (8 steps, no CFG): Qwen3-4B prompt
#' encoding (thinking-enabled chat template, penultimate hidden state),
#' FlowMatch denoising with the reversed-timestep convention, and
#' 16-channel VAE decode. Strong at legible text rendering, English and
#' Chinese both.
#'
#' @param prompt Character. The prompt.
#' @param pipeline A \code{zimage_pipeline} from
#'   \code{\link{zimage_load_pipeline}}; NULL loads one (passing
#'   \code{...} through).
#' @param width,height Integers, divisible by 16.
#' @param num_inference_steps Integer. Denoising steps (Turbo: 8).
#' @param max_sequence_length Integer. Qwen3 token length (512).
#' @param seed Integer or NULL. Latents are drawn on the CPU, so a seed
#'   matches a Python diffusers run with a CPU generator.
#' @param prompt_embeds Optional precomputed [L, 2560] caption
#'   embeddings (valid tokens only).
#' @param save_file Logical. Write a PNG.
#' @param filename Output path (default derived from the prompt).
#' @param verbose Logical, or one of "silent", "progress", "steps".
#'   TRUE = "steps" (full per-phase chatter), FALSE = "silent".
#'   "progress" prints a one-line generation summary plus a denoise
#'   progress bar (interactive) or periodic step ticks (captured logs).
#' @param ... Passed to \code{\link{zimage_load_pipeline}} when
#'   \code{pipeline} is NULL.
#'
#' @return Invisibly, \code{list(image, metadata)} where \code{image} is
#'   an [H, W, 3] array in [0, 1].
#'
#' @export
txt2img_zimage <- function(prompt, pipeline = NULL, width = 1024L,
                           height = 1024L, num_inference_steps = 8L,
                           max_sequence_length = 512L, seed = NULL,
                           prompt_embeds = NULL, save_file = TRUE,
                           filename = NULL, verbose = TRUE, ...) {
    level <- .verbosity(verbose)
    verbose <- level == "steps"
    if (level != "silent") {
        message(sprintf("Z-Image Turbo: %dx%d, %d steps", as.integer(width),
                        as.integer(height), as.integer(num_inference_steps)))
    }
    if (is.null(pipeline)) {
        pipeline <- zimage_load_pipeline(..., verbose = verbose)
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
    staging <- pipeline$staging %||% list()
    # Components move by name so pinned staging (see staging.R) can carry
    # the CPU<->GPU transfer when the loader prepared it; otherwise the
    # pageable module$to() path runs.
    onload <- function(what) {
        module <- pipeline[[what]]
        if (phase_offload) {
            st <- staging[[what]]
            if (is.null(st)) {
                module$to(device = device)
            } else {
                .staged_onload(st, device)
            }
        }
        module
    }
    offload <- function(what) {
        module <- pipeline[[what]]
        if (phase_offload) {
            st <- staging[[what]]
            if (is.null(st)) {
                module$to(device = "cpu")
            } else {
                .staged_offload(st)
            }
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
        # An explicit text_device = "cpu" keeps Qwen3 fp32 on the host
        # (it computes in place); only onload it when it phase-swaps.
        gpu_encode <- !identical(pipeline$text_device, "cpu")
        if (gpu_encode) {
            onload("text_encoder")
        }
        te_device <- pipeline$text_encoder$model$embed_tokens$weight$device
        prompt_embeds <- .zimage_encode_prompt(prompt, pipeline$text_encoder,
            pipeline$tokenizer,
            penult_layer = pipeline$te_penult_layer %||% 35L,
            max_sequence_length = max_sequence_length, device = te_device)
        if (gpu_encode) {
            offload("text_encoder")
        }
    }
    prompt_embeds <- prompt_embeds$to(device = device, dtype = compute_dtype)

    # --- Phase 2: latents and schedule ----------------------------------------------
    h8 <- height %/% 8L
    w8 <- width %/% 8L
    if (!is.null(seed)) {
        torch::torch_manual_seed(seed)
    }
    latents <- torch::torch_randn(c(1L, 16L, h8, w8), dtype = f32)$
    to(device = device)

    n_steps <- as.integer(num_inference_steps)
    sched <- flowmatch_scheduler_create(
                                        shift = pipeline$sched_shift %||% 3.0,
                                        use_dynamic_shifting = FALSE
    )
    sched <- flowmatch_set_timesteps(
                                     sched, n_steps,
                                     sigmas = seq(1, 1 / n_steps, length.out = n_steps)
    )

    # --- Phase 3: denoise ------------------------------------------------------------
    transformer <- onload("transformer")
    # Resident fp8's plain weight fields ride to the GPU with the phase.
    # When staging covers the transformer it already moved them (and the
    # field reassignment here would orphan the staged pairs), so skip it.
    fp8_manual <- isTRUE(pipeline$fp8_resident) &&
    is.null(staging[["transformer"]])
    if (fp8_manual) {
        .flux_fp8_to_device(transformer, device)
    }
    if (verbose) {
        message(sprintf("Denoising: %d steps at %dx%d...", n_steps, width,
                        height))
    }
    latents <- .zimage_denoise(
                               transformer, latents, sched, prompt_embeds,
                               compute_dtype, chunk_size = pipeline$attn_chunk,
                               verbose = level
    )
    if (fp8_manual && phase_offload) {
        .flux_fp8_to_device(pipeline$transformer, "cpu")
    }
    offload("transformer")
    ltx23_release_dequant_buffers()

    # --- Phase 4: decode ---------------------------------------------------------------
    if (verbose) {
        message("Decoding...")
    }
    latents <- latents$div(pipeline$vae_scaling_factor %||% 0.3611)$
    add(pipeline$vae_shift_factor %||% 0.1159)

    decoder <- onload("decoder")
    torch::with_no_grad({
        dec_param <- decoder$conv_in$weight
        img <- decoder(latents$to(device = dec_param$device,
                                  dtype = dec_param$dtype))
        img <- img$to(dtype = f32)$cpu()
    })
    offload("decoder")

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
                     steps = n_steps, seed = seed, model = "zimage-turbo",
                     precision = pipeline$format, seconds = gen_seconds
    )
    invisible(list(image = img_array, metadata = metadata))
}
