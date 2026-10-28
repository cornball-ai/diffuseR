#' FLUX 3 Image Klein Pipeline
#'
#' FLUX 3 Image Klein text-to-image and single-reference editing, ported
#' from the diffusers reference (Apache-2.0,
#' src/diffusers/modular_pipelines/flux3/). The distilled checkpoint is a
#' flow map: 16 steps, no classifier-free guidance, and every step also
#' tells the transformer the timestep it jumps to. The FlowMatch
#' schedule uses the static shift 6.93. Latent noise is drawn directly in
#' the packed 128-channel space.
#'
#' The weights are under the FLUX Kommunity License (non-commercial use,
#' plus commercial use of outputs by Qualifying Users with under US$5M
#' revenue, who must filter or review outputs; see the license in the
#' downloaded repository). diffuseR never redistributes them.
#'
#' @name txt2img_flux3
NULL

.flux3_repo <- "black-forest-labs/flux-3-image-klein"

# The snapshot the port was validated against; support files resolve
# straight out of snapshots/<revision>/ (see .flux2_rev).
.flux3_revision <- "337337621b8526e120f2275d250242f4a6bedd75"

.flux3_artifact_prefix <- function() {
    file.path(tools::R_user_dir("diffuseR", "data"), "flux3-klein-")
}

# Resolve a FLUX 3 file from the HuggingFace cache
.flux3_cached <- function(file, revision = .flux3_revision) {
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("The hfhub package is required to locate model files.")
    }
    rev <- .flux2_rev(revision)
    tryCatch(
             do.call(hfhub::hub_download,
                     c(list(.flux3_repo, file, local_files_only = TRUE), rev)),
             error = function(e) {
        stop("Missing ", file, " in the HuggingFace cache; ",
             "run download_flux3_klein() first.", call. = FALSE)
    }
    )
}

#' Load the FLUX 3 Image Klein pipeline
#'
#' Loads the quantized transformer artifact plus the VAE (encoder and
#' decoder), the Qwen3-VL text encoder (text model only) and the
#' tokenizer from the HuggingFace cache populated by
#' \code{\link{download_flux3_klein}}. With phase offload each component
#' visits the GPU only for its own phase, so the GPU peak is the largest
#' phase rather than the sum: the ~8 GB Qwen3-VL encode, or the fp8
#' transformer (~6 GB) plus activations.
#'
#' @param model_dir Quantized artifact directory (default: the
#'   \code{download_flux3_klein} location for \code{precision}), or a raw
#'   diffusers transformer directory.
#' @param device Character. Compute device.
#' @param precision "auto" (default: reuse an existing artifact, else
#'   fp8 when safetensors supports float8, else nf4), "fp8", "nf4", or
#'   "bf16" (the raw diffusers transformer).
#' @param text_device Device for the Qwen3-VL encoder (default:
#'   \code{device}; it encodes in its own phase and offloads). "cpu"
#'   keeps it in float32 on the host.
#' @param attn_chunk Integer or NULL. Attention query-chunk override.
#' @param phase_offload Logical. One GPU tenant per phase.
#' @param pin Logical or NULL. Page-lock the phase-swapped weights (see
#'   \code{\link{staging}}).
#' @param encoder Logical. Also load the VAE encoder, which
#'   \code{\link{img2img_flux3}} needs.
#' @param revision Exact 40-hex commit for the support files (default:
#'   the validated snapshot).
#' @param dtype Compute dtype, "bfloat16" or "float32". NULL (default):
#'   bfloat16 on CUDA, float32 on the CPU. "bfloat16" on the CPU halves
#'   host memory (the raw transformer is ~11 GB instead of ~22 GB).
#' @param verbose Logical.
#'
#' @return A \code{flux3_pipeline} list.
#'
#' @export
flux3_load_pipeline <- function(model_dir = NULL, device = "cuda",
                                precision = c("auto", "fp8", "nf4", "bf16"),
                                text_device = NULL, attn_chunk = NULL,
                                phase_offload = TRUE, pin = NULL,
                                encoder = TRUE, revision = .flux3_revision,
                                dtype = NULL, verbose = TRUE) {
    .flux2_rev(revision)
    precision <- match.arg(precision)
    if (precision != "bf16") {
        precision <- .flux_resolve_precision(precision,
            .flux3_artifact_prefix())
    }
    text_device <- text_device %||% device
    if (is.null(model_dir)) {
        model_dir <- if (precision == "bf16") {
            dirname(.flux3_cached("transformer/config.json", revision))
        } else {
            paste0(.flux3_artifact_prefix(), precision)
        }
    }

    ckpt <- if (file.exists(file.path(model_dir, "manifest.json"))) {
        flux_open_quantized(model_dir)
    } else {
        flux_open_checkpoint(model_dir)
    }
    pin <- .resolve_pin(pin, "flux3", ckpt$format %||% "full")

    if (!nzchar(Sys.getenv("PYTORCH_CUDA_ALLOC_CONF"))) {
        Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "backend:native")
    }
    if (device == "cuda") {
        # Gate footprint sized to the largest phase (the ~8 GB Qwen3-VL
        # encode), as for klein
        .flux_gc_gates(footprint_gb = 12)
    }
    component_device <- if (phase_offload) "cpu" else device
    compute_dtype <- dtype %||% if (device == "cpu") "float32" else "bfloat16"
    compute_dtype <- match.arg(compute_dtype, c("bfloat16", "float32"))

    pipe <- list(
                 format = ckpt$format %||% "full",
                 device = device,
                 compute_dtype = compute_dtype,
                 text_device = text_device,
                 phase_offload = phase_offload,
                 attn_chunk = if (is.null(attn_chunk)) NULL else as.integer(attn_chunk),
                 config = ckpt$config
    )

    if (verbose) {
        message("Loading FLUX 3 transformer (", pipe$format, ")...")
    }
    pipe$transformer <- flux_load_transformer(
        ckpt, device = component_device, dtype = compute_dtype,
        pin = FALSE, fp8_resident = FALSE, verbose = verbose
    )
    pipe$fp8_resident <- identical(pipe$format, "fp8") && device == "cuda"

    if (verbose) {
        message("Loading FLUX 3 VAE...")
    }
    vae_config <- jsonlite::fromJSON(.flux3_cached("vae/config.json", revision))
    vae_path <- .flux3_cached("vae/diffusion_pytorch_model.safetensors",
                              revision)
    latent_channels <- as.integer(vae_config$latent_channels %||% 32L)
    pipe$vae_bn_eps <- vae_config$batch_norm_eps %||% 1e-4
    pipe$decoder <- load_flux2_vae_decoder(vae_path,
        latent_channels = latent_channels,
        verbose = verbose)
    if (encoder) {
        pipe$encoder <- load_flux2_vae_encoder(vae_path,
            latent_channels = latent_channels,
            verbose = verbose)
    }
    for (nm in intersect(c("decoder", "encoder"), names(pipe))) {
        pipe[[nm]]$to(device = component_device)
        # The checkpoint VAE is float32; cast once so the per-phase
        # onload is device-only
        if (compute_dtype == "bfloat16") {
            pipe[[nm]]$to(dtype = torch::torch_bfloat16())
        }
    }

    if (verbose) {
        message("Loading Qwen3-VL text encoder...")
    }
    te_dir <- dirname(.flux3_cached("text_encoder/config.json", revision))
    for (f in c("text_encoder/model.safetensors.index.json",
                sprintf("text_encoder/model-%05d-of-00002.safetensors", 1:2))) {
        .flux3_cached(f, revision)
    }
    pipe$text_encoder <- load_qwen3_text_encoder(
        te_dir, device = if (phase_offload) "cpu" else text_device,
        dtype = if (text_device == "cpu") "float32" else "bfloat16",
        verbose = verbose
    )
    pipe$tokenizer <- qwen_bpe_tokenizer(
        .flux3_cached("text_encoder/tokenizer.json", revision))

    components <- intersect(c("transformer", "decoder", "encoder"), names(pipe))
    if (!identical(text_device, "cpu")) {
        components <- c(components, "text_encoder")
    }
    pipe$staging <- .flux_build_staging(pipe, pin, phase_offload, device,
                                        components, verbose = verbose)

    structure(pipe, class = "flux3_pipeline")
}

# Flow-map Euler loop: one CFG-free forward per step, given the current
# flow time and the time it jumps to (the next scheduler timestep, 0
# after the last). Reference-image tokens (img2img) are a separate
# transformer input; only the image tokens come back.
.flux3_denoise <- function(transformer, latents, schedule, prompt_embeds,
                           txt_ids, img_ids, compute_dtype,
                           cond_tokens = NULL, cond_ids = NULL,
                           chunk_size = NULL, verbose = TRUE) {
    timesteps <- as.numeric(schedule$timesteps$cpu())
    n <- length(timesteps)
    pb <- .denoise_progress(n, NULL, verbose)
    f32 <- torch::torch_float32()
    device <- latents$device
    as_t <- function(v) {
        torch::torch_tensor(v, dtype = f32, device = device)$div(1000)$to(dtype = compute_dtype)$reshape(1L)
    }

    torch::with_no_grad({
        for (i in seq_len(n)) {
            t <- timesteps[[i]]
            next_t <- if (i < n) timesteps[[i + 1L]] else 0
            noise_pred <- transformer(hidden_states = latents,
                                      encoder_hidden_states = prompt_embeds,
                                      timestep = as_t(t), img_ids = img_ids,
                                      txt_ids = txt_ids,
                                      image_cond_hidden_states = cond_tokens,
                                      image_cond_ids = cond_ids,
                                      target_timestep = as_t(next_t),
                                      chunk_size = chunk_size)
            # Euler step in float32, result back in the compute dtype
            # (diffusers FlowMatchEulerDiscreteScheduler.step)
            step <- flowmatch_scheduler_step(
                noise_pred$to(dtype = f32), t, latents$to(dtype = f32),
                schedule
            )
            latents <- step$prev_sample$to(dtype = compute_dtype)
            schedule <- step$schedule
            rm(noise_pred, step)
            pb$tick(i)
        }
    })
    pb$done()
    latents
}

#' Generate an image with FLUX 3 Image Klein
#'
#' Step-distilled text-to-image (16 steps, no guidance): Qwen3-VL prompt
#' encoding, flow-map FlowMatch denoising with the static shift 6.93, and
#' 32-channel VAE decode through the BatchNorm latent statistics.
#'
#' FLUX 3 takes full scene captions: a short prompt gives generic
#' output. Expand short requests into one detailed caption (framing,
#' subjects with colors, materials and positions, background, lighting)
#' with an LLM first. Layout boxes go in the caption as BFL documents
#' (a JSON list of elements with \code{bbox = [top, left, bottom,
#' right]} on a 0 to 1000 grid).
#'
#' @param prompt Character. The (full) caption.
#' @param pipeline A \code{flux3_pipeline} from
#'   \code{\link{flux3_load_pipeline}}; NULL loads one (passing
#'   \code{...} through).
#' @param width,height Integers, divisible by 16.
#' @param num_inference_steps Integer. Denoising steps (16).
#' @param max_sequence_length Integer. Qwen3-VL token cap (8192).
#' @param seed Integer or NULL. Latents are drawn on the CPU in float32
#'   in the packed shape, so a seed matches a diffusers run with a CPU
#'   generator.
#' @param prompt_embeds Optional precomputed [1, S, 20480] embeddings.
#' @param save_file Logical. Write a PNG.
#' @param filename Output path (default derived from the prompt).
#' @param verbose Logical, or one of "silent", "progress", "steps".
#' @param ... Passed to \code{\link{flux3_load_pipeline}} when
#'   \code{pipeline} is NULL.
#'
#' @return Invisibly, \code{list(image, metadata)} where \code{image} is
#'   an [H, W, 3] array in [0, 1].
#'
#' @export
txt2img_flux3 <- function(prompt, pipeline = NULL, width = 1024L,
                          height = 1024L, num_inference_steps = 16L,
                          max_sequence_length = 8192L, seed = NULL,
                          prompt_embeds = NULL, save_file = TRUE,
                          filename = NULL, verbose = TRUE, ...) {
    level <- .verbosity(verbose)
    if (level != "silent") {
        message(sprintf("FLUX 3 klein: %dx%d, %d steps", as.integer(width),
                        as.integer(height), as.integer(num_inference_steps)))
    }
    if (is.null(pipeline)) {
        pipeline <- flux3_load_pipeline(..., verbose = level == "steps")
    }
    .flux3_generate(prompt, pipeline, width, height, num_inference_steps,
                    max_sequence_length, seed, prompt_embeds, save_file,
                    filename, level)
}

# The shared FLUX 3 generation body. `reference` is NULL (txt2img) or the
# list from flux3_preprocess_image (img2img); `crop` is the box to crop
# the decoded image back to, or NULL.
.flux3_generate <- function(prompt, pipeline, width, height,
                            num_inference_steps, max_sequence_length, seed,
                            prompt_embeds, save_file, filename, level,
                            reference = NULL, crop = NULL) {
    verbose <- level == "steps"
    device <- pipeline$device
    width <- as.integer(width)
    height <- as.integer(height)
    if (width %% 16L != 0L || height %% 16L != 0L) {
        stop("width and height must be divisible by 16")
    }
    f32 <- torch::torch_float32()
    compute_dtype <- .flux_dtype(pipeline$compute_dtype %||%
        if (device == "cpu") "float32" else "bfloat16")

    phase_offload <- isTRUE(pipeline$phase_offload) && device != "cpu"
    staging <- pipeline$staging %||% list()
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
            message("Encoding prompt (Qwen3-VL)...")
        }
        gpu_encode <- !identical(pipeline$text_device, "cpu")
        if (gpu_encode) {
            onload("text_encoder")
        }
        te_device <- pipeline$text_encoder$model$embed_tokens$weight$device
        enc <- flux3_encode_prompt(prompt, pipeline$text_encoder,
                                   pipeline$tokenizer,
                                   max_sequence_length = max_sequence_length,
                                   device = te_device)
        prompt_embeds <- enc$prompt_embeds
        rm(enc)
        if (gpu_encode) {
            offload("text_encoder")
        }
    }
    prompt_embeds <- prompt_embeds$to(device = device, dtype = compute_dtype)
    txt_ids <- flux3_prepare_text_ids(prompt_embeds$shape[2], device = device)

    # --- Phase 2: latents, ids, schedule ------------------------------------------
    h2 <- height %/% 16L
    w2 <- width %/% 16L
    if (!is.null(seed)) {
        torch::torch_manual_seed(seed)
    }
    # Drawn in float32 (the reference's RNG stream), then held in the
    # compute dtype as the reference does
    latents <- torch::torch_randn(c(1L, 128L, h2, w2), dtype = f32)
    latents <- flux2_pack_latents(latents)$to(device = device,
        dtype = compute_dtype)
    img_ids <- flux3_prepare_image_ids(h2, w2, device = device)

    # --- Reference image (img2img): encode once, reuse every step -------------------
    cond_tokens <- NULL
    cond_ids <- NULL
    if (!is.null(reference)) {
        if (is.null(pipeline$encoder)) {
            stop("this pipeline has no VAE encoder; load it with ",
                 "flux3_load_pipeline(encoder = TRUE)", call. = FALSE)
        }
        if (verbose) {
            message("Encoding reference image...")
        }
        encoder <- onload("encoder")
        ref <- flux2_encode_image(encoder, reference,
                                  bn_eps = pipeline$vae_bn_eps %||% 1e-4)
        offload("encoder")
        cond_tokens <- ref$tokens$to(device = device, dtype = compute_dtype)
        cond_ids <- flux3_prepare_image_ids(ref$height, ref$width,
            l_coord = -1, device = device)
        rm(ref)
    }

    n_steps <- as.integer(num_inference_steps)
    sched <- flux3_schedule(n_steps)

    # --- Phase 3: denoise ------------------------------------------------------------
    transformer <- onload("transformer")
    fp8_manual <- isTRUE(pipeline$fp8_resident) &&
    is.null(staging[["transformer"]])
    if (fp8_manual) {
        .flux_fp8_to_device(transformer, device)
    }
    if (verbose) {
        message(sprintf("Denoising: %d steps at %dx%d...", n_steps, width,
                        height))
    }
    latents <- .flux3_denoise(transformer, latents, sched, prompt_embeds,
                              txt_ids, img_ids, compute_dtype,
                              cond_tokens = cond_tokens, cond_ids = cond_ids,
                              chunk_size = pipeline$attn_chunk,
                              verbose = level)
    rm(cond_tokens)
    if (fp8_manual && phase_offload) {
        .flux_fp8_to_device(pipeline$transformer, "cpu")
    }
    offload("transformer")
    ltx23_release_dequant_buffers()

    # --- Phase 4: decode ---------------------------------------------------------------
    if (verbose) {
        message("Decoding...")
    }
    # [1, h*w, 128] -> [1, 128, h, w]
    latents <- latents$reshape(c(1L, h2, w2, 128L))$permute(c(1L, 4L, 2L, 3L))
    latents <- flux2_bn_normalize(latents, pipeline$decoder$bn$running_mean,
                                  pipeline$decoder$bn$running_var,
                                  eps = pipeline$vae_bn_eps %||% 1e-4,
                                  inverse = TRUE)
    latents <- flux2_unpatchify_latents(latents)

    decoder <- onload("decoder")
    torch::with_no_grad({
        dec_param <- decoder$post_quant_conv$weight
        img <- decoder(latents$to(device = dec_param$device,
                                  dtype = dec_param$dtype))
        img <- img$to(dtype = f32)$cpu()
    })
    offload("decoder")

    img <- img$squeeze(1)$permute(c(2L, 3L, 1L))
    img <- img$add(1)$div(2)$clamp(0, 1)
    img_array <- as.array(img)
    if (!is.null(crop)) {
        img_array <- img_array[(crop[["top"]] + 1):crop[["bottom"]],
            (crop[["left"]] + 1):crop[["right"]],,
            drop = FALSE]
    }

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
                     prompt = prompt, width = dim(img_array)[2],
                     height = dim(img_array)[1], steps = n_steps, seed = seed,
                     model = "flux3-image-klein", precision = pipeline$format,
                     seconds = gen_seconds
    )
    invisible(list(image = img_array, metadata = metadata))
}
