#' AnimateDiff text-to-video on Stable Diffusion 1.5
#'
#' SD 1.5 (CreativeML OpenRAIL-M) with the AnimateDiff v2 or v3 motion
#' module (Apache-2.0, \code{guoyww/animatediff}). The pipeline follows
#' diffusers' \code{AnimateDiffPipeline}: DDIM with a linear beta schedule
#' and linspace timesteps, classifier-free guidance, frames decoded one
#' chunk at a time.
#'
#' @name animatediff
NULL

.sd15_repo <- "stable-diffusion-v1-5/stable-diffusion-v1-5"

# hub path -> name the native loaders read
.sd15_files <- c(
    "unet/config.json" = "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors" =
        "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json" = "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors" =
        "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json" = "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors" = "text_encoder/model.safetensors")

.animatediff_motion_files <- c(v3 = "v3_sd15_mm.ckpt", v2 = "mm_sd_v15_v2.ckpt")

.animatediff_dir <- function() {
    file.path(tools::R_user_dir("diffuseR", "data"), "animatediff")
}

#' Download Stable Diffusion 1.5
#'
#' Fetches the fp16 UNet, VAE and CLIP text encoder (about 2.1 GB) from
#' \code{stable-diffusion-v1-5/stable-diffusion-v1-5} and links them under
#' the file names the native loaders read.
#'
#' @param verbose Logical.
#'
#' @return Invisibly, the diffusers-layout directory.
#' @export
download_sd15 <- function(verbose = TRUE) {
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("The hfhub package is required to download model weights.")
    }
    if (!.hub_all_cached(.sd15_repo, names(.sd15_files)) &&
        !.ltx23_consent("Stable Diffusion 1.5 UNet + VAE + CLIP text encoder (~2.1 GB)")) {
        stop("Download cancelled.", call. = FALSE)
    }
    out <- file.path(.animatediff_dir(), "sd15")
    for (f in names(.sd15_files)) {
        src <- hfhub::hub_download(.sd15_repo, f)
        dest <- file.path(out, .sd15_files[[f]])
        dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
        if (!file.exists(dest) && !file.symlink(src, dest)) {
            file.copy(src, dest)
        }
    }
    if (verbose) {
        message("SD 1.5 ready: ", out)
    }
    invisible(out)
}

#' Download an AnimateDiff motion module
#'
#' Fetches the original Apache-2.0 checkpoint from
#' \code{guoyww/animatediff} and converts it once to an fp16 safetensors
#' file (the pickle is read by \code{torch::load_state_dict()}, which only
#' rebuilds tensors). Keys keep their original names;
#' \code{\link{load_motion_adapter}} maps them.
#'
#' @param version "v3" (\code{v3_sd15_mm.ckpt}, no mid-block module) or
#'   "v2" (\code{mm_sd_v15_v2.ckpt}).
#' @param verbose Logical.
#'
#' @return Invisibly, the path of the converted safetensors file.
#' @export
download_animatediff <- function(version = c("v3", "v2"), verbose = TRUE) {
    version <- match.arg(version)
    out <- file.path(.animatediff_dir(), paste0("motion-sd15-", version,
                                                ".safetensors"))
    if (file.exists(out)) {
        return(invisible(out))
    }
    ckpt_name <- .animatediff_motion_files[[version]]
    if (!.hub_all_cached("guoyww/animatediff", ckpt_name) &&
        !.ltx23_consent(sprintf("AnimateDiff %s motion module (~1.7 GB)", version))) {
        stop("Download cancelled.", call. = FALSE)
    }
    ckpt <- hfhub::hub_download("guoyww/animatediff", ckpt_name)
    if (verbose) {
        message("Converting ", ckpt_name, " to fp16 safetensors...")
    }
    .animatediff_convert(ckpt, out)
    invisible(out)
}

# Original motion checkpoint -> fp16 safetensors with the same keys.
.animatediff_convert <- function(ckpt, out) {
    sd <- torch::load_state_dict(ckpt)
    if (!is.null(sd$state_dict)) {
        sd <- sd$state_dict
    }
    sd <- lapply(sd, function(t) {
        if (t$is_floating_point()) t$to(dtype = torch::torch_float16()) else t
    })
    dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
    tmp <- paste0(out, ".part")
    safetensors::safe_save_file(sd, tmp)
    file.rename(tmp, out)
    invisible(out)
}

#' Load the AnimateDiff SD 1.5 pipeline
#'
#' @param motion Motion module: "v3", "v2", or a path to a motion
#'   safetensors file.
#' @param model_dir SD 1.5 diffusers directory; NULL uses
#'   \code{\link{download_sd15}}.
#' @param device Compute device for the UNet and VAE.
#' @param dtype UNet and motion module dtype; NULL means float16 on CUDA,
#'   float32 on CPU.
#' @param verbose Logical.
#'
#' @return A list with \code{unet} (a \code{\link{unet_motion}}),
#'   \code{decoder}, \code{text_encoder}, \code{device}, \code{dtype} and
#'   \code{max_frames}.
#' @export
animatediff_load_pipeline <- function(motion = "v3", model_dir = NULL,
                                      device = "cuda", dtype = NULL,
                                      verbose = TRUE) {
    if (is.null(model_dir)) {
        model_dir <- download_sd15(verbose = verbose)
    }
    path <- if (motion %in% names(.animatediff_motion_files)) {
        download_animatediff(motion, verbose = verbose)
    } else {
        motion
    }
    if (is.null(dtype)) {
        dtype <- if (identical(device, "cuda")) torch::torch_float16() else torch::torch_float32()
    }
    pipe <- sd_pipeline_from_safetensors(model_dir, model_name = "sd15",
        devices = list(unet = "cpu", decoder = "cpu", text_encoder = "cpu"),
        unet_dtype = torch::torch_float32(), verbose = verbose)
    adapter <- load_motion_adapter(path, verbose = verbose)
    unet <- unet_motion(pipe$unet, adapter)
    unet$to(device = device, dtype = dtype)
    pipe$decoder$to(device = device)
    list(unet = unet, decoder = pipe$decoder, text_encoder = pipe$text_encoder,
         device = device, dtype = dtype, max_frames = adapter$max_len)
}

# diffusers DDIMScheduler timesteps. "linspace" (AnimateDiff): 0..999
# evenly, rounded half to even, no offset. "leading" (SD 1.5's own
# config): k * 1000 %/% n + steps_offset.
.ddim_timesteps <- function(n, spacing = c("linspace", "leading"),
                            steps_offset = 1L, train_steps = 1000L) {
    spacing <- match.arg(spacing)
    t <- if (spacing == "linspace") {
        round(seq(0, train_steps - 1, length.out = n))
    } else {
        seq.int(0L, n - 1L) * (train_steps %/% n) + steps_offset
    }
    as.integer(rev(t))
}

.ddim_alphas_cumprod <- function(beta_schedule = c("linear", "scaled_linear"),
                                 beta_start = 0.00085, beta_end = 0.012,
                                 train_steps = 1000L) {
    betas <- switch(match.arg(beta_schedule),
                    linear = seq(beta_start, beta_end, length.out = train_steps),
                    scaled_linear = seq(sqrt(beta_start), sqrt(beta_end),
                                        length.out = train_steps) ^ 2)
    cumprod(1 - betas)
}

# One deterministic DDIM step (eta 0, epsilon prediction). The previous
# timestep is t - 1000 / n whatever the spacing, as in diffusers; past 0
# it uses alphas_cumprod[0] (set_alpha_to_one = FALSE).
.ddim_step <- function(sample, eps, t, n, alphas_cumprod) {
    a_t <- alphas_cumprod[t + 1L]
    prev <- t - length(alphas_cumprod) %/% n
    a_prev <- if (prev >= 0) alphas_cumprod[prev + 1L] else alphas_cumprod[1]
    x0 <- (sample - eps * sqrt(1 - a_t)) / sqrt(a_t)
    x0 * sqrt(a_prev) + eps * sqrt(1 - a_prev)
}

# Prompt schedule -> one embedding per frame [F, 77, D]. A single prompt
# is one row (the UNet repeats it). Keyframes are named by 1-based frame;
# frames between two keyframes blend their embeddings linearly, frames
# before the first or after the last hold it.
.animatediff_prompt_embeds <- function(prompt, num_frames, encode) {
    if (length(prompt) == 1L && is.null(names(prompt))) {
        return(encode(prompt))
    }
    frames <- as.integer(names(prompt))
    if (anyNA(frames) || any(frames < 1L | frames > num_frames)) {
        stop("Prompt keyframes must be named by frame number, 1 to ",
             num_frames, call. = FALSE)
    }
    o <- order(frames)
    frames <- frames[o]
    embeds <- lapply(unname(as.character(unlist(prompt)))[o], encode)
    rows <- lapply(seq_len(num_frames), function(f) {
        k <- findInterval(f, frames)
        if (k == 0L) {
            return(embeds[[1]])
        }
        if (k == length(frames)) {
            return(embeds[[k]])
        }
        w <- (f - frames[k]) / (frames[k + 1L] - frames[k])
        embeds[[k]] * (1 - w) + embeds[[k + 1L]] * w
    })
    torch::torch_cat(rows, dim = 1L)
}

#' Generate a video with AnimateDiff (SD 1.5)
#'
#' Text-to-video from pure noise. The image model draws every frame and the
#' motion module ties them together loosely, so objects drift and morph;
#' \code{prompt} keyframes push that further by changing the subject over
#' the clip.
#'
#' @param prompt A prompt, or keyframes: a character vector or list named
#'   by 1-based frame number, e.g. \code{c("1" = "a red fox", "16" = "a
#'   wolf made of smoke")}. Frames between keyframes blend the two prompt
#'   embeddings.
#' @param negative_prompt Negative prompt (one for the whole clip).
#' @param pipeline From \code{\link{animatediff_load_pipeline}}; NULL loads
#'   the default (v3, CUDA).
#' @param width Output width, a multiple of 8 (SD 1.5 is trained at 512).
#' @param height Output height, a multiple of 8.
#' @param num_frames Frames, up to the motion module's window (32).
#' @param num_inference_steps DDIM steps.
#' @param guidance_scale Classifier-free guidance. Higher values (10-15)
#'   push harder towards the prompt and can add artifacts.
#' @param seed Integer seed, or NULL.
#' @param latents Optional starting noise \code{[1, 4, F, H/8, W/8]}.
#' @param decode_chunk Frames per VAE decode.
#' @param fps Frame rate of the saved file.
#' @param filename Output video path; NULL with \code{save_file = TRUE}
#'   derives one from the prompt.
#' @param save_file Logical; write the video with \code{\link{save_video}}.
#' @param verbose Logical.
#'
#' @return A list: \code{video} (array \code{[F, H, W, 3]} in 0..1),
#'   \code{file} (or NULL) and \code{metadata}.
#'
#' @examples
#' \dontrun{
#' pipe <- animatediff_load_pipeline("v3")
#' txt2vid_animatediff("a lighthouse in a storm, oil painting",
#'                     pipeline = pipe, seed = 7, filename = "storm.mp4")
#' # prompt travel: the subject melts from one prompt into the next
#' txt2vid_animatediff(c("1" = "a red fox in snow", "16" = "a wolf made of smoke"),
#'                     pipeline = pipe, guidance_scale = 12, filename = "fox.mp4")
#' }
#' @export
txt2vid_animatediff <- function(prompt, negative_prompt = NULL, pipeline = NULL,
                                width = 512L, height = 512L, num_frames = 16L,
                                num_inference_steps = 25L, guidance_scale = 7.5,
                                seed = NULL, latents = NULL, decode_chunk = 4L,
                                fps = 8L, filename = NULL, save_file = TRUE,
                                verbose = TRUE) {
    if (is.null(pipeline)) {
        pipeline <- animatediff_load_pipeline(verbose = verbose)
    }
    if (width %% 8 != 0 || height %% 8 != 0) {
        stop("width and height must be multiples of 8", call. = FALSE)
    }
    if (num_frames > pipeline$max_frames) {
        stop(sprintf("num_frames is limited to the motion module's %d-frame window",
                     pipeline$max_frames), call. = FALSE)
    }
    device <- pipeline$device
    dtype <- pipeline$dtype
    start <- proc.time()[[3]]

    torch::with_no_grad({
        encode <- function(p) {
            pipeline$text_encoder(CLIPTokenizer(p, pad_token = 49407L))
        }
        cond <- .animatediff_prompt_embeds(prompt, num_frames, encode)
        uncond <- encode(if (is.null(negative_prompt)) "" else negative_prompt)
        cond <- cond$to(device = device, dtype = dtype)
        uncond <- uncond$to(device = device, dtype = dtype)

        if (is.null(latents)) {
            if (!is.null(seed)) {
                torch::torch_manual_seed(seed)
            }
            latents <- torch::torch_randn(c(1L, 4L, num_frames, height %/% 8L,
                                            width %/% 8L))
        }
        latents <- latents$to(device = device, dtype = torch::torch_float32())
        ac <- .ddim_alphas_cumprod("linear")
        steps <- .ddim_timesteps(num_inference_steps, "linspace")
        for (i in seq_along(steps)) {
            t <- torch::torch_tensor(steps[i], device = device)
            x <- latents$to(dtype = dtype)
            eps_u <- pipeline$unet(x, t, uncond)
            eps_c <- pipeline$unet(x, t, cond)
            eps <- (eps_u + (eps_c - eps_u) * guidance_scale)$to(dtype = torch::torch_float32())
            rm(eps_u, eps_c, x)
            latents <- .ddim_step(latents, eps, steps[i], num_inference_steps, ac)
            if (verbose) {
                message(sprintf("step %d/%d", i, length(steps)))
            }
        }
        video <- .animatediff_decode(pipeline$decoder, latents, decode_chunk)
    })

    file <- NULL
    if (save_file) {
        first <- if (is.list(prompt)) prompt[[1]] else prompt[1]
        file <- filename %||% sub("\\.png$", ".mp4", filename_from_prompt(first, datetime = TRUE))
        save_video(video, file, fps = fps, verbose = verbose)
    }
    elapsed <- proc.time()[[3]] - start
    if (verbose) {
        message(sprintf("Video generated in %.1f s", elapsed))
    }
    list(video = video, file = file,
         metadata = list(prompt = prompt, negative_prompt = negative_prompt,
                         width = width, height = height, num_frames = num_frames,
                         num_inference_steps = num_inference_steps,
                         guidance_scale = guidance_scale, seed = seed,
                         fps = fps, seconds = elapsed, model = "animatediff-sd15"))
}

# Latents [1, 4, F, h, w] -> R array [F, H, W, 3] in 0..1, a few frames
# per VAE call.
.animatediff_decode <- function(decoder, latents, chunk = 4L) {
    f <- as.integer(latents$shape[3])
    frames <- (latents / 0.18215)[1, , , , ]$permute(c(2L, 1L, 3L, 4L))
    out <- NULL
    for (s in seq.int(1L, f, by = chunk)) {
        idx <- s:min(f, s + chunk - 1L)
        x <- decoder(frames[idx, , , , drop = FALSE]$contiguous())
        x <- ((x / 2 + 0.5)$clamp(0, 1))$permute(c(1L, 3L, 4L, 2L))$cpu()
        a <- as.array(x)
        if (is.null(out)) {
            out <- array(0, dim = c(f, dim(a)[2:4]))
        }
        out[idx, , , ] <- a
        rm(x)
    }
    out
}
