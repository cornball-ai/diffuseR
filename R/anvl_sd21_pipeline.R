#' Stable Diffusion 2.1 end-to-end text-to-image pipeline (anvl)
#'
#' Wires the three already-ported anvl components into the full SD 2.1
#' text-to-image path: the CLIP text encoder (\code{\link{yq_clip_encoder}},
#' conditional + unconditional), a host-side DDIM scheduler
#' (\code{\link{yq_sd21_ddim_sigmas}} / \code{\link{yq_sd21_ddim_step}}),
#' the UNet (\code{\link{yq_sd_unet}}) run under classifier-free guidance,
#' and the VAE decoder (\code{\link{yq_sd_vae_decode}}). Mirrors
#' \code{diffuseR::txt2img_sd21} on the native-safetensors path
#' (\code{sd_pipeline_from_safetensors}): scaled-linear beta schedule,
#' v-prediction DDIM (eta 0), CFG combine
#' \code{uncond + scale * (cond - uncond)}, latent init scale 1.0, and the
#' 0.18215 VAE latent rescale before decode.
#'
#' Only the UNet and CLIP forwards are anvl (jit-compiled once, called per
#' step); the DDIM alpha schedule and per-step coefficients are plain R
#' scalars, exactly as the FlowMatch sigmas were for the FLUX ports.
#'
#' @name anvl_sd21_pipeline
NULL

#' Host-side DDIM schedule for SD 2.1 (alpha coefficients per step)
#'
#' Ports \code{diffuseR::ddim_scheduler_create} +
#' \code{ddim_scheduler_step} (v-prediction, eta 0) to plain R scalars.
#' Builds the scaled-linear beta schedule, the cumulative-alpha product,
#' the reversed inference timesteps, and, for each step, the four
#' square-root coefficients the v-prediction DDIM update needs:
#' \code{a_t = sqrt(alpha_prod_t)}, \code{b_t = sqrt(1 - alpha_prod_t)},
#' \code{a_prev = sqrt(alpha_prod_t_prev)},
#' \code{d_prev = sqrt(1 - alpha_prod_t_prev)}. The timestep -> alpha-index
#' convention (\code{index = timestep + 1}, the final step falling back to
#' \code{alpha_prod_t_prev = alphas_cumprod[1]}) mirrors the reference
#' exactly. Computed in f64 (more accurate than the reference's f32
#' \code{torch_cumprod}; the ~1e-6 relative gap is far below the f32
#' end-to-end tolerance).
#'
#' @param num_inference_steps Integer. Denoising steps.
#' @param num_train_timesteps Integer. Training timesteps (1000).
#' @param beta_start,beta_end Numeric. Scaled-linear beta endpoints.
#'
#' @return List with \code{timesteps} (integer vector, the UNet timesteps
#'   per step), \code{alphas_cumprod} (length \code{num_train_timesteps}),
#'   and \code{coeff} (a list of per-step \code{a_t/b_t/a_prev/d_prev}).
#'
#' @export
yq_sd21_ddim_sigmas <- function(num_inference_steps, num_train_timesteps = 1000L,
                                beta_start = 0.00085, beta_end = 0.012) {
    num_inference_steps <- as.integer(num_inference_steps)
    betas <- seq(sqrt(beta_start), sqrt(beta_end),
                 length.out = num_train_timesteps)^2
    alphas <- 1 - betas
    alphas_cumprod <- cumprod(alphas)

    step_ratio <- num_train_timesteps %/% num_inference_steps
    timesteps <- as.integer(rev(round((0:(num_inference_steps - 1L)) *
                                      step_ratio) + 1L))

    coeff <- lapply(seq_len(num_inference_steps), function(i) {
        t <- timesteps[i]
        timestep_index <- t + 1L
        if (timestep_index <= 2L) {
            prev_timestep_index <- 1L
        } else {
            # timesteps strictly decreasing distinct, so which() == i
            prev_timestep <- timesteps[which(timesteps == t) + 1L]
            prev_timestep_index <- prev_timestep + 1L
        }
        apt <- alphas_cumprod[timestep_index]
        apt_prev <- alphas_cumprod[prev_timestep_index]
        list(a_t = sqrt(apt), b_t = sqrt(1 - apt),
             a_prev = sqrt(apt_prev), d_prev = sqrt(1 - apt_prev))
    })

    list(timesteps = timesteps, alphas_cumprod = alphas_cumprod, coeff = coeff)
}

#' One v-prediction DDIM step (anvl)
#'
#' Applies a single deterministic (eta 0) v-prediction DDIM update using
#' the host-side coefficients from \code{\link{yq_sd21_ddim_sigmas}}:
#' \deqn{x_0 = a_t \cdot sample - b_t \cdot v}
#' \deqn{\epsilon = a_t \cdot v + b_t \cdot sample}
#' \deqn{prev = a_{prev} \cdot x_0 + d_{prev} \cdot \epsilon}
#' All four coefficients enter as f32 scalars (auto-broadcast), so the
#' latents never leave the device.
#'
#' @param sample AnvlArray \code{[B, 4, H, W]} current noisy latents.
#' @param model_output AnvlArray \code{[B, 4, H, W]} predicted velocity
#'   (post-CFG).
#' @param coeff One element of \code{yq_sd21_ddim_sigmas()$coeff}.
#' @param device Character. Device for the scalar coefficients.
#'
#' @return AnvlArray \code{[B, 4, H, W]} previous (less noisy) sample.
#'
#' @export
yq_sd21_ddim_step <- function(sample, model_output, coeff, device = "cpu") {
    a_t <- anvl::nv_scalar(coeff$a_t, "f32", device = device)
    b_t <- anvl::nv_scalar(coeff$b_t, "f32", device = device)
    a_prev <- anvl::nv_scalar(coeff$a_prev, "f32", device = device)
    d_prev <- anvl::nv_scalar(coeff$d_prev, "f32", device = device)
    x0 <- sample * a_t - model_output * b_t
    eps <- model_output * a_t + sample * b_t
    x0 * a_prev + eps * d_prev
}

# Tokenize a prompt to a 0-based CLIP id matrix [1, 77] via the diffuseR
# CLIP tokenizer (host-side, torch-backed). Only used when the caller
# passes a prompt rather than explicit ids.
.yq_sd21_tokenize <- function(prompt) {
    matrix(as.integer(as.array(CLIPTokenizer(prompt))), nrow = 1L)
}

#' Stable Diffusion 2.1 end-to-end generation (anvl)
#'
#' Runs the full text-to-image pipeline: tokenize (or take explicit ids)
#' -> CLIP encode conditional + unconditional prompts -> CFG DDIM denoise
#' loop over the UNet -> (optionally) VAE decode to pixels. The CLIP
#' encoder and UNet are each \code{anvl::jit()}ed once and reused across
#' steps; the UNet is called twice per step (unconditional, then
#' conditional) and combined with classifier-free guidance
#' \code{uncond + guidance_scale * (cond - uncond)}. Feed \code{noise}
#' (and \code{ids} / \code{uncond_ids}) for bit-for-bit parity with a torch
#' reference; the RNG fallback does not match torch.
#'
#' @param prompt Character. Prompt (ignored if \code{ids} given).
#' @param w_clip CLIP weights pytree (\code{\link{yq_clip_load_weights}}).
#' @param w_unet UNet weights pytree (\code{\link{yq_sd_unet_load_weights}}).
#' @param w_vae Optional VAE weights pytree
#'   (\code{\link{yq_sd_vae_load_weights}}); required to decode pixels.
#' @param negative_prompt Character. Unconditional prompt (default "").
#' @param ids,uncond_ids Optional integer matrices \code{[1, S]} of 0-based
#'   CLIP ids overriding tokenization (for parity fixtures).
#' @param noise Optional AnvlArray \code{[1, 4, latent_dim, latent_dim]}
#'   initial latents; supply the reference's noise for parity.
#' @param latent_dim Integer. Latent H/W (image px / 8).
#' @param num_inference_steps Integer. DDIM steps.
#' @param guidance_scale Numeric. CFG scale (SD 2.1 default 7.5).
#' @param seed Optional integer seed for the anvl RNG fallback.
#' @param decode Logical. Decode latents to pixels (needs \code{w_vae}).
#' @param device Character. Target device.
#' @param precision Character. Matmul precision for CLIP.
#' @param clip_fn,unet_fn Optional pre-jitted closures (built from
#'   \code{\link{yq_clip_encoder}} / \code{\link{yq_sd_unet}} if NULL).
#'
#' @return List with \code{latents} (final AnvlArray \code{[1, 4, H, W]}),
#'   \code{pixels} (AnvlArray \code{[1, 3, 8H, 8W]} in [-1, 1] or NULL),
#'   and the \code{cond_embed} / \code{uncond_embed} CLIP states.
#'
#' @export
yq_sd21_generate <- function(prompt = NULL, w_clip, w_unet, w_vae = NULL,
                             negative_prompt = "", ids = NULL,
                             uncond_ids = NULL, noise = NULL, latent_dim = 96L,
                             num_inference_steps = 50L, guidance_scale = 7.5,
                             seed = NULL, decode = !is.null(w_vae),
                             device = "cpu", precision = "highest",
                             clip_fn = NULL, unet_fn = NULL) {
    if (is.null(ids)) {
        if (is.null(prompt)) stop("Provide `prompt` or `ids`.")
        ids <- .yq_sd21_tokenize(prompt)
    }
    if (is.null(uncond_ids)) {
        uncond_ids <- .yq_sd21_tokenize(negative_prompt)
    }
    ids <- matrix(as.integer(ids), nrow = 1L)
    uncond_ids <- matrix(as.integer(uncond_ids), nrow = 1L)
    S <- ncol(ids)

    if (is.null(clip_fn)) {
        clip_fn <- anvl::jit(yq_clip_encoder(apply_final_ln = TRUE,
                                             precision = precision))
    }
    mask <- yq_clip_mask(S, batch = 1L, device = device)
    embed <- function(id_mat) {
        clip_fn(yq_clip_embed(w_clip$token_embedding, w_clip$position_embedding,
                              id_mat, device = device), mask, w_clip)
    }
    cond_embed <- embed(ids)
    uncond_embed <- embed(uncond_ids)

    if (is.null(noise)) {
        st <- anvl::nv_rng_state(if (is.null(seed)) 0L else as.integer(seed))
        latents <- anvl::nv_rnorm(shape = c(1L, 4L, latent_dim, latent_dim),
                                  initial_state = st)[[2L]]
    } else {
        latents <- noise
    }

    sched <- yq_sd21_ddim_sigmas(num_inference_steps)
    if (is.null(unet_fn)) {
        unet_fn <- anvl::jit(yq_sd_unet())
    }
    gs <- anvl::nv_scalar(guidance_scale, "f32", device = device)
    for (i in seq_along(sched$timesteps)) {
        t_sin <- yq_sd_time_embed(sched$timesteps[i], dim = 320L,
                                  device = device)
        noise_uncond <- unet_fn(latents, t_sin, uncond_embed, w_unet)
        noise_cond <- unet_fn(latents, t_sin, cond_embed, w_unet)
        noise_pred <- noise_uncond + (noise_cond - noise_uncond) * gs
        latents <- yq_sd21_ddim_step(latents, noise_pred, sched$coeff[[i]],
                                     device = device)
    }

    pixels <- NULL
    if (decode && !is.null(w_vae)) {
        pixels <- yq_sd_vae_decode(yq_sd_vae_prepare(latents), w_vae)
    }
    list(latents = latents, pixels = pixels,
         cond_embed = cond_embed, uncond_embed = uncond_embed)
}
