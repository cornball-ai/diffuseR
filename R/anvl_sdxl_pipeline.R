#' Stable Diffusion XL end-to-end text-to-image pipeline (anvl)
#'
#' Wires the ported anvl SDXL components into the full text-to-image path:
#' the dual CLIP text encoders (\code{\link{yq_sdxl_clip_encoders}}) for the
#' concatenated penultimate context and the bigG pooled embed, a host-side
#' DDIM scheduler (\code{\link{yq_sdxl_ddim_schedule}} /
#' \code{\link{yq_sdxl_ddim_step}}), the UNet (\code{\link{yq_sdxl_unet}})
#' run under classifier-free guidance with the SDXL added text-time
#' conditioning (pooled text embeds + sinusoidal time-ids), and the VAE
#' decoder (\code{\link{yq_sdxl_vae_decode}}). Mirrors
#' \code{diffuseR::txt2img_sdxl}: scaled-linear beta schedule, epsilon
#' DDIM (eta 0), CFG combine \code{uncond + scale * (cond - uncond)}, and
#' the \code{negative_prompt = NULL} convention of zeroed unconditional
#' embeddings.
#'
#' \strong{Conditioning provenance.} The context is the concatenated
#' \emph{penultimate} CLIP-L + bigG hidden states (diffusers SDXL
#' clip-skip = None, \code{hidden_states[-2]}), matching the anvl encoder
#' design and the \code{anvl_test_sdxl_clip} parity fixture, not the
#' simplified full-forward call in \code{txt2img_sdxl}.
#'
#' Only the UNet and CLIP forwards are anvl (jit-compiled once, the UNet
#' called twice per step); the DDIM alpha schedule and per-step
#' coefficients are plain R scalars, exactly as the SD 2.1 pipeline did.
#'
#' @name anvl_sdxl_pipeline
NULL

#' Host-side DDIM schedule for SDXL (alpha coefficients per step)
#'
#' Ports \code{diffuseR::ddim_scheduler_create} +
#' \code{ddim_scheduler_step} (epsilon prediction, eta 0) to plain R
#' scalars. Builds the scaled-linear beta schedule, the cumulative-alpha
#' product, the reversed inference timesteps, and, for each step, the four
#' square-root coefficients the epsilon DDIM update needs:
#' \code{a_t = sqrt(alpha_prod_t)}, \code{b_t = sqrt(1 - alpha_prod_t)},
#' \code{a_prev = sqrt(alpha_prod_t_prev)},
#' \code{d_prev = sqrt(1 - alpha_prod_t_prev)}. The timestep -> alpha-index
#' convention (\code{index = timestep + 1}, the final step falling back to
#' \code{alpha_prod_t_prev = alphas_cumprod[1]}) mirrors the reference
#' exactly. Computed in f64 (more accurate than the reference's f32
#' \code{torch_cumprod}; the ~1e-6 relative gap is far below the f32
#' end-to-end tolerance). The coefficients are prediction-type agnostic;
#' only \code{\link{yq_sdxl_ddim_step}}'s formula is epsilon-specific.
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
yq_sdxl_ddim_schedule <- function(num_inference_steps, num_train_timesteps = 1000L,
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

#' One epsilon-prediction DDIM step (anvl)
#'
#' Applies a single deterministic (eta 0) epsilon DDIM update using the
#' host-side coefficients from \code{\link{yq_sdxl_ddim_schedule}}:
#' \deqn{x_0 = (sample - b_t \cdot \epsilon) / a_t}
#' \deqn{prev = a_{prev} \cdot x_0 + d_{prev} \cdot \epsilon}
#' where \eqn{\epsilon} is the (post-CFG) predicted noise. The reciprocal
#' \code{1 / a_t} is precomputed host-side so the update is scalar-broadcast
#' multiplies and the latents never leave the device.
#'
#' @param sample AnvlArray \code{[B, 4, H, W]} current noisy latents.
#' @param model_output AnvlArray \code{[B, 4, H, W]} predicted noise
#'   (post-CFG).
#' @param coeff One element of \code{yq_sdxl_ddim_schedule()$coeff}.
#' @param device Character. Device for the scalar coefficients.
#'
#' @return AnvlArray \code{[B, 4, H, W]} previous (less noisy) sample.
#'
#' @export
yq_sdxl_ddim_step <- function(sample, model_output, coeff, device = "cpu") {
    b_t <- anvl::nv_scalar(coeff$b_t, "f32", device = device)
    inv_a_t <- anvl::nv_scalar(1 / coeff$a_t, "f32", device = device)
    a_prev <- anvl::nv_scalar(coeff$a_prev, "f32", device = device)
    d_prev <- anvl::nv_scalar(coeff$d_prev, "f32", device = device)
    x0 <- (sample - model_output * b_t) * inv_a_t
    x0 * a_prev + model_output * d_prev
}

# Tokenize a prompt to a 0-based CLIP id matrix [1, 77] via the diffuseR
# CLIP tokenizer (host-side, torch-backed). Only used when the caller
# passes a prompt rather than explicit ids.
.yq_sdxl_tokenize <- function(prompt) {
    matrix(as.integer(as.array(CLIPTokenizer(prompt))), nrow = 1L)
}

#' Stable Diffusion XL end-to-end generation (anvl)
#'
#' Runs the full text-to-image pipeline: tokenize (or take explicit ids)
#' -> dual CLIP encode (concatenated penultimate context + bigG pooled) ->
#' build the SDXL added text-time conditioning (pooled + sinusoidal
#' time-ids) -> CFG DDIM denoise loop over the UNet -> (optionally) VAE
#' decode to pixels. The dual CLIP encoders and the UNet are each
#' \code{anvl::jit()}ed once and reused; the UNet is called twice per step
#' (conditional, then unconditional) and combined with classifier-free
#' guidance \code{uncond + guidance_scale * (cond - uncond)}. The
#' unconditional pass uses zeroed context and pooled embeds (the
#' \code{negative_prompt = NULL} path of \code{txt2img_sdxl}). Feed
#' \code{noise} (and \code{ids}) for bit-for-bit parity with a torch
#' reference.
#'
#' @param w_unet UNet weights pytree (\code{\link{yq_sdxl_unet_load_weights}}).
#' @param w_clipl,w_bigg CLIP-L / bigG weights pytrees
#'   (\code{\link{yq_sdxl_clip_load_weights}}).
#' @param w_vae Optional VAE weights pytree
#'   (\code{\link{yq_sdxl_vae_load_weights}}); required to decode pixels.
#' @param prompt Character. Prompt (ignored if \code{ids} given).
#' @param ids Optional integer matrix \code{[1, S]} of 0-based CLIP ids
#'   overriding tokenization (for parity fixtures). Shared by both encoders.
#' @param eos_index Optional integer. 1-based EOS position for the pooled
#'   gather (default \code{which.max(ids)}).
#' @param noise Optional AnvlArray \code{[1, 4, latent_dim, latent_dim]}
#'   initial latents; supply the reference's noise for parity.
#' @param img_dim Integer. Image side in pixels (also the SDXL time-ids
#'   original/target size).
#' @param latent_dim Integer. Latent H/W (default \code{img_dim / 8}).
#' @param time_ids Optional numeric length-6 SDXL micro-conditioning
#'   \code{c(orig_h, orig_w, crop_top, crop_left, target_h, target_w)}
#'   (default \code{c(img_dim, img_dim, 0, 0, img_dim, img_dim)}).
#' @param num_inference_steps Integer. DDIM steps.
#' @param guidance_scale Numeric. CFG scale (SDXL default 7.5).
#' @param decode Logical. Decode latents to pixels (needs \code{w_vae}).
#' @param device Character. Target device.
#' @param precision Character. Matmul precision for CLIP.
#' @param unet_fn Optional pre-jitted UNet closure (built from
#'   \code{\link{yq_sdxl_unet}} if NULL).
#'
#' @return List with \code{latents} (final AnvlArray \code{[1, 4, H, W]}),
#'   \code{pixels} (AnvlArray \code{[1, 3, 8H, 8W]} in [-1, 1] or NULL),
#'   the \code{context} / \code{pooled} CLIP conditioning, and \code{step1}
#'   (the step-1 \code{noise_cond} / \code{noise_uncond} / \code{latents},
#'   for isolating the CFG + scheduler wiring in parity tests).
#'
#' @export
yq_sdxl_generate <- function(w_unet, w_clipl, w_bigg, w_vae = NULL,
                             prompt = NULL, ids = NULL, eos_index = NULL,
                             noise = NULL, img_dim = 128L, latent_dim = NULL,
                             time_ids = NULL, num_inference_steps = 4L,
                             guidance_scale = 7.5, decode = !is.null(w_vae),
                             device = "cpu", precision = "highest",
                             unet_fn = NULL) {
    if (is.null(ids)) {
        if (is.null(prompt)) stop("Provide `prompt` or `ids`.")
        ids <- .yq_sdxl_tokenize(prompt)
    }
    ids <- matrix(as.integer(ids), nrow = 1L)
    S <- ncol(ids)
    if (is.null(eos_index)) eos_index <- which.max(ids[1L, ])
    if (is.null(latent_dim)) latent_dim <- as.integer(img_dim / 8L)

    # ---- dual CLIP: concatenated penultimate context + bigG pooled ----
    mask <- yq_sdxl_clip_mask(S, batch = 1L, device = device)
    clipl_embeds <- yq_sdxl_clip_embed(w_clipl$token_embedding,
                                       w_clipl$position_embedding, ids,
                                       device = device)
    bigg_embeds <- yq_sdxl_clip_embed(w_bigg$token_embedding,
                                      w_bigg$position_embedding, ids,
                                      device = device)
    enc <- yq_sdxl_clip_encoders(clipl_embeds, bigg_embeds, mask, eos_index,
                                 w_clipl, w_bigg, precision = precision,
                                 jit = TRUE)
    context <- enc$context                        # [1, S, 2048]
    pooled <- enc$pooled                          # [1, 1280]

    # ---- unconditional embeddings = zeros (negative_prompt = NULL path) ----
    zero <- anvl::nv_scalar(0, "f32", device = device)
    uncond_context <- context * zero
    uncond_pooled <- pooled * zero

    # ---- SDXL added text-time conditioning (sinusoidal time-ids) ----
    if (is.null(time_ids)) {
        time_ids <- c(img_dim, img_dim, 0, 0, img_dim, img_dim)
    }
    time_ids_sin <- yq_sdxl_time_ids_embed(time_ids, dim = 256L, device = device)

    if (is.null(noise)) stop("Provide `noise` for a deterministic run.")
    latents <- noise

    if (is.null(unet_fn)) unet_fn <- anvl::jit(yq_sdxl_unet())
    gs <- anvl::nv_scalar(guidance_scale, "f32", device = device)
    sched <- yq_sdxl_ddim_schedule(num_inference_steps)

    step1 <- NULL
    for (i in seq_along(sched$timesteps)) {
        t_sin <- yq_sdxl_time_embed(sched$timesteps[i], dim = 320L,
                                    device = device)
        noise_cond <- unet_fn(latents, t_sin, time_ids_sin, pooled, context,
                              w_unet)
        noise_uncond <- unet_fn(latents, t_sin, time_ids_sin, uncond_pooled,
                                uncond_context, w_unet)
        noise_pred <- noise_uncond + (noise_cond - noise_uncond) * gs
        latents <- yq_sdxl_ddim_step(latents, noise_pred, sched$coeff[[i]],
                                     device = device)
        if (i == 1L) {
            step1 <- list(noise_cond = noise_cond, noise_uncond = noise_uncond,
                          latents = latents)
        }
    }

    pixels <- NULL
    if (decode && !is.null(w_vae)) {
        pixels <- yq_sdxl_vae_decode(yq_sdxl_vae_prepare(latents), w_vae)
    }
    list(latents = latents, pixels = pixels, context = context,
         pooled = pooled, step1 = step1)
}
