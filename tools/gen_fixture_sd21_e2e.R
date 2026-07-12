# Generate the SD 2.1 end-to-end parity fixture (torch reference). Loads
# the real F16 diffusers checkpoints into the native SD 2.1 pipeline
# (sd_pipeline_from_safetensors, all CPU/f32 - the same wiring
# txt2img_sd21 uses on the native-safetensors path), then runs the full
# text-to-image loop on a FIXED prompt, seed, and a small step count at
# low resolution (256px -> 32x32 latent): tokenize cond + uncond -> CLIP
# encode -> v-prediction DDIM denoise with classifier-free guidance
# (scale 7.5) -> VAE decode. Saves the token ids, the initial noise,
# step-1 UNet outputs (to isolate CFG/scheduler wiring from the UNet),
# the post-step-1 latents, the final latents, and the decoded pixels.
# The anvl test reloads the same checkpoints and feeds these ids + noise.
#
# Usage: /home/troy/diffuseR-anvl-lib/ranvl tools/gen_fixture_sd21_e2e.R

suppressMessages(library(torch))
suppressMessages(library(diffuseR))

diffusers_dir <- file.path(Sys.getenv("HOME"),
                           ".local/share/R/diffuseR/sd21-diffusers")
stopifnot(dir.exists(file.path(diffusers_dir, "unet")),
          dir.exists(file.path(diffusers_dir, "vae")),
          dir.exists(file.path(diffusers_dir, "text_encoder")))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "sd21_e2e.safetensors")

PROMPT <- "a red panda astronaut floating in space, photorealistic"
NEG <- ""
N_STEPS <- 6L
GUIDANCE <- 7.5
LATENT <- 32L                       # 256px image
SEED <- 21L

devices <- list(unet = "cpu", decoder = "cpu", text_encoder = "cpu")
pipe <- sd_pipeline_from_safetensors(diffusers_dir, model_name = "sd21",
                                     devices = devices,
                                     unet_dtype = torch_float32(),
                                     verbose = TRUE)

# ---- tokenize (raw CLIP ids, [1, 77]) ----
cond_tokens <- CLIPTokenizer(PROMPT)
uncond_tokens <- CLIPTokenizer(NEG)
cond_embed <- pipe$text_encoder(cond_tokens)      # [1, 77, 1024], f32
uncond_embed <- pipe$text_encoder(uncond_tokens)

# ---- initial noise ----
set.seed(SEED)
torch_manual_seed(SEED)
latents0 <- torch_randn(c(1L, 4L, LATENT, LATENT), dtype = torch_float32())

# ---- DDIM schedule (scaled-linear beta, v-prediction) ----
schedule <- ddim_scheduler_create(num_inference_steps = N_STEPS,
                                  beta_schedule = "scaled_linear",
                                  device = torch_device("cpu"))
timesteps <- schedule$timesteps
cat("timesteps:", paste(timesteps, collapse = ", "), "\n")

# ---- CFG DDIM denoise loop (mirrors txt2img_sd21) ----
latents <- latents0
nu1 <- NULL; nc1 <- NULL; lat1 <- NULL
with_no_grad({
    for (i in seq_along(timesteps)) {
        timestep <- torch_tensor(timesteps[i], dtype = torch_long())
        noise_uncond <- pipe$unet(latents, timestep, uncond_embed)
        noise_cond <- pipe$unet(latents, timestep, cond_embed)
        noise_pred <- noise_uncond + GUIDANCE * (noise_cond - noise_uncond)
        latents <- ddim_scheduler_step(model_output = noise_pred,
                                       timestep = timestep, sample = latents,
                                       schedule = schedule,
                                       prediction_type = "v_prediction",
                                       device = "cpu")
        if (i == 1L) {
            nu1 <- noise_uncond; nc1 <- noise_cond; lat1 <- latents
        }
    }
})
final <- latents

# ---- VAE decode (raw decoder output, [-1, 1]) ----
scaled_latent <- final / 0.18215
pixels <- with_no_grad(pipe$decoder(scaled_latent))   # [1, 3, 256, 256]

safetensors::safe_save_file(list(
    cond_ids = torch_tensor(cond_tokens$to(dtype = torch_float32())),
    uncond_ids = torch_tensor(uncond_tokens$to(dtype = torch_float32())),
    latents0 = latents0$contiguous(),
    cond_embed = cond_embed$contiguous(),
    uncond_embed = uncond_embed$contiguous(),
    noise_uncond_1 = nu1$contiguous(),
    noise_cond_1 = nc1$contiguous(),
    latents_1 = lat1$contiguous(),
    final = final$contiguous(),
    pixels = pixels$contiguous()
), fixture)

cat(sprintf("fixture: %s (%.2f MB)\n", fixture, file.size(fixture) / 1e6))
cat(sprintf("final latents sd %.4f range [%.3f, %.3f]\n",
            final$std()$item(), final$min()$item(), final$max()$item()))
cat(sprintf("pixels sd %.4f range [%.3f, %.3f]\n",
            pixels$std()$item(), pixels$min()$item(), pixels$max()$item()))
