# End-to-end parity: the full anvl SD 2.1 text-to-image pipeline
# (CLIP encode cond + uncond -> v-prediction DDIM denoise with CFG over
# the UNet -> VAE decode) vs the torch reference from
# tools/gen_fixture_sd21_e2e.R, on real F16 weights. Feeds the SAME token
# ids and initial noise as the fixture (never chase RNG). Small (32x32
# latent, 256px), so it runs on CPU. Loads all three components.
#
# Two gates:
#  1. STEP 1 (tight, cor 1.0000000): isolates the newly-written CFG +
#     DDIM-step wiring. Before the trajectory has had steps to chaotically
#     amplify f32 rounding, anvl and torch agree to f32 (< 1e-3 * scale).
#  2. FULL 6-step trajectory (realistic): SD 2.1 CFG (guidance 7.5) is a
#     chaotic ODE - two independent f32 implementations (anvl XLA vs torch)
#     diverge in max-abs at a few worst-case pixels while staying highly
#     correlated. Deterministic (no RNG), so the numbers are reproducible;
#     the tolerance reflects that amplification, not a wiring error.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
base <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/sd21-diffusers")
unet_ckpt <- file.path(base, "unet", "diffusion_pytorch_model.safetensors")
vae_ckpt <- file.path(base, "vae", "diffusion_pytorch_model.safetensors")
te_dir <- file.path(base, "text_encoder")
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/diffuseR/fixtures/sd21_e2e.safetensors")
if (!file.exists(unet_ckpt) || !file.exists(vae_ckpt) || !dir.exists(te_dir) ||
    !file.exists(fixture)) {
    exit_file("checkpoint or fixture missing; run tools/gen_fixture_sd21_e2e.R")
}

N_STEPS <- 6L
GUIDANCE <- 7.5
LATENT <- 32L

f <- anvl::nv_read(fixture)
cond_ids <- matrix(as.integer(round(as.array(f$cond_ids))), nrow = 1L)
uncond_ids <- matrix(as.integer(round(as.array(f$uncond_ids))), nrow = 1L)
S <- ncol(cond_ids)

w_clip <- yq_clip_load_weights(te_dir, device = "cpu")
w_unet <- yq_sd_unet_load_weights(unet_ckpt, device = "cpu")
w_vae <- yq_sd_vae_load_weights(vae_ckpt, device = "cpu")

# jit once, reuse for the step-1 gate and the full run
clip_fn <- anvl::jit(yq_clip_encoder(apply_final_ln = TRUE, precision = "highest"))
unet_fn <- anvl::jit(yq_sd_unet())
sched <- yq_sd21_ddim_sigmas(N_STEPS)

# ---- host-side DDIM schedule matches the reference timesteps ----
step_ratio <- 1000L %/% N_STEPS
want_ts <- as.integer(rev(round((0:(N_STEPS - 1L)) * step_ratio) + 1L))
expect_equal(sched$timesteps, want_ts)

# ---- GATE 1: step 1 (tight; the new CFG + DDIM wiring) ----
mask <- yq_clip_mask(S, batch = 1L, device = "cpu")
ce <- clip_fn(yq_clip_embed(w_clip$token_embedding, w_clip$position_embedding,
                            cond_ids, "cpu"), mask, w_clip)
ue <- clip_fn(yq_clip_embed(w_clip$token_embedding, w_clip$position_embedding,
                            uncond_ids, "cpu"), mask, w_clip)
t1 <- yq_sd_time_embed(sched$timesteps[1], dim = 320L, device = "cpu")
nu <- unet_fn(f$latents0, t1, ue, w_unet)
nc <- unet_fn(f$latents0, t1, ce, w_unet)
np <- nu + (nc - nu) * anvl::nv_scalar(GUIDANCE, "f32", device = "cpu")
lat1 <- as.array(yq_sd21_ddim_step(f$latents0, np, sched$coeff[[1]],
                                   device = "cpu"))
w1 <- as.array(f$latents_1); s1 <- max(abs(w1))
cat(sprintf("sd21 e2e step-1 latents: max %.3e mean %.3e cor %.7f (scale %.3f)\n",
            max(abs(lat1 - w1)), mean(abs(lat1 - w1)),
            cor(as.vector(lat1), as.vector(w1)), s1))
expect_true(cor(as.vector(lat1), as.vector(w1)) > 0.99999)
expect_true(max(abs(lat1 - w1)) < 1e-3 * s1)
expect_true(mean(abs(lat1 - w1)) < 1e-4 * s1)

# ---- GATE 2: full 6-step pipeline (final latents + decoded pixels) ----
res <- yq_sd21_generate(w_clip = w_clip, w_unet = w_unet, w_vae = w_vae,
                        ids = cond_ids, uncond_ids = uncond_ids,
                        noise = f$latents0, latent_dim = LATENT,
                        num_inference_steps = N_STEPS, guidance_scale = GUIDANCE,
                        decode = TRUE, device = "cpu",
                        clip_fn = clip_fn, unet_fn = unet_fn)

got <- as.array(res$latents); want <- as.array(f$final)
max_abs <- max(abs(got - want)); scale <- max(abs(want))
correlation <- cor(as.vector(got), as.vector(want))
cat(sprintf("sd21 e2e final latents: max %.3e mean %.3e cor %.7f (scale %.3f)\n",
            max_abs, mean(abs(got - want)), correlation, scale))
expect_equal(dim(got), c(1L, 4L, LATENT, LATENT))
expect_true(correlation > 0.9999)               # chaotic CFG trajectory
expect_true(max_abs < 0.05 * scale)
expect_true(mean(abs(got - want)) < 1e-3 * scale)

pgot <- as.array(res$pixels); pwant <- as.array(f$pixels)
pmax <- max(abs(pgot - pwant)); pscale <- max(abs(pwant))
pcor <- cor(as.vector(pgot), as.vector(pwant))
cat(sprintf("sd21 e2e final pixels:  max %.3e mean %.3e cor %.7f (scale %.3f)\n",
            pmax, mean(abs(pgot - pwant)), pcor, pscale))
expect_equal(dim(pgot), c(1L, 3L, 8L * LATENT, 8L * LATENT))
expect_true(pcor > 0.9999)
expect_true(pmax < 0.05 * pscale)
