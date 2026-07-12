# End-to-end parity: the full anvl SDXL text-to-image pipeline (dual CLIP
# encode -> SDXL added text-time conditioning -> epsilon DDIM denoise with
# CFG over the UNet -> VAE decode) vs the torch reference from
# tools/gen_fixture_sdxl_e2e.R, on real F16 weights. Feeds the SAME token
# ids and initial noise as the fixture (never chase RNG). Small (16x16
# latent, 128px, 4 steps), so it runs on CPU.
#
# Gates:
#  1. STEP-1 UNet outputs (cond + uncond): the UNet is already green, so
#     this confirms the real-weight loads + conditioning wiring feed it the
#     same inputs as the torch reference.
#  2. STEP-1 latents (tight): isolates the CFG + epsilon-DDIM-step wiring
#     before the trajectory amplifies f32 rounding.
#  3. FULL 4-step trajectory: final latents + decoded pixels. Deterministic
#     (eta 0), so reproducible; a chaotic CFG ODE lets two independent f32
#     implementations (anvl XLA vs torch) diverge slightly in max-abs while
#     staying essentially perfectly correlated.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
io_file <- file.path(fixture_dir, "sdxl_e2e_io.safetensors")
clipl_file <- file.path(fixture_dir, "sdxl_e2e_clipl_weights.safetensors")
bigg_file <- file.path(fixture_dir, "sdxl_e2e_bigg_weights.safetensors")
unet_file <- file.path(fixture_dir, "sdxl_e2e_unet_weights.safetensors")
vae <- Sys.glob(file.path(
    Sys.getenv("HOME"),
    ".cache/huggingface/hub/datasets--cornball-ai--sdxl-R",
    "snapshots/*/diffusers/vae/diffusion_pytorch_model.safetensors"))[1]
if (!file.exists(io_file) || !file.exists(clipl_file) ||
    !file.exists(bigg_file) || !file.exists(unet_file) ||
    is.na(vae) || !file.exists(vae)) {
    exit_file("fixture or checkpoint missing; run tools/gen_fixture_sdxl_e2e.R")
}

N_STEPS <- 4L
GUIDANCE <- 7.5
IMG_DIM <- 128L
LATENT <- 16L

f <- anvl::nv_read(io_file)
ids0 <- matrix(as.integer(round(as.array(f$ids0))), nrow = 1L)

w_clipl <- yq_sdxl_clip_load_weights(clipl_file, num_layers = 12L,
                                     has_text_projection = FALSE, device = "cpu")
w_bigg <- yq_sdxl_clip_load_weights(bigg_file, num_layers = 32L,
                                    has_text_projection = TRUE, device = "cpu")
w_unet <- yq_sdxl_unet_load_weights(unet_file, device = "cpu")
w_vae <- yq_sdxl_vae_load_weights(vae, device = "cpu")

# ---- host-side DDIM schedule matches the reference timesteps ----
sched <- yq_sdxl_ddim_schedule(N_STEPS)
step_ratio <- 1000L %/% N_STEPS
want_ts <- as.integer(rev(round((0:(N_STEPS - 1L)) * step_ratio) + 1L))
expect_equal(sched$timesteps, want_ts)

# jit the UNet once; reuse across steps.
unet_fn <- anvl::jit(yq_sdxl_unet())
res <- yq_sdxl_generate(w_unet = w_unet, w_clipl = w_clipl, w_bigg = w_bigg,
                        w_vae = w_vae, ids = ids0, noise = f$latents0,
                        img_dim = IMG_DIM, num_inference_steps = N_STEPS,
                        guidance_scale = GUIDANCE, decode = TRUE,
                        device = "cpu", unet_fn = unet_fn)

report <- function(tag, got, want) {
    max_abs <- max(abs(got - want)); scale <- max(abs(want))
    correlation <- cor(as.vector(got), as.vector(want))
    cat(sprintf("%s: max %.3e mean %.3e cor %.7f (scale %.3f)\n",
                tag, max_abs, mean(abs(got - want)), correlation, scale))
    c(max_abs = max_abs, scale = scale, cor = correlation)
}

# ---- CLIP conditioning parity (context + pooled) ----
ctx <- report("sdxl e2e context ", as.array(res$context), as.array(f$context))
expect_true(ctx["cor"] > 0.99999)
expect_true(ctx["max_abs"] < 1e-3 * ctx["scale"])
pl <- report("sdxl e2e pooled  ", as.array(res$pooled), as.array(f$pooled))
expect_true(pl["cor"] > 0.99999)
expect_true(pl["max_abs"] < 1e-3 * pl["scale"])

# ---- GATE 1: step-1 UNet outputs (cond + uncond) ----
nc <- report("sdxl e2e step1 cond  ", as.array(res$step1$noise_cond),
             as.array(f$noise_cond_1))
expect_true(nc["cor"] > 0.99999)
expect_true(nc["max_abs"] < 1e-3 * nc["scale"])
nu <- report("sdxl e2e step1 uncond", as.array(res$step1$noise_uncond),
             as.array(f$noise_uncond_1))
expect_true(nu["cor"] > 0.99999)
expect_true(nu["max_abs"] < 1e-3 * nu["scale"])

# ---- GATE 2: step-1 latents (tight; CFG + DDIM-step wiring) ----
s1 <- report("sdxl e2e step1 latents", as.array(res$step1$latents),
             as.array(f$latents_1))
expect_true(s1["cor"] > 0.999999)
expect_true(s1["max_abs"] < 1e-3 * s1["scale"])

# ---- GATE 3: full 4-step trajectory (final latents + decoded pixels) ----
got <- as.array(res$latents); want <- as.array(f$final)
fin <- report("sdxl e2e final latents", got, want)
expect_equal(dim(got), c(1L, 4L, LATENT, LATENT))
expect_true(fin["cor"] > 0.9999)
expect_true(fin["max_abs"] < 0.02 * fin["scale"])
expect_true(mean(abs(got - want)) < 1e-3 * fin["scale"])

pgot <- as.array(res$pixels); pwant <- as.array(f$pixels)
pix <- report("sdxl e2e final pixels ", pgot, pwant)
expect_equal(dim(pgot), c(1L, 3L, 8L * LATENT, 8L * LATENT))
expect_true(pix["cor"] > 0.9999)
expect_true(pix["max_abs"] < 0.02 * pix["scale"])
