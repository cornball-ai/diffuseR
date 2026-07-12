# Generate the SDXL end-to-end parity fixture (torch reference) on REAL
# F16 weights. Loads the diffusers SDXL checkpoints (cornball-ai/sdxl-R
# cache) into the native torch modules and runs the full text-to-image
# loop on a FIXED prompt, seed, a small step count (4), and low resolution
# (128px -> 16x16 latent): dual CLIP encode (concatenated PENULTIMATE
# context + bigG pooled, diffusers SDXL clip-skip=None) -> epsilon DDIM
# denoise with classifier-free guidance (scale 7.5) and the SDXL added
# text-time conditioning (pooled + time-ids) -> VAE decode (0.13025). Saves
# the token ids, initial noise, time-ids, cond context + pooled, step-1
# UNet outputs (to isolate CFG/scheduler from the UNet), post-step-1
# latents, final latents, and decoded pixels; plus the three native
# state_dicts (UNet/CLIP-L/bigG) the anvl loaders read back.
#
# NB: the reference conditioning uses the PENULTIMATE hidden states
# (hidden_states[-2]), matching the anvl encoder + anvl_test_sdxl_clip, not
# txt2img_sdxl's simplified full-forward `pipeline$text_encoder(tokens)`.
#
# Usage: /home/troy/diffuseR-sdxl-lib/ranvl tools/gen_fixture_sdxl_e2e.R

suppressMessages(library(torch))
suppressMessages(library(diffuseR))

diffusers <- Sys.glob(file.path(
  Sys.getenv("HOME"),
  ".cache/huggingface/hub/datasets--cornball-ai--sdxl-R",
  "snapshots/*/diffusers"))[1]
stopifnot(!is.na(diffusers), dir.exists(file.path(diffusers, "unet")))
te_dir <- file.path(diffusers, "text_encoder")
te2_dir <- file.path(diffusers, "text_encoder_2")
unet_dir <- file.path(diffusers, "unet")
vae_dir <- file.path(diffusers, "vae")
vae_file <- file.path(vae_dir, "diffusion_pytorch_model.safetensors")

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
io_file <- file.path(fixture_dir, "sdxl_e2e_io.safetensors")
clipl_file <- file.path(fixture_dir, "sdxl_e2e_clipl_weights.safetensors")
bigg_file <- file.path(fixture_dir, "sdxl_e2e_bigg_weights.safetensors")
unet_file <- file.path(fixture_dir, "sdxl_e2e_unet_weights.safetensors")

PROMPT <- "a red panda astronaut floating in space, photorealistic"
N_STEPS <- 4L
GUIDANCE <- 7.5
IMG_DIM <- 128L
LATENT <- IMG_DIM %/% 8L                       # 16
SCALING <- 0.13025
SEED <- 77L

# int64-safe, row-major F32 safetensors writer (shared with the SDXL UNet /
# CLIP fixtures): the R `safetensors` writer overflows int32 when computing
# data offsets past 2 GB (corrupting the header); yunque's reader handles
# int64 offsets fine.
save_st_f32 <- function(tensors, path) {
    keys <- names(tensors)
    shapes <- lapply(tensors, function(t) as.integer(t$shape))
    nbytes <- vapply(shapes, function(s) prod(as.double(s)) * 4, numeric(1))
    ends <- cumsum(nbytes)
    starts <- c(0, ends[-length(ends)])
    header <- list()
    for (i in seq_along(keys)) {
        header[[keys[i]]] <- list(
            dtype = "F32",
            shape = as.list(shapes[[i]]),
            data_offsets = list(starts[i], ends[i]))
    }
    jraw <- charToRaw(jsonlite::toJSON(header, auto_unbox = TRUE, digits = NA))
    con <- file(path, "wb")
    on.exit(close(con))
    writeBin(length(jraw), con, size = 4L, endian = "little")
    writeBin(0L, con, size = 4L, endian = "little")
    writeBin(jraw, con)
    for (k in keys) {
        v <- as.array(tensors[[k]]$to(dtype = torch_float32())$contiguous()$reshape(c(-1L)))
        writeBin(as.numeric(v), con, size = 4L, endian = "little")
    }
}

# ---- native modules, real weights (F16 upcast to f32) ----
clipl <- text_encoder_native_from_safetensors(te_dir, apply_final_ln = FALSE,
                                              gelu_type = "quick", verbose = FALSE)
clipl$eval()
bigg <- text_encoder2_native(vocab_size = 49408L, context_length = 77L,
                             embed_dim = 1280L, num_layers = 32L,
                             num_heads = 20L, mlp_dim = 5120L)
load_text_encoder_safetensors(bigg, te2_dir, verbose = FALSE)
bigg$eval()
unet <- unet_sdxl_native_from_safetensors(unet_dir, verbose = TRUE)
dec <- vae_decoder_native_from_safetensors(vae_dir, latent_channels = 4L,
                                           verbose = FALSE)
handle <- safetensors::safetensors$new(vae_file, framework = "torch")
pq_w <- handle$get_tensor("post_quant_conv.weight")$to(dtype = torch_float32())
pq_b <- handle$get_tensor("post_quant_conv.bias")$to(dtype = torch_float32())

# ---- tokenize (shared ids for both encoders, exactly like txt2img_sdxl) ----
ids_t <- CLIPTokenizer(PROMPT)                 # [1, 77] long, 0-based
ids0 <- as.integer(as.array(ids_t))
S <- length(ids0)
eos_pos <- which.max(ids0)
cat(sprintf("tokens: %d, EOS (which.max) at %d\n", S, eos_pos))

# ---- SDXL conditioning: concatenated PENULTIMATE context + bigG pooled ----
penultimate <- function(model, ids_t, n_layers) {
    tok <- model$token_embedding(ids_t + 1L)
    S <- ids_t$shape[2]
    pos <- model$position_embedding[1:S, ]$unsqueeze(1)$expand(
        c(ids_t$shape[1], -1, -1))
    h <- tok + pos
    for (i in seq_len(n_layers - 1L)) h <- model$transformer_blocks[[i]](h)
    h
}
bigg_pooled <- function(model, ids_t, n_layers) {
    tok <- model$token_embedding(ids_t + 1L)
    S <- ids_t$shape[2]
    pos <- model$position_embedding[1:S, ]$unsqueeze(1)$expand(
        c(ids_t$shape[1], -1, -1))
    h <- tok + pos
    for (i in seq_len(n_layers)) h <- model$transformer_blocks[[i]](h)
    h_ln <- model$final_layer_norm(h)
    eos <- torch_argmax(ids_t, dim = 2L, keepdim = TRUE)
    pre <- h_ln$gather(dim = 2L,
                       index = eos$unsqueeze(-1L)$expand(
                           c(-1L, -1L, model$embed_dim)))$squeeze(2L)
    model$text_projection(pre)
}

cond <- with_no_grad({
    pen_l <- penultimate(clipl, ids_t, 12L)            # [1, S, 768]
    pen_g <- penultimate(bigg, ids_t, 32L)             # [1, S, 1280]
    context <- torch_cat(list(pen_l, pen_g), dim = 3L) # [1, S, 2048]
    pooled <- bigg_pooled(bigg, ids_t, 32L)            # [1, 1280]
    list(context = context, pooled = pooled)
})
context <- cond$context
pooled <- cond$pooled
uncond_context <- torch_zeros_like(context)
uncond_pooled <- torch_zeros_like(pooled)

# ---- SDXL micro-conditioning time-ids (orig/crop/target) ----
time_ids <- torch_tensor(
    matrix(c(IMG_DIM, IMG_DIM, 0, 0, IMG_DIM, IMG_DIM), nrow = 1L),
    dtype = torch_float32())

# ---- initial noise ----
set.seed(SEED); torch_manual_seed(SEED)
latents0 <- torch_randn(c(1L, 4L, LATENT, LATENT), dtype = torch_float32())

# ---- epsilon DDIM CFG loop (mirrors txt2img_sdxl) ----
schedule <- ddim_scheduler_create(num_inference_steps = N_STEPS,
                                  beta_schedule = "scaled_linear",
                                  beta_start = 0.00085, beta_end = 0.012,
                                  device = torch_device("cpu"))
timesteps <- schedule$timesteps
cat("timesteps:", paste(timesteps, collapse = ", "), "\n")

latents <- latents0
nc1 <- NULL; nu1 <- NULL; lat1 <- NULL
with_no_grad({
    for (i in seq_along(timesteps)) {
        timestep <- torch_tensor(timesteps[i], dtype = torch_long())
        noise_cond <- unet(latents, timestep, context, pooled, time_ids)
        noise_uncond <- unet(latents, timestep, uncond_context, uncond_pooled,
                             time_ids)
        noise_pred <- noise_uncond + GUIDANCE * (noise_cond - noise_uncond)
        latents <- ddim_scheduler_step(model_output = noise_pred,
                                       timestep = timestep, sample = latents,
                                       schedule = schedule,
                                       prediction_type = "epsilon",
                                       device = "cpu")
        if (i == 1L) { nc1 <- noise_cond; nu1 <- noise_uncond; lat1 <- latents }
    }
})
final <- latents

# ---- VAE decode ----
pixels <- with_no_grad({
    scaled <- final / SCALING
    dec(nnf_conv2d(scaled, weight = pq_w, bias = pq_b))
})   # [1, 3, 128, 128], [-1, 1]

# ---- save native state_dicts (int64-safe) ----
save_st_f32(clipl$state_dict(), clipl_file)
save_st_f32(bigg$state_dict(), bigg_file)
save_st_f32(unet$state_dict(), unet_file)
cat(sprintf("clipl weights: %.2f GB (%d)\n", file.size(clipl_file) / 1e9,
            length(clipl$state_dict())))
cat(sprintf("bigg  weights: %.2f GB (%d)\n", file.size(bigg_file) / 1e9,
            length(bigg$state_dict())))
cat(sprintf("unet  weights: %.2f GB (%d)\n", file.size(unet_file) / 1e9,
            length(unet$state_dict())))

# ---- save IO (small; the R safetensors writer is fine here) ----
safetensors::safe_save_file(list(
    ids0 = torch_tensor(matrix(ids0, 1L), dtype = torch_float32()),
    latents0 = latents0$contiguous(),
    time_ids = time_ids$contiguous(),
    context = context$contiguous(),
    pooled = pooled$contiguous(),
    noise_cond_1 = nc1$contiguous(),
    noise_uncond_1 = nu1$contiguous(),
    latents_1 = lat1$contiguous(),
    final = final$contiguous(),
    pixels = pixels$contiguous()
), io_file)

cat(sprintf("io: %s (%.2f MB)\n", io_file, file.size(io_file) / 1e6))
cat(sprintf("final latents sd %.4f range [%.3f, %.3f]\n",
            final$std()$item(), final$min()$item(), final$max()$item()))
cat(sprintf("pixels sd %.4f range [%.3f, %.3f]\n",
            pixels$std()$item(), pixels$min()$item(), pixels$max()$item()))
