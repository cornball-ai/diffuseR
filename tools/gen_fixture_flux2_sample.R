# End-to-end FLUX.2 Klein sampling fixture (torch reference): Qwen3
# encoder -> DiT -> 4-step FlowMatch Euler loop, producing final packed
# latents. Encoder and DiT are loaded sequentially to bound memory. The
# anvl test reruns the whole pipeline and compares final latents.
#
# Usage: r tools/gen_fixture_flux2_sample.R

suppressMessages(library(torch))

snap <- file.path(
  Sys.getenv("HOME"),
  ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B",
  "snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294")
te <- file.path(snap, "text_encoder")
dit_ckpt <- file.path(snap, "transformer/diffusion_pytorch_model.safetensors")
fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/yunque/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "flux2_sample.safetensors")

H2 <- 16L; W2 <- 16L; S_txt <- 32L; N_ENC <- 27L; N_STEPS <- 4L
OUT_LAYERS <- c(9L, 18L, 27L)

read_shard_dir <- function(dir) {
  idx <- jsonlite::fromJSON(file.path(dir, "model.safetensors.index.json"))
  wm <- idx$weight_map
  cons <- lapply(unique(unlist(wm)), function(s) {
    con <- file(file.path(dir, s), "rb")
    hl <- readBin(con, "integer", n = 1, size = 8, endian = "little")
    list(con = con, header = jsonlite::fromJSON(readChar(con, hl, useBytes = TRUE)),
         start = 8 + hl)
  })
  names(cons) <- unique(unlist(wm))
  list(cons = cons, wm = wm)
}
read_bf16_at <- function(sc, meta) {
  n <- prod(meta$shape)
  seek(sc$con, sc$start + meta$data_offsets[1])
  if (meta$dtype == "F32") {
    vals <- readBin(sc$con, "numeric", n = n, size = 4L, endian = "little")
  } else {
    b <- readBin(sc$con, "raw", n = n * 2L); o <- raw(n * 4L); ii <- seq_len(n)
    o[4L * ii - 1L] <- b[2L * ii - 1L]; o[4L * ii] <- b[2L * ii]
    vals <- readBin(o, "numeric", n = n, size = 4L, endian = "little")
  }
  torch_tensor(vals, dtype = torch_float32())$reshape(meta$shape)
}

# ---- inputs ----
set.seed(11)
vocab <- 151936L
ids0 <- sample.int(vocab, S_txt, replace = TRUE) - 1L
attn <- rep(1L, S_txt)
torch_manual_seed(11)
noise <- torch_randn(c(1L, 128L, H2, W2), dtype = torch_float32())
latents0 <- noise$reshape(c(1L, 128L, H2 * W2))$permute(c(1L, 3L, 2L))  # pack
seq_img <- H2 * W2

# ---- sigmas (dynamic exponential shift + terminal 0) ----
mu <- diffuseR::flux2_empirical_mu(seq_img, N_STEPS)
sig_in <- seq(1, 1 / N_STEPS, length.out = N_STEPS)
sig <- exp(mu) / (exp(mu) + (1 / sig_in - 1))     # exponential, sigma exp = 1
sigmas <- c(sig, 0)

# ---- Phase 1: text encode (then free) ----
cat("encode...\n")
sd_te <- read_shard_dir(te)
enc <- diffuseR::qwen3_encoder(num_hidden_layers = N_ENC)
need <- names(enc$state_dict())
sdE <- setNames(lapply(need, function(k) {
  sc <- sd_te$cons[[sd_te$wm[[k]]]]; read_bf16_at(sc, sc$header[[k]])
}), need)
enc$load_state_dict(sdE); enc$eval()
for (c in sd_te$cons) close(c$con)
rm(sdE); gc()
ids_t <- torch_tensor(matrix(ids0 + 1L, 1L), dtype = torch_long())
mask_t <- torch_tensor(matrix(attn, 1L), dtype = torch_long())
states <- with_no_grad(enc(ids_t, attention_mask = mask_t, out_layers = OUT_LAYERS))
text_embeds <- torch_cat(states, dim = -1L)
rm(enc, states); gc()

# ---- Phase 2: DiT + loop ----
cat("load DiT...\n")
con <- file(dit_ckpt, "rb")
hl <- readBin(con, "integer", n = 1, size = 8, endian = "little")
hdr <- jsonlite::fromJSON(readChar(con, hl, useBytes = TRUE)); dstart <- 8 + hl
read_dit <- function(key) {
  m <- hdr[[key]]; n <- prod(m$shape); seek(con, dstart + m$data_offsets[1])
  b <- readBin(con, "raw", n = n * 2L); o <- raw(n * 4L); ii <- seq_len(n)
  o[4L * ii - 1L] <- b[2L * ii - 1L]; o[4L * ii] <- b[2L * ii]
  torch_tensor(readBin(o, "numeric", n = n, size = 4L, endian = "little"),
               dtype = torch_float32())$reshape(m$shape)
}
dit <- diffuseR::flux2_transformer()
kd <- names(dit$state_dict())
dit$load_state_dict(setNames(lapply(kd, read_dit), kd)); dit$eval()
close(con)

txt_ids <- diffuseR::flux2_prepare_text_ids(S_txt)
latent_ids <- diffuseR::flux2_prepare_latent_ids(H2, W2)
ids <- torch_cat(list(txt_ids, latent_ids), dim = 1L)
rope <- diffuseR::flux_pos_embed(ids, axes_dim = c(32L, 32L, 32L, 32L), theta = 2000)

cat("denoise", N_STEPS, "steps...\n")
lat <- latents0
v1 <- NULL; lat1 <- NULL
with_no_grad({
  for (i in seq_len(N_STEPS)) {
    v <- dit(hidden_states = lat, encoder_hidden_states = text_embeds,
             timestep = torch_tensor(sigmas[i])$reshape(1L),
             image_rotary_emb = rope)
    lat <- lat + (sigmas[i + 1] - sigmas[i]) * v
    if (i == 1L) { v1 <- v; lat1 <- lat }
  }
})

# $contiguous() is load-bearing on latents0: it is a permute view of a
# reshape, and safe_save_file writes views from the wrong strides.
safetensors::safe_save_file(list(
  ids0 = torch_tensor(matrix(ids0, 1L), dtype = torch_float32()),
  attn = torch_tensor(matrix(attn, 1L), dtype = torch_float32()),
  latents0 = latents0$contiguous(),
  sigmas = torch_tensor(sigmas, dtype = torch_float32()),
  text_embeds = text_embeds$contiguous(),
  v1 = v1$contiguous(), lat1 = lat1$contiguous(),
  final = lat$contiguous()
), fixture)
cat(sprintf("fixture: %s (%.2f MB)\n", fixture, file.size(fixture) / 1e6))
cat(sprintf("final latents sd %.4f range [%.3f, %.3f]\n",
            lat$std()$item(), lat$min()$item(), lat$max()$item()))
