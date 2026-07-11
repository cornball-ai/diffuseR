# Full FLUX.2 Klein transformer forward fixture (torch reference).
# Loads all bf16 weights (upcast f32) into diffuseR::flux2_transformer,
# runs at a small resolution, saves inputs + output. Weights are NOT
# saved (the test reloads them from the checkpoint via
# yunque::yq_flux2_load_weights). CPU f32 reference.
#
# Usage: r tools/gen_fixture_flux2_forward.R

suppressMessages(library(torch))

ckpt <- file.path(
  Sys.getenv("HOME"),
  ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B",
  "snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294",
  "transformer/diffusion_pytorch_model.safetensors"
)
stopifnot(file.exists(ckpt))
fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/yunque/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "flux2_forward.safetensors")

# ---- read full state dict (bf16 -> f32) ----
con <- file(ckpt, "rb")
header_len <- readBin(con, "integer", n = 1, size = 8, endian = "little")
header <- jsonlite::fromJSON(readChar(con, header_len, useBytes = TRUE))
data_start <- 8 + header_len
keys <- setdiff(names(header), "__metadata__")
read_one <- function(key) {
  meta <- header[[key]]
  n <- prod(meta$shape)
  seek(con, data_start + meta$data_offsets[1])
  if (meta$dtype == "F32") {
    vals <- readBin(con, "numeric", n = n, size = 4L, endian = "little")
  } else {
    b <- readBin(con, "raw", n = n * 2L)
    out <- raw(n * 4L); idx <- seq_len(n)
    out[4L * idx - 1L] <- b[2L * idx - 1L]
    out[4L * idx] <- b[2L * idx]
    vals <- readBin(out, "numeric", n = n, size = 4L, endian = "little")
  }
  torch_tensor(vals, dtype = torch_float32())$reshape(meta$shape)
}
cat("reading", length(keys), "tensors...\n")
sd <- setNames(lapply(keys, read_one), keys)
close(con)

m <- diffuseR::flux2_transformer()
m$load_state_dict(sd)
m$eval()
rm(sd); gc()

# ---- inputs: 16x16 packed grid (256 image tokens), 64 text tokens ----
H <- 16L; W <- 16L; S_txt <- 64L
in_channels <- 128L; joint_dim <- 7680L
torch_manual_seed(123)
latents <- torch_randn(1, H * W, in_channels)
text_embeds <- torch_randn(1, S_txt, joint_dim) * 0.5
timestep <- torch_tensor(0.7)

text_ids <- diffuseR::flux2_prepare_text_ids(S_txt)
latent_ids <- diffuseR::flux2_prepare_latent_ids(H, W)
ids <- torch_cat(list(text_ids, latent_ids), dim = 1L)
rope <- diffuseR::flux_pos_embed(ids, axes_dim = c(32L, 32L, 32L, 32L),
                                 theta = 2000)

out <- with_no_grad(m(
  hidden_states = latents, encoder_hidden_states = text_embeds,
  timestep = timestep, image_rotary_emb = rope
))

safetensors::safe_save_file(list(
  latents = latents, text_embeds = text_embeds,
  timestep = timestep$reshape(1L), out = out
), fixture)
cat(sprintf("fixture: %s (%.2f MB)\n", fixture, file.size(fixture) / 1e6))
cat(sprintf("out shape %s  sd %.4f  range [%.3f, %.3f]\n",
            paste(dim(out), collapse = "x"), out$std()$item(),
            out$min()$item(), out$max()$item()))
