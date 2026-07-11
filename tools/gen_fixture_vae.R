# FLUX.2 VAE decoder parity fixture (torch reference). Loads the real
# decoder weights (bf16->f32) into diffuseR::flux2_vae_decoder, decodes
# a random 32-channel latent, saves input + output. The anvl test
# reloads weights via yunque::yq_flux2_load_vae.
#
# Usage: r tools/gen_fixture_vae.R

suppressMessages(library(torch))

vae <- Sys.glob(file.path(
  Sys.getenv("HOME"),
  ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B",
  "snapshots/*/vae/diffusion_pytorch_model.safetensors"))[1]
stopifnot(file.exists(vae))
fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/yunque/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "flux2_vae.safetensors")

con <- file(vae, "rb")
hl <- readBin(con, "integer", n = 1, size = 8, endian = "little")
hdr <- jsonlite::fromJSON(readChar(con, hl, useBytes = TRUE)); dstart <- 8 + hl
read_one <- function(key) {
  m <- hdr[[key]]; n <- prod(m$shape); seek(con, dstart + m$data_offsets[1])
  if (m$dtype == "F32") {
    vals <- readBin(con, "numeric", n = n, size = 4L, endian = "little")
  } else {
    b <- readBin(con, "raw", n = n * 2L); o <- raw(n * 4L); ii <- seq_len(n)
    o[4L * ii - 1L] <- b[2L * ii - 1L]; o[4L * ii] <- b[2L * ii]
    vals <- readBin(o, "numeric", n = n, size = 4L, endian = "little")
  }
  torch_tensor(vals, dtype = torch_float32())$reshape(m$shape)
}

dec <- diffuseR::flux2_vae_decoder(latent_channels = 32L,
                                   block_channels = c(512L, 512L, 256L, 128L))
need <- names(dec$state_dict())
# module keys mirror the checkpoint (decoder.*, post_quant_conv.*, bn stats)
avail <- setdiff(names(hdr), "__metadata__")
sd <- list()
for (k in need) {
  ck <- k
  if (ck %in% avail) sd[[k]] <- read_one(ck)
}
close(con)
miss <- setdiff(need, names(sd))
if (length(miss)) cat("unfilled module keys:", length(miss), "->",
                      paste(head(miss, 5), collapse = ", "), "\n")
dec$load_state_dict(sd, strict = FALSE)
dec$eval()

set.seed(9); torch_manual_seed(9)
z <- torch_randn(1, 32L, 8L, 8L)
out <- with_no_grad(dec(z))

safetensors::safe_save_file(list(
  z = z$contiguous(),
  out = out$contiguous()
), fixture)
cat(sprintf("fixture: %s (%.2f MB)\n", fixture, file.size(fixture) / 1e6))
cat(sprintf("out shape %s sd %.4f range [%.3f, %.3f]\n",
            paste(dim(out), collapse = "x"), out$std()$item(),
            out$min()$item(), out$max()$item()))
