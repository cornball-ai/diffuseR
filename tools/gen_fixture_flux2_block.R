# Generate the FLUX.2 single-block parity fixture for yunque.
#
# Reads the 4 real bf16 tensors of single_transformer_blocks.0 straight
# from the HF checkpoint (partial read; the file is ~8GB), upcasts to
# f32, runs diffuseR::flux2_single_block on fixed random inputs, and
# saves weights + inputs + output to a f32 safetensors fixture that the
# anvl side loads with anvl::nv_read().
#
# Usage: r tools/gen_fixture_flux2_block.R

suppressMessages({
  library(torch)
})

ckpt <- file.path(
  Sys.getenv("HOME"),
  ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B",
  "snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294",
  "transformer/diffusion_pytorch_model.safetensors"
)
stopifnot(file.exists(ckpt))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/yunque/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "flux2_single_block.safetensors")

# ---- partial safetensors reader (bf16 -> f32, base R) -----------------
con <- file(ckpt, "rb")
header_len <- readBin(con, "integer", n = 1, size = 8, endian = "little")
header <- jsonlite::fromJSON(readChar(con, header_len, useBytes = TRUE))
data_start <- 8 + header_len

read_bf16 <- function(key) {
  meta <- header[[key]]
  stopifnot(meta$dtype == "BF16")
  n <- prod(meta$shape)
  seek(con, data_start + meta$data_offsets[1])
  b <- readBin(con, "raw", n = n * 2L)
  # bf16 is the top half of an f32: place the two bytes as the high
  # bytes of a little-endian float32
  out <- raw(n * 4L)
  idx <- seq_len(n)
  out[4L * idx - 1L] <- b[2L * idx - 1L]
  out[4L * idx] <- b[2L * idx]
  vals <- readBin(out, "numeric", n = n, size = 4L, endian = "little")
  # file is row-major; torch reshape is row-major too
  torch_tensor(vals, dtype = torch_float32())$reshape(meta$shape)
}

pfx <- "single_transformer_blocks.0.attn."
w_qkv <- read_bf16(paste0(pfx, "to_qkv_mlp_proj.weight"))
w_out <- read_bf16(paste0(pfx, "to_out.weight"))
norm_q_w <- read_bf16(paste0(pfx, "norm_q.weight"))
norm_k_w <- read_bf16(paste0(pfx, "norm_k.weight"))
close(con)
cat(sprintf("weights: qkv %s, out %s\n",
            paste(dim(w_qkv), collapse = "x"),
            paste(dim(w_out), collapse = "x")))

# ---- torch reference forward -----------------------------------------
dim <- 3072L
heads <- 24L
head_dim <- 128L
S <- 512L

blk <- diffuseR::flux2_single_block(dim, heads, head_dim)
with_no_grad({
  blk$attn$to_qkv_mlp_proj$weight$copy_(w_qkv)
  blk$attn$to_out$weight$copy_(w_out)
  blk$attn$norm_q$weight$copy_(norm_q_w)
  blk$attn$norm_k$weight$copy_(norm_k_w)
})
blk$eval()

torch_manual_seed(42)
h <- torch_randn(1, S, dim) * 0.5
temb_mod <- torch_randn(1, dim * 3L) * 0.1
theta <- (torch_rand(S, head_dim %/% 2L) - 0.5) * (2 * pi)
cos <- torch_repeat_interleave(torch_cos(theta), 2L, dim = -1L)
sin <- torch_repeat_interleave(torch_sin(theta), 2L, dim = -1L)

out <- with_no_grad(blk(h, temb_mod, image_rotary_emb = list(cos, sin)))

# split the modulation the way the block does, so the anvl side gets
# shift/scale/gate directly. clone() is load-bearing: chunk() returns
# storage-offset views, safetensors::safe_save_file writes them from
# offset 0 (silently saving three copies of the first chunk), and
# contiguous() doesn't copy because the size-1 dims make the view's
# strides look packed already.
mod <- temb_mod$unsqueeze(2L)
parts <- mod$chunk(3L, dim = -1L)

safetensors::safe_save_file(list(
  w_qkv = w_qkv, w_out = w_out,
  norm_q_w = norm_q_w, norm_k_w = norm_k_w,
  h = h,
  shift = parts[[1]]$clone(),
  scale = parts[[2]]$clone(),
  gate = parts[[3]]$clone(),
  cos = cos, sin = sin, out = out
), fixture)

cat(sprintf("fixture: %s (%.0f MB)\n", fixture,
            file.size(fixture) / 1e6))
cat(sprintf("out: mean %.6f sd %.6f range [%.4f, %.4f]\n",
            out$mean()$item(), out$std()$item(),
            out$min()$item(), out$max()$item()))
