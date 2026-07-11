# Generate the FLUX.2 double-stream (MMDiT) block parity fixture.
# Reads the 16 real bf16 tensors of transformer_blocks.0 from the HF
# checkpoint (partial read), upcasts to f32, runs
# diffuseR::flux2_double_block on fixed random inputs, saves weights +
# inputs + outputs to an f32 safetensors fixture.
#
# Usage: r tools/gen_fixture_flux2_double.R

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
fixture <- file.path(fixture_dir, "flux2_double_block.safetensors")

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
  out <- raw(n * 4L); idx <- seq_len(n)
  out[4L * idx - 1L] <- b[2L * idx - 1L]
  out[4L * idx] <- b[2L * idx]
  vals <- readBin(out, "numeric", n = n, size = 4L, endian = "little")
  torch_tensor(vals, dtype = torch_float32())$reshape(meta$shape)
}
pfx <- "transformer_blocks.0."
keys <- c(
  "attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight",
  "attn.norm_q.weight", "attn.norm_k.weight",
  "attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight",
  "attn.norm_added_q.weight", "attn.norm_added_k.weight",
  "attn.to_out.0.weight", "attn.to_add_out.weight",
  "ff.linear_in.weight", "ff.linear_out.weight",
  "ff_context.linear_in.weight", "ff_context.linear_out.weight"
)
wt <- lapply(keys, function(k) read_bf16(paste0(pfx, k)))
names(wt) <- keys
close(con)

dim <- 3072L; heads <- 24L; head_dim <- 128L
S_txt <- 64L; S_img <- 256L

blk <- diffuseR::flux2_double_block(dim, heads, head_dim)
with_no_grad({
  blk$attn$to_q$weight$copy_(wt[["attn.to_q.weight"]])
  blk$attn$to_k$weight$copy_(wt[["attn.to_k.weight"]])
  blk$attn$to_v$weight$copy_(wt[["attn.to_v.weight"]])
  blk$attn$norm_q$weight$copy_(wt[["attn.norm_q.weight"]])
  blk$attn$norm_k$weight$copy_(wt[["attn.norm_k.weight"]])
  blk$attn$add_q_proj$weight$copy_(wt[["attn.add_q_proj.weight"]])
  blk$attn$add_k_proj$weight$copy_(wt[["attn.add_k_proj.weight"]])
  blk$attn$add_v_proj$weight$copy_(wt[["attn.add_v_proj.weight"]])
  blk$attn$norm_added_q$weight$copy_(wt[["attn.norm_added_q.weight"]])
  blk$attn$norm_added_k$weight$copy_(wt[["attn.norm_added_k.weight"]])
  blk$attn$to_out[[1]]$weight$copy_(wt[["attn.to_out.0.weight"]])
  blk$attn$to_add_out$weight$copy_(wt[["attn.to_add_out.weight"]])
  blk$ff$linear_in$weight$copy_(wt[["ff.linear_in.weight"]])
  blk$ff$linear_out$weight$copy_(wt[["ff.linear_out.weight"]])
  blk$ff_context$linear_in$weight$copy_(wt[["ff_context.linear_in.weight"]])
  blk$ff_context$linear_out$weight$copy_(wt[["ff_context.linear_out.weight"]])
})
blk$eval()

torch_manual_seed(42)
h <- torch_randn(1, S_img, dim) * 0.5
cc <- torch_randn(1, S_txt, dim) * 0.5
mod_img <- torch_randn(1, dim * 6L) * 0.1
mod_txt <- torch_randn(1, dim * 6L) * 0.1
# rope over the concatenated [text; image] sequence
theta <- (torch_rand(S_txt + S_img, head_dim %/% 2L) - 0.5) * (2 * pi)
cos <- torch_repeat_interleave(torch_cos(theta), 2L, dim = -1L)
sin <- torch_repeat_interleave(torch_sin(theta), 2L, dim = -1L)

out <- with_no_grad(blk(
  hidden_states = h, encoder_hidden_states = cc,
  temb_mod_img = mod_img, temb_mod_txt = mod_txt,
  image_rotary_emb = list(cos, sin)
))
c_out <- out[[1]]; h_out <- out[[2]]

save_list <- c(
  setNames(lapply(keys, function(k) wt[[k]]$contiguous()), keys),
  list(h = h, c = cc, mod_img = mod_img, mod_txt = mod_txt,
       cos = cos, sin = sin, c_out = c_out, h_out = h_out)
)
safetensors::safe_save_file(save_list, fixture)
cat(sprintf("fixture: %s (%.0f MB)\n", fixture, file.size(fixture) / 1e6))
cat(sprintf("h_out sd %.4f  c_out sd %.4f\n",
            h_out$std()$item(), c_out$std()$item()))
