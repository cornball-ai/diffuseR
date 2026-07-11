# Dump per-stage intermediates of the FLUX.2 single block (torch side),
# mirroring diffuseR::flux2_single_block exactly, for divergence hunting.
# Usage: r tools/debug_stages_torch.R

suppressMessages(library(torch))

fdir <- file.path(Sys.getenv("HOME"), ".local/share/R/yunque/fixtures")
w <- safetensors::safe_load_file(file.path(fdir, "flux2_single_block.safetensors"),
                                 framework = "torch")

dim <- 3072L
heads <- 24L
head_dim <- 128L
inner <- heads * head_dim
mlp_hidden <- as.integer(dim * 3)

with_no_grad({
  norm1 <- nnf_layer_norm(w$h, dim, eps = 1e-6)
  norm_h <- norm1 * (1 + w$scale) + w$shift

  proj <- nnf_linear(norm_h, w$w_qkv)
  qkv <- proj$narrow(-1L, 1L, 3L * inner)
  mlp <- proj$narrow(-1L, 3L * inner + 1L, mlp_hidden * 2L)
  parts <- qkv$chunk(3L, dim = -1L)
  q <- parts[[1]]$unflatten(3L, c(heads, -1L))
  k <- parts[[2]]$unflatten(3L, c(heads, -1L))

  rms <- function(x, wt) {
    v <- x$pow(2)$mean(dim = -1L, keepdim = TRUE)
    x * torch_rsqrt(v + 1e-6) * wt
  }
  qn <- rms(q, w$norm_q_w)
  kn <- rms(k, w$norm_k_w)

  qt <- qn$transpose(2L, 3L)
  cos <- w$cos$unsqueeze(1L)$unsqueeze(1L)
  sin <- w$sin$unsqueeze(1L)$unsqueeze(1L)
  pairs <- qt$unflatten(4L, c(-1L, 2L))
  x_real <- pairs[, , , , 1]
  x_imag <- pairs[, , , , 2]
  x_rot <- torch_stack(list(-x_imag, x_real), dim = -1L)$flatten(start_dim = 4L)
  q_rope <- qt * cos + x_rot * sin

  kt <- kn$transpose(2L, 3L)
  pairs <- kt$unflatten(4L, c(-1L, 2L))
  k_rot <- torch_stack(list(-pairs[, , , , 2], pairs[, , , , 1]),
                       dim = -1L)$flatten(start_dim = 4L)
  k_rope <- kt * cos + k_rot * sin

  v <- parts[[3]]$unflatten(3L, c(heads, -1L))$transpose(2L, 3L)
  scores <- torch_matmul(q_rope$mul(1 / sqrt(head_dim)),
                         k_rope$transpose(-2L, -1L))
  attn <- torch_matmul(nnf_softmax(scores, dim = -1L), v)
  attn_flat <- attn$transpose(2L, 3L)$flatten(start_dim = 3L)

  m1 <- mlp$narrow(-1L, 1L, mlp_hidden)
  m2 <- mlp$narrow(-1L, mlp_hidden + 1L, mlp_hidden)
  mlp_act <- nnf_silu(m1) * m2

  out <- nnf_linear(torch_cat(list(attn_flat, mlp_act), dim = -1L), w$w_out)
  final <- w$h + w$gate * out
})

safetensors::safe_save_file(list(
  norm_h = norm_h, proj = proj, qn = qn, q_rope = q_rope, k_rope = k_rope,
  attn_flat = attn_flat, mlp_act = mlp_act, final = final
), file.path(fdir, "flux2_stages_torch.safetensors"))

# sanity: manual mirror must equal the fixture's block output
cat(sprintf("manual vs block out: %.3e\n",
            (final - w$out)$abs()$max()$item()))
