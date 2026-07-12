# Z-Image DiT parity fixture (torch reference, RANDOM-INIT weights).
#
# Architecture parity: instantiate diffuseR::zimage_transformer with a
# small config and torch_manual_seed random weights, run the real forward
# under no_grad on random inputs, and save the module state_dict + the
# inputs the anvl closure needs + the reference output to one f32
# safetensors. The anvl loader reads the SAME state_dict.
#
# The DiT closure works on packed tokens and returns the packed
# final-layer output; patchify/unpatchify are host-side glue. The
# reference packed image-span output is zimage_patchify(forward_output)
# (patchify and unpatchify are inverses over the image span), which lets
# us compare without touching the unpatchify glue.
#
# Everything is $contiguous() before saving (view-save trap).
#
# Usage: r tools/gen_fixture_zimage_dit.R

suppressMessages(library(torch))
suppressMessages(library(diffuseR))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "zimage_dit.safetensors")

# ---- small but architecturally faithful config ----
# dim > adaln_dim (256) exercises the min(dim, 256) modulation-width
# logic; axes_dims sum to head_dim = dim / n_heads. head_dim = 56 is
# deliberately NOT a multiple of 32: padded sub-sequence lengths are
# always multiples of 32, so head_dim can never equal a sequence length
# and the shared .ltx23_sdpa scratch-buffer aliasing (attn_buf and
# out_buf collide when n_q == head_dim) never triggers.
in_channels <- 4L
dim <- 336L
n_layers <- 2L
n_refiner_layers <- 2L
n_heads <- 6L
cap_feat_dim <- 48L
axes_dims <- c(16L, 20L, 20L)   # sum 56 = 336 / 6
patch_size <- 2L
f_patch_size <- 1L
theta <- 256
t_scale <- 1000

torch_manual_seed(11)
m <- zimage_transformer(in_channels = in_channels, dim = dim,
                        n_layers = n_layers, n_refiner_layers = n_refiner_layers,
                        n_heads = n_heads, cap_feat_dim = cap_feat_dim,
                        rope_theta = theta, t_scale = t_scale,
                        axes_dims = axes_dims, patch_size = patch_size,
                        f_patch_size = f_patch_size)
m$eval()

# ---- random inputs ----
F_ <- 1L; H <- 8L; W <- 12L
cap_len <- 5L
torch_manual_seed(101)
x <- torch_randn(in_channels, F_, H, W)
cap_feats <- torch_randn(cap_len, cap_feat_dim)
t <- torch_tensor(0.7)                     # [1], in [0, 1]

out <- with_no_grad(m(x, t, cap_feats))    # [C, F, H, W]

# ---- inputs the anvl closure needs ----
tokens <- zimage_patchify(x, patch_size, f_patch_size)$unsqueeze(1L)  # [1, img_len, patch_dim]
cap_in <- cap_feats$unsqueeze(1L)                                     # [1, cap_len, cap_feat_dim]
t_freq <- ltx23_get_timestep_embedding((t * t_scale)$reshape(1L), 256L,
                                        flip_sin_to_cos = TRUE,
                                        downscale_freq_shift = 0)      # [1, 256]

# ---- reference RoPE tables (image + caption), matching the forward ----
h_tokens <- H %/% patch_size
w_tokens <- W %/% patch_size
f_tokens <- F_ %/% f_patch_size
cap_padded <- cap_len + ((-cap_len) %% 32L)
cap_freqs <- zimage_pos_embed(zimage_cap_pos_ids(cap_padded),
                              axes_dim = axes_dims, theta = theta)
img_freqs <- zimage_pos_embed(
    zimage_img_pos_ids(h_tokens, w_tokens, start0 = cap_padded + 1L,
                       f_tokens = f_tokens),
    axes_dim = axes_dims, theta = theta)

# reference packed image-span output = patchify of the [C,F,H,W] output
out_img <- zimage_patchify(out, patch_size, f_patch_size)             # [img_len, patch_dim]

ct <- function(x) x$contiguous()
sd <- lapply(m$state_dict(), ct)
inputs <- list(
    tokens = ct(tokens),
    cap_feats = ct(cap_in),
    t_freq = ct(t_freq),
    cos_img = ct(img_freqs[[1]]),
    sin_img = ct(img_freqs[[2]]),
    cos_cap = ct(cap_freqs[[1]]),
    sin_cap = ct(cap_freqs[[2]]),
    out_img = ct(out_img)
)
stopifnot(length(intersect(names(sd), names(inputs))) == 0L)
safetensors::safe_save_file(c(sd, inputs), fixture)

cat(sprintf("fixture: %s (%.2f MB)\n", fixture, file.size(fixture) / 1e6))
cat(sprintf("config: dim=%d heads=%d layers=%d refiners=%d cap_dim=%d axes=%s\n",
            dim, n_heads, n_layers, n_refiner_layers, cap_feat_dim,
            paste(axes_dims, collapse = ",")))
cat(sprintf("tokens %s  cap %s  out_img %s\n",
            paste(dim(tokens), collapse = "x"),
            paste(dim(cap_in), collapse = "x"),
            paste(dim(out_img), collapse = "x")))
cat(sprintf("out_img sd %.4f  range [%.3f, %.3f]\n",
            out_img$std()$item(), out_img$min()$item(), out_img$max()$item()))
