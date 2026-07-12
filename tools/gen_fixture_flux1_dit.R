# FLUX.1 MMDiT transformer parity fixture (torch reference), RANDOM-INIT
# at a small, memory-light config. Instantiates diffuseR::flux_transformer
# with random weights, runs it under with_no_grad on random latents +
# pooled text + T5 text embeds + precomputed RoPE, and saves the full
# state dict together with the inputs and output to one f32 safetensors
# file. The anvl test (inst/tinytest/anvl_test_flux1_dit.R) reloads the
# weights from this same file via yq_flux1_load_weights and feeds the
# saved inputs. CPU f32 reference.
#
# Keys: state-dict tensors under their diffusers names; inputs under
# input.*; the reference output under `output`.
#
# Usage: /home/troy/diffuseR-f1-lib/ranvl tools/gen_fixture_flux1_dit.R

suppressMessages(library(torch))
suppressMessages(library(diffuseR))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/yunque/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "flux1_dit.safetensors")

# ---- small config (few blocks, tiny dims -> fixture is a few hundred KB) ----
in_channels <- 8L
num_layers <- 2L
num_single_layers <- 2L
head_dim <- 16L
heads <- 4L
joint_dim <- 32L
pooled_dim <- 16L
axes_dims_rope <- c(4L, 6L, 6L)   # sum == head_dim
out_channels <- 8L

set.seed(1)
torch_manual_seed(1)

m <- flux_transformer(
    in_channels = in_channels, num_layers = num_layers,
    num_single_layers = num_single_layers, attention_head_dim = head_dim,
    num_attention_heads = heads, joint_attention_dim = joint_dim,
    pooled_projection_dim = pooled_dim, axes_dims_rope = axes_dims_rope,
    out_channels = out_channels)
m$eval()

# ---- inputs: 4x4 packed grid (16 image tokens), 8 text tokens ----
H <- 4L; W <- 4L; S_img <- H * W; S_txt <- 8L
latents <- torch_randn(1L, S_img, in_channels)
text_embeds <- torch_randn(1L, S_txt, joint_dim)$mul(0.5)
pooled <- torch_randn(1L, pooled_dim)
timestep <- torch_tensor(0.7)$reshape(1L)

# RoPE tables (static): text ids are all-zero, image ids from the packed
# grid. axes_dims_rope sums to head_dim.
text_ids <- torch_zeros(S_txt, 3L)
latent_ids <- flux_prepare_latent_image_ids(H, W)
ids <- torch_cat(list(text_ids, latent_ids), dim = 1L)
rope <- flux_pos_embed(ids, axes_dim = axes_dims_rope, theta = 10000)

# Reference internal timestep sinusoid (timestep * 1000), saved so the
# anvl parity path feeds it directly (isolates cos/sin f32 rounding).
time_sin <- diffuseR:::ltx23_get_timestep_embedding(
    timestep$mul(1000), 256L, flip_sin_to_cos = TRUE,
    downscale_freq_shift = 0)

out <- with_no_grad(m(
    hidden_states = latents, encoder_hidden_states = text_embeds,
    pooled_projections = pooled, timestep = timestep,
    image_rotary_emb = rope))

# ---- assemble the fixture: state dict (f32, contiguous) + inputs ----
sd <- m$state_dict()
save_list <- lapply(sd, function(t) t$to(dtype = torch_float32())$contiguous())
save_list[["input.latents"]] <- latents$contiguous()
save_list[["input.text_embeds"]] <- text_embeds$contiguous()
save_list[["input.pooled"]] <- pooled$contiguous()
save_list[["input.time_sin"]] <- time_sin$contiguous()
save_list[["input.cos"]] <- rope[[1]]$contiguous()
save_list[["input.sin"]] <- rope[[2]]$contiguous()
save_list[["input.timestep"]] <- timestep$contiguous()
save_list[["output"]] <- out$contiguous()

safetensors::safe_save_file(save_list, fixture)

cat(sprintf("fixture: %s (%.2f KB, %d state-dict keys)\n",
            fixture, file.size(fixture) / 1e3, length(sd)))
cat(sprintf("out shape %s  sd %.4f  range [%.3f, %.3f]\n",
            paste(dim(out), collapse = "x"), out$std()$item(),
            out$min()$item(), out$max()$item()))
