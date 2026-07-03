# NF4 quantization: round trip quality, linear parity, and resident
# transformer loading from a tiny official-named checkpoint.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

torch::torch_manual_seed(321)

# --- Quantize/dequantize round trip ------------------------------------------

w <- torch::torch_randn(64L, 128L) * 0.02
q <- ltx23_nf4_quantize(w)
expect_equal(as.integer(q$packed$shape), 64L * 128L / 2L)
expect_equal(as.integer(q$absmax$shape), 64L * 128L / 64L)
expect_equal(q$packed$dtype$.type(), "Byte")

w_rt <- ltx23_nf4_dequantize(q$packed, q$absmax, c(64L, 128L),
  dtype = torch::torch_float32())
expect_equal(as.integer(w_rt$shape), c(64L, 128L))
# NF4 round-trip error on Gaussian data: a few percent relative
rel <- as.numeric((w_rt - w)$abs()$mean() / w$abs()$mean())
expect_true(rel < 0.1)

# Values land exactly on table levels x absmax
blocks <- w_rt$reshape(c(-1L, 64L)) / q$absmax$unsqueeze(2L)
dists <- torch::torch_min(
  (blocks$flatten()$unsqueeze(2L) -
    torch::torch_tensor(diffuseR:::.ltx23_nf4_table)$unsqueeze(1L))$abs(),
  dim = 2L
)[[1]]
expect_true(as.numeric(dists$max()) < 1e-5)

# Chunked dequant is identical to single-shot
w_chunked <- ltx23_nf4_dequantize(q$packed, q$absmax, c(64L, 128L),
  dtype = torch::torch_float32(), chunk_elements = 512L)
expect_true(as.numeric((w_chunked - w_rt)$abs()$max()) == 0)

# --- nf4_linear parity -----------------------------------------------------------

lin <- torch::nn_linear(128L, 64L)
qw <- ltx23_nf4_quantize(lin$weight)
nf4 <- ltx23_nf4_linear(64L, 128L, bias = TRUE)
nf4$set_nf4_weight(qw$packed, qw$absmax)
torch::with_no_grad(nf4$bias$copy_(lin$bias))

x <- torch::torch_randn(5L, 128L)
torch::with_no_grad({
  y_ref <- lin(x)
  y_nf4 <- nf4(x)
})
rel_y <- as.numeric((y_nf4 - y_ref)$abs()$mean() / y_ref$abs()$mean())
expect_true(rel_y < 0.15)

# Buffers move with the module and survive dtype casts
nf4$to(dtype = torch::torch_float64())
expect_equal(nf4$weight_nf4$dtype$.type(), "Byte")

# --- Tiny transformer through the NF4 artifact ------------------------------------

tiny_cfg <- list(
  in_channels = 4L, out_channels = 4L,
  num_attention_heads = 2L, attention_head_dim = 8L,
  cross_attention_dim = 16L,
  audio_in_channels = 4L, audio_out_channels = 4L,
  audio_num_attention_heads = 2L, audio_attention_head_dim = 4L,
  audio_cross_attention_dim = 8L,
  num_layers = 1L
)
ref_model <- do.call(ltx23_transformer, tiny_cfg)
ref_model$eval()

to_official <- function(name) {
  name <- gsub("av_cross_attn_video_scale_shift", "av_ca_video_scale_shift_adaln_single", name, fixed = TRUE)
  name <- gsub("av_cross_attn_video_a2v_gate", "av_ca_a2v_gate_adaln_single", name, fixed = TRUE)
  name <- gsub("av_cross_attn_audio_scale_shift", "av_ca_audio_scale_shift_adaln_single", name, fixed = TRUE)
  name <- gsub("av_cross_attn_audio_v2a_gate", "av_ca_v2a_gate_adaln_single", name, fixed = TRUE)
  name <- gsub("video_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_video", name, fixed = TRUE)
  name <- gsub("audio_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_audio", name, fixed = TRUE)
  name <- gsub("prompt_adaln", "prompt_adaln_single", name, fixed = TRUE)
  name <- sub("^audio_time_embed\\.", "audio_adaln_single.", name)
  name <- sub("^time_embed\\.", "adaln_single.", name)
  name <- gsub("proj_in", "patchify_proj", name, fixed = TRUE)
  name <- gsub("norm_q", "q_norm", name, fixed = TRUE)
  name <- gsub("norm_k", "k_norm", name, fixed = TRUE)
  paste0("model.diffusion_model.", name)
}

params <- ref_model$named_parameters()
tensors <- list()
for (name in names(params)) {
  tensors[[to_official(name)]] <- params[[name]]$detach()
}
src <- tempfile(fileext = ".safetensors")
on.exit(unlink(src), add = TRUE)
safetensors::safe_save_file(tensors, src, metadata = list(model_version = "2.3.0"))

nf4_dir <- tempfile("nf4_")
on.exit(unlink(nf4_dir, recursive = TRUE), add = TRUE)
manifest <- ltx23_quantize_nf4(src, nf4_dir, verbose = FALSE)
expect_equal(manifest$format, "nf4")
expect_equal(manifest$nf4_cast, sum(ltx23_is_fp8_cast_key(names(params))))

ckpt <- ltx23_open_fp8_checkpoint(nf4_dir)
expect_equal(ckpt$format, "nf4")

nf4_model <- do.call(ltx23_load_transformer_nf4, c(
  list(ckpt = ckpt, device = "cpu", verbose = FALSE),
  tiny_cfg
))
expect_inherits(nf4_model$transformer_blocks[[1]]$attn1$to_q, "ltx23_nf4_linear")

# Forward runs in bf16 and stays finite
torch::torch_manual_seed(5)
bf <- torch::torch_bfloat16()
torch::with_no_grad({
  out <- nf4_model(
    hidden_states = torch::torch_randn(1L, 24L, 4L, dtype = bf),
    audio_hidden_states = torch::torch_randn(1L, 5L, 4L, dtype = bf),
    encoder_hidden_states = torch::torch_randn(1L, 7L, 16L, dtype = bf),
    audio_encoder_hidden_states = torch::torch_randn(1L, 7L, 8L, dtype = bf),
    timestep = torch::torch_tensor(700, dtype = torch::torch_float32()),
    sigma = torch::torch_tensor(700, dtype = torch::torch_float32()),
    num_frames = 2L, height = 3L, width = 4L, audio_num_frames = 5L,
    use_cross_timestep = TRUE
  )
})
expect_true(as.logical(out$sample$isfinite()$all()$item()))
expect_true(as.logical(out$audio_sample$isfinite()$all()$item()))
