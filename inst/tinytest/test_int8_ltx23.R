# INT8 quantization: round trip quality, linear parity, and streamed
# transformer loading from a tiny official-named checkpoint.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

torch::torch_manual_seed(321)

# --- Quantize round trip -----------------------------------------------------------

w <- torch::torch_randn(64L, 128L) * 0.02
q <- ltx23_int8_quantize(w)
expect_equal(as.integer(q$weight_int8$shape), c(64L, 128L))
expect_equal(as.integer(q$scale$shape), 64L)
expect_equal(q$weight_int8$dtype$.type(), "Char")

w_rt <- q$weight_int8$to(dtype = torch::torch_float32()) * q$scale$unsqueeze(2L)
# Symmetric rounding: every element within half a step of its row scale
worst <- as.numeric(((w_rt - w)$abs() / q$scale$unsqueeze(2L))$max())
expect_true(worst <= 0.5 + 1e-4)
# Per-channel int8 is ~20x tighter than NF4 on Gaussian data
rel <- as.numeric((w_rt - w)$abs()$mean() / w$abs()$mean())
expect_true(rel < 0.01)

# --- int8_linear parity ------------------------------------------------------------

lin <- torch::nn_linear(128L, 64L)
qw <- ltx23_int8_quantize(lin$weight)
i8 <- ltx23_int8_linear(64L, 128L, bias = TRUE)
i8$set_int8_weight(qw$weight_int8, qw$scale)
torch::with_no_grad(i8$bias$copy_(lin$bias))

x <- torch::torch_randn(5L, 128L)
torch::with_no_grad({
  y_ref <- lin(x)
  y_i8 <- i8(x)
})
rel_y <- as.numeric((y_i8 - y_ref)$abs()$mean() / y_ref$abs()$mean())
expect_true(rel_y < 0.02)

# Exact against the reconstructed weight (same math, buffered)
w_recon <- qw$weight_int8$to(dtype = torch::torch_float32()) *
  qw$scale$unsqueeze(2L)
torch::with_no_grad({
  y_recon <- torch::nnf_linear(x, w_recon, lin$bias)
})
expect_true(as.numeric((y_i8 - y_recon)$abs()$max()) < 1e-6)

# Weight fields stay int8 through module dtype casts
i8$to(dtype = torch::torch_float64())
expect_equal(i8$weight_int8$dtype$.type(), "Char")

# --- Tiny transformer through the int8 artifact ------------------------------------

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

int8_dir <- tempfile("int8_")
on.exit(unlink(int8_dir, recursive = TRUE), add = TRUE)
manifest <- ltx23_quantize_int8(src, int8_dir, verbose = FALSE)
expect_equal(manifest$format, "int8")
expect_equal(manifest$int8_cast, sum(ltx23_is_fp8_cast_key(names(params))))

ckpt <- ltx23_open_fp8_checkpoint(int8_dir)
expect_equal(ckpt$format, "int8")

i8_model <- do.call(ltx23_load_transformer_int8, c(
  list(ckpt = ckpt, device = "cpu", pin = FALSE, verbose = FALSE),
  tiny_cfg
))
expect_inherits(i8_model$transformer_blocks[[1]]$attn1$to_q, "ltx23_int8_linear")

# Forward runs in bf16 and stays finite
torch::torch_manual_seed(5)
bf <- torch::torch_bfloat16()
torch::with_no_grad({
  out <- i8_model(
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
