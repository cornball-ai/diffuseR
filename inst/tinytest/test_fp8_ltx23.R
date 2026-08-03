# FP8 quantization round trip: quantize a tiny official-named checkpoint,
# reload with fp8 linears, and compare against the bf16 original.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}
# fp8 needs the F8-capable safetensors build
# Same gate the quantizer uses, so the diffuseR.st_caps override
# reaches this guard too; a private probe would disagree with it.
f8_ok <- diffuseR:::.st_can_write("float8_e4m3fn")
if (!f8_ok) exit_file("safetensors build lacks F8 support")

library(diffuseR)

torch::torch_manual_seed(123)

tiny_cfg <- list(
  in_channels = 4L, out_channels = 4L,
  num_attention_heads = 2L, attention_head_dim = 8L,
  cross_attention_dim = 16L,
  audio_in_channels = 4L, audio_out_channels = 4L,
  audio_num_attention_heads = 2L, audio_attention_head_dim = 4L,
  audio_cross_attention_dim = 8L,
  num_layers = 2L
)
ref_model <- do.call(ltx23_transformer, tiny_cfg)
ref_model$eval()

# Reverse the diffusers renames to synthesize an official-named checkpoint
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
# Round trip through the mapper must be lossless
expect_equal(
  sort(vapply(names(tensors), ltx23_map_dit_key, character(1), USE.NAMES = FALSE)),
  sort(names(params))
)

src <- tempfile(fileext = ".safetensors")
on.exit(unlink(src), add = TRUE)
safetensors::safe_save_file(tensors, src, metadata = list(model_version = "2.3.0"))

# --- Quantize -----------------------------------------------------------------

fp8_dir <- tempfile("fp8_")
on.exit(unlink(fp8_dir, recursive = TRUE), add = TRUE)
manifest <- ltx23_quantize_fp8(src, fp8_dir, verbose = FALSE)
expect_true(file.exists(file.path(fp8_dir, "manifest.json")))
expect_true(manifest$fp8_cast > 0)
# 2 layers x (3 video self + 3 audio self + 3 x-attn x 2 + 4 a2v/v2a) qkv/out
# + 4 ff projections: just assert the exact official policy count
n_expected <- sum(ltx23_is_fp8_cast_key(names(params)))
expect_equal(manifest$fp8_cast, n_expected)

# Re-running is a no-op (skip-if-exists)
m2 <- ltx23_quantize_fp8(src, fp8_dir, verbose = FALSE)
expect_equal(m2$shards, manifest$shards)

# --- Load fp8 + forward parity ---------------------------------------------------

ckpt <- ltx23_open_fp8_checkpoint(fp8_dir)
expect_equal(ckpt$version, "2.3.0")

old_opt <- getOption("diffuseR.use_fp8")
fp8_model <- do.call(ltx23_load_transformer_fp8, c(
  list(ckpt = ckpt, device = "cpu", pin = FALSE, verbose = FALSE),
  tiny_cfg
))
options(diffuseR.use_fp8 = old_opt)

# Cast linears were swapped
expect_inherits(fp8_model$transformer_blocks[[1]]$attn1$to_q, "ltx23_fp8_linear")
expect_equal(
  fp8_model$transformer_blocks[[1]]$attn1$to_q$weight_fp8$dtype$.type(),
  "Float8_e4m3fn"
)
# Non-cast params match exactly (bf16 storage of an fp32 original)
expect_true(as.numeric(torch::torch_max(torch::torch_abs(
  fp8_model$proj_in$weight$to(dtype = torch::torch_float32()) -
    ref_model$proj_in$weight
))) < 0.01)

# Dequantized weights reconstruct the originals within fp8 e4m3 tolerance
# (a random tiny network amplifies chaotically, so end-to-end output error
# is not a meaningful metric; real-weight quality shows in the E2E render)
fp8_q <- fp8_model$transformer_blocks[[1]]$attn1$to_q
deq <- fp8_q$weight_fp8$to(dtype = torch::torch_float32()) *
  fp8_q$weight_scale
orig <- ref_model$transformer_blocks[[1]]$attn1$to_q$weight
rel_w <- as.numeric((deq - orig)$abs()$mean() / orig$abs()$mean())
expect_true(rel_w < 0.05)

# The fp8_linear layer itself matches a plain linear closely
x_probe <- torch::torch_randn(3L, 16L)
torch::with_no_grad({
  y_fp8 <- fp8_q(x_probe$to(dtype = torch::torch_bfloat16()))
  y_ref <- torch::nnf_linear(x_probe, orig,
    ref_model$transformer_blocks[[1]]$attn1$to_q$bias)
})
rel_y <- as.numeric(
  (y_fp8$to(dtype = torch::torch_float32()) - y_ref)$abs()$mean() /
    y_ref$abs()$mean()
)
expect_true(rel_y < 0.1)

# Full fp8 model forward runs in bf16 and stays finite
run <- function(model, dtype) {
  torch::torch_manual_seed(5)
  hidden <- torch::torch_randn(1L, 24L, 4L, dtype = dtype)
  audio <- torch::torch_randn(1L, 5L, 4L, dtype = dtype)
  enc <- torch::torch_randn(1L, 7L, 16L, dtype = dtype)
  aenc <- torch::torch_randn(1L, 7L, 8L, dtype = dtype)
  t <- torch::torch_tensor(700, dtype = torch::torch_float32())
  torch::with_no_grad({
    model(
      hidden_states = hidden, audio_hidden_states = audio,
      encoder_hidden_states = enc, audio_encoder_hidden_states = aenc,
      timestep = t, sigma = t,
      num_frames = 2L, height = 3L, width = 4L, audio_num_frames = 5L,
      use_cross_timestep = TRUE
    )
  })
}
fp8_out <- run(fp8_model, torch::torch_bfloat16())
expect_equal(as.integer(fp8_out$sample$shape), c(1L, 24L, 4L))
expect_true(as.logical(fp8_out$sample$isfinite()$all()$item()))
expect_true(as.logical(fp8_out$audio_sample$isfinite()$all()$item()))

# --- attn chunking + memory profile -------------------------------------------------

ltx23_set_attn_chunk(fp8_model, 8L)
expect_equal(fp8_model$transformer_blocks[[1]]$attn2$attn_chunk, 8L)
ltx23_set_attn_chunk(fp8_model, NULL)
expect_null(fp8_model$transformer_blocks[[1]]$attn2$attn_chunk)

prof <- ltx23_memory_profile(vram_gb = 16)
expect_equal(prof$name, "high")
expect_true(is.integer(prof$attn_chunk) || is.null(prof$attn_chunk))
expect_equal(prof$precision, "nf4")
expect_equal(ltx23_memory_profile(vram_gb = 4)$name, "cpu_only")
