# LTX-2.5 transformer deltas on the shared 2.3 module tree: the video
# feed-forward has no biases and a learned keyframe marker joins
# proj_in. Both flags come from the checkpoint's embedded config.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)

# --- Config flags ----------------------------------------------------------------

expect_identical(ltx23_dit_config_args(NULL), list())
# A 2.3 config carries none of the keys: the defaults apply
expect_identical(ltx23_dit_config_args(list(transformer = list(num_layers = 48L))),
                 list())
cfg25 <- list(transformer = list(ff_bias = FALSE,
                                 use_keyframes_abs_pos_embedding = TRUE))
expect_identical(ltx23_dit_config_args(cfg25),
                 list(ff_bias = FALSE, use_keyframes_abs_pos_embedding = TRUE))

# --- Module shape ----------------------------------------------------------------

tiny_cfg <- list(
  in_channels = 4L, out_channels = 4L,
  num_attention_heads = 2L, attention_head_dim = 8L,
  cross_attention_dim = 16L,
  audio_in_channels = 4L, audio_out_channels = 4L,
  audio_num_attention_heads = 2L, audio_attention_head_dim = 4L,
  audio_cross_attention_dim = 8L,
  num_layers = 1L
)
torch::torch_manual_seed(3)
m23 <- do.call(ltx23_transformer, tiny_cfg)
m25 <- do.call(ltx23_transformer, c(tiny_cfg, ltx23_dit_config_args(cfg25)))
blk <- m25$transformer_blocks[[1]]
expect_null(blk$ff$net[[1]]$proj$bias)
expect_null(blk$ff$net[[3]]$bias)
expect_false(is.null(blk$audio_ff$net[[1]]$proj$bias))
expect_null(m23$keyframes_abs_pos_embedding)
expect_equal(as.integer(m25$keyframes_abs_pos_embedding$shape), c(1L, 16L))
# Exactly the two video FF biases and the marker differ, as in the real
# checkpoints (2 x 48 biases fewer, one marker more)
n23 <- names(m23$named_parameters())
n25 <- names(m25$named_parameters())
expect_equal(sort(setdiff(n23, n25)),
             sort(c("transformer_blocks.0.ff.net.0.proj.bias",
                    "transformer_blocks.0.ff.net.2.bias")))
expect_equal(setdiff(n25, n23), "keyframes_abs_pos_embedding")

# --- Keyframe marker -------------------------------------------------------------

m25$eval()
run <- function(model, ...) {
  torch::torch_manual_seed(5)
  torch::with_no_grad({
    model(
      hidden_states = torch::torch_randn(1L, 24L, 4L),
      audio_hidden_states = torch::torch_randn(1L, 5L, 4L),
      encoder_hidden_states = torch::torch_randn(1L, 7L, 16L),
      audio_encoder_hidden_states = torch::torch_randn(1L, 7L, 8L),
      timestep = torch::torch_tensor(700),
      sigma = torch::torch_tensor(700),
      num_frames = 2L, height = 3L, width = 4L, audio_num_frames = 5L,
      use_cross_timestep = TRUE, ...
    )
  })
}
max_abs_diff <- function(a, b) as.numeric((a - b)$abs()$max())

torch::with_no_grad(m25$keyframes_abs_pos_embedding$fill_(0.5))
first_frame <- torch::torch_zeros(1L, 24L, 1L)
first_frame[1, 1:12, 1] <- 1
base <- run(m25)
expect_true(max_abs_diff(run(m25, video_keyframes_mask = first_frame)$sample,
                         base$sample) > 1e-4)
# An all-zero mask marks nothing
expect_true(max_abs_diff(run(m25, video_keyframes_mask = torch::torch_zeros(1L, 24L, 1L))$sample,
                         base$sample) < 1e-7)

# --- 2.5 checkpoint through the NF4 and fp8 artifacts ---------------------------

if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

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

params <- m25$named_parameters()

# Every temp file and directory is created and removed inside its own
# local() scope, so a failing expectation or loader error still cleans up
src_seen <- local({
  tensors <- list()
  for (name in names(params)) {
    tensors[[to_official(name)]] <- params[[name]]$detach()$clone()
  }
  src <- tempfile(fileext = ".safetensors")
  on.exit(unlink(src), add = TRUE)
  safetensors::safe_save_file(tensors, src, metadata = list(
    model_version = "2.5.0",
    config = as.character(jsonlite::toJSON(cfg25, auto_unbox = TRUE))
  ))

  # The single-file opener accepts 2.5 and parses the flags back out
  ckpt25 <- ltx23_open_checkpoint(src)
  expect_equal(ckpt25$version, "2.5.0")
  expect_identical(ltx23_dit_config_args(ckpt25$config),
                   list(ff_bias = FALSE, use_keyframes_abs_pos_embedding = TRUE))

  # Dense (bf16/fp32) loader, the ltx23_load_pipeline() path. No ff_bias
  # argument: it must arrive through the embedded config, and an
  # incomplete load stops inside the loader. Loading at float32 makes
  # the round trip exact.
  d <- do.call(diffuseR:::.ltx23_load_transformer,
               c(list(ckpt = ckpt25, dtype = torch::torch_float32(),
                      verbose = FALSE), tiny_cfg))
  expect_null(d$transformer_blocks[[1]]$ff$net[[3]]$bias)
  loaded <- d$named_parameters()
  expect_equal(sort(names(loaded)), sort(names(params)))
  same <- vapply(names(params), function(n) {
    as.logical(torch::torch_equal(loaded[[n]], params[[n]]))
  }, logical(1))
  expect_true(all(same))

  for (fmt in c("nf4", "fp8")) local({
    dir <- tempfile(paste0(fmt, "_"))
    on.exit(unlink(dir, recursive = TRUE), add = TRUE)
    quantize <- if (fmt == "nf4") ltx23_quantize_nf4 else ltx23_quantize_fp8
    load <- if (fmt == "nf4") ltx23_load_transformer_nf4 else ltx23_load_transformer_fp8
    manifest <- quantize(src, dir, verbose = FALSE)
    expect_equal(manifest[[paste0(fmt, "_cast")]],
                 sum(ltx23_is_fp8_cast_key(names(params))), info = fmt)

    # The flags reach the quantized loaders through the manifest config
    q <- do.call(load, c(list(ckpt = ltx23_open_fp8_checkpoint(dir),
                              device = "cpu", verbose = FALSE), tiny_cfg))
    qblk <- q$transformer_blocks[[1]]
    expect_null(qblk$ff$net[[3]]$bias, info = fmt)
    expect_false(is.null(qblk$audio_ff$net[[3]]$bias), info = fmt)
    expect_equal(as.numeric(q$keyframes_abs_pos_embedding$float()$mean()), 0.5,
                 info = fmt)

    q$to(dtype = torch::torch_float32())
    out <- run(q, video_keyframes_mask = first_frame)
    expect_true(as.logical(out$sample$isfinite()$all()$item()), info = fmt)
  })
  src
})

# The scopes above removed their fixtures
expect_false(file.exists(src_seen))
