# End-to-end plumbing test for the LTX-2.3 pipeline with tiny
# random-weight components (no model downloads; CPU-safe).

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)

torch::torch_manual_seed(99)

# Tiny components wired to the pipeline's structural constants:
# 128 video latent channels, 8 audio channels x 16 latent mel bins.
transformer <- ltx23_transformer(
  in_channels = 128L, out_channels = 128L,
  num_attention_heads = 2L, attention_head_dim = 8L,
  cross_attention_dim = 16L,
  audio_in_channels = 128L, audio_out_channels = 128L,
  audio_num_attention_heads = 2L, audio_attention_head_dim = 4L,
  audio_cross_attention_dim = 8L,
  num_layers = 1L
)
transformer$eval()

connectors <- ltx23_text_connectors(
  caption_channels = 8L, text_proj_in_factor = 3L,
  video_connector_num_attention_heads = 2L,
  video_connector_attention_head_dim = 8L,
  video_connector_num_layers = 1L,
  video_connector_num_learnable_registers = 4L,
  audio_connector_num_attention_heads = 2L,
  audio_connector_attention_head_dim = 4L,
  audio_connector_num_layers = 1L,
  audio_connector_num_learnable_registers = 4L,
  video_hidden_dim = 16L, audio_hidden_dim = 8L
)
connectors$eval()

vae <- ltx23_video_vae(
  latent_channels = 128L,
  block_out_channels = c(8L, 8L, 8L, 8L),
  decoder_block_out_channels = c(4L, 8L, 8L, 16L),
  layers_per_block = c(1L, 1L, 1L, 1L, 1L),
  decoder_layers_per_block = c(1L, 1L, 1L, 1L, 1L)
)
vae$eval()

audio_vae <- ltx23_audio_vae(
  base_channels = 128L, ch_mult = c(1L, 1L), num_res_blocks = 1L,
  latent_channels = 8L, mel_bins = 64L
)
audio_vae$eval()

vocoder <- ltx23_vocoder_with_bwe(
  in_channels = 128L, hidden_channels = 16L,
  upsample_kernel_sizes = c(4L, 4L), upsample_factors = c(2L, 2L),
  resnet_kernel_sizes = c(3L), resnet_dilations = list(c(1L, 3L)),
  bwe_in_channels = 128L, bwe_hidden_channels = 8L,
  bwe_upsample_kernel_sizes = c(8L, 4L), bwe_upsample_factors = c(4L, 2L),
  bwe_resnet_kernel_sizes = c(3L), bwe_resnet_dilations = list(c(1L, 3L)),
  filter_length = 8L, hop_length = 2L, window_length = 8L,
  num_mel_channels = 64L,
  input_sampling_rate = 100L, output_sampling_rate = 400L
)
vocoder$eval()

pipe <- structure(
  list(
    transformer = transformer,
    connectors = connectors,
    vae = vae,
    audio_vae = audio_vae,
    vocoder = vocoder
  ),
  class = "ltx23_pipeline"
)

stub_embeds <- list(
  prompt_embeds = torch::torch_randn(1L, 8L, 8L, 3L),
  prompt_attention_mask = torch::torch_ones(1L, 8L)
)

res <- txt2vid_ltx2(
  prompt = "a tiny test",
  pipeline = pipe,
  prompt_embeds = stub_embeds,
  width = 64L, height = 64L, num_frames = 9L, frame_rate = 24,
  seed = 7L,
  device = "cpu",
  dtype = "float32",
  verbose = FALSE
)

# Video: [frames, height, width, 3] in [0, 1]
expect_equal(dim(res$video), c(9L, 64L, 64L, 3L))
expect_true(all(res$video >= 0 & res$video <= 1))
expect_true(all(is.finite(res$video)))

# Audio: [2, samples] in [-1, 1]; latent L = round(9/24*25) = 9 ->
# mel frames 4*9-3 = 33 (padded to hop multiple 34) -> vocoder x4 -> BWE x4
expect_equal(nrow(res$audio), 2L)
expect_true(ncol(res$audio) > 100L)
expect_true(all(abs(res$audio) <= 1))
expect_true(all(is.finite(res$audio)))
expect_equal(res$sample_rate, 48000L)

# Guardrails
expect_error(
  txt2vid_ltx2("x", pipe, prompt_embeds = stub_embeds, guidance_scale = 3,
    device = "cpu", dtype = "float32"),
  pattern = "guidance_scale"
)
expect_error(
  txt2vid_ltx2("x", pipe, prompt_embeds = stub_embeds, width = 50L,
    device = "cpu", dtype = "float32"),
  pattern = "multiples of 32"
)
expect_error(
  txt2vid_ltx2("x", pipe, prompt_embeds = stub_embeds, num_frames = 10L,
    device = "cpu", dtype = "float32"),
  pattern = "8k"
)

# WAV writer round trip (header + size)
wav_path <- tempfile(fileext = ".wav")
write_wav(res$audio, wav_path, sample_rate = 48000L)
expect_true(file.exists(wav_path))
expect_equal(readBin(wav_path, "raw", 4), charToRaw("RIFF"))
expect_equal(file.size(wav_path), 44 + 2 * 2 * ncol(res$audio))
unlink(wav_path)

# MP4 mux when av is available
if (requireNamespace("av", quietly = TRUE)) {
  mp4 <- tempfile(fileext = ".mp4")
  save_video_ltx23(res$video, mp4, fps = 24, audio = res$audio,
    sample_rate = 48000L, verbose = FALSE)
  expect_true(file.exists(mp4) && file.size(mp4) > 0)
  unlink(mp4)
}

# --- Prefix conditioning smoke tests -------------------------------------------------

# i2v: start image conditions frame 0; pipeline runs end to end
start_img <- array(runif(64 * 64 * 3), dim = c(64L, 64L, 3L))
res_i2v <- txt2vid_ltx2(
  prompt = "tiny i2v",
  pipeline = pipe,
  prompt_embeds = stub_embeds,
  width = 64L, height = 64L, num_frames = 9L, frame_rate = 24,
  seed = 7L, device = "cpu", dtype = "float32",
  image = start_img,
  decode_audio = FALSE,
  verbose = FALSE
)
expect_equal(dim(res_i2v$video), c(9L, 64L, 64L, 3L))
expect_true(all(is.finite(res_i2v$video)))

# Same seed without conditioning gives a different video (mask engaged)
res_t2v <- txt2vid_ltx2(
  prompt = "tiny i2v",
  pipeline = pipe,
  prompt_embeds = stub_embeds,
  width = 64L, height = 64L, num_frames = 9L, frame_rate = 24,
  seed = 7L, device = "cpu", dtype = "float32",
  decode_audio = FALSE,
  verbose = FALSE
)
expect_true(max(abs(res_i2v$video - res_t2v$video)) > 1e-4)

# Continuation: 9-frame tail array as the frozen prefix
tail_arr <- array(runif(9 * 64 * 64 * 3), dim = c(9L, 64L, 64L, 3L))
res_cont <- txt2vid_ltx2(
  prompt = "tiny continuation",
  pipeline = pipe,
  prompt_embeds = stub_embeds,
  width = 64L, height = 64L, num_frames = 17L, frame_rate = 24,
  seed = 7L, device = "cpu", dtype = "float32",
  condition_video = tail_arr, conditioning_frames = 9L,
  decode_audio = FALSE,
  verbose = FALSE
)
expect_equal(dim(res_cont$video), c(17L, 64L, 64L, 3L))
expect_true(all(is.finite(res_cont$video)))

# Guardrails
expect_error(
  txt2vid_ltx2("x", pipe, prompt_embeds = stub_embeds,
    image = start_img, condition_video = tail_arr,
    device = "cpu", dtype = "float32"),
  pattern = "not both"
)
