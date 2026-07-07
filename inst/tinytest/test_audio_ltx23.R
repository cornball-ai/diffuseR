# Parity tests for the LTX-2.3 audio VAE decoder and vocoder port
# against diffusers reference fixtures (tools/gen_fixtures_audio.py).

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

fixture_path <- system.file("tinytest", "fixtures", "audio_ltx23.safetensors",
  package = "diffuseR")
if (fixture_path == "") fixture_path <- "fixtures/audio_ltx23.safetensors"
if (!file.exists(fixture_path)) exit_file("audio fixtures missing")

fx <- safetensors::safe_load_file(fixture_path, framework = "torch")

max_abs_diff <- function(a, b) {
  as.numeric(torch::torch_max(torch::torch_abs(
    a$to(dtype = torch::torch_float32()) - b$to(dtype = torch::torch_float32())
  )))
}

load_group <- function(module, prefix) {
  keys <- grep(paste0("^", prefix, "\\."), names(fx), value = TRUE)
  w <- fx[keys]
  names(w) <- sub(paste0("^", prefix, "\\."), "", keys)
  dests <- c(module$named_parameters(), module$named_buffers())
  if (!setequal(names(w), names(dests))) {
    stop(
      prefix, ": name mismatch. missing dest: ",
      paste(utils::head(setdiff(names(w), names(dests)), 4), collapse = ", "),
      " | unfilled: ",
      paste(utils::head(setdiff(names(dests), names(w)), 4), collapse = ", ")
    )
  }
  torch::with_no_grad({
    for (name in names(w)) dests[[name]]$copy_(w[[name]])
  })
  module$eval()
  module
}

# --- Audio VAE decoder --------------------------------------------------------

adec <- load_group(
  ltx23_audio_decoder(
    base_channels = 8L, output_channels = 2L, num_res_blocks = 1L,
    latent_channels = 4L, ch_mult = c(1L, 2L), mel_bins = 8L
  ),
  "adec"
)
torch::with_no_grad(out_adec <- adec(fx$adec_x))
expect_equal(as.integer(out_adec$shape), c(1L, 2L, 17L, 8L))# 5 * 4 - 3 frames
expect_true(max_abs_diff(out_adec, fx$adec_out) < 1e-5)

# --- Kaiser sinc filter + window ------------------------------------------------

expect_true(max_abs_diff(
  ltx23_kaiser_sinc_filter1d(0.25, 0.3, 12L), fx$kaiser_filt_12
) < 1e-5)
expect_true(max_abs_diff(
  ltx23_kaiser_sinc_filter1d(0.1, 0.12, 13L), fx$kaiser_filt_13
) < 1e-5)
expect_true(max_abs_diff(
  diffuseR:::.ltx23_kaiser_window(12L, 4.7), fx$kaiser_window_12
) < 1e-6)

# --- Up/DownSample1d ---------------------------------------------------------------

ds1 <- ltx23_downsample1d(ratio = 2L, kernel_size = 12L)
torch::with_no_grad(out_down <- ds1(fx$w_x))
expect_true(max_abs_diff(out_down, fx$w_down) < 1e-5)

us1 <- ltx23_upsample1d(ratio = 2L, kernel_size = 12L)
torch::with_no_grad(out_up <- us1(fx$w_x))
expect_equal(as.integer(out_up$shape), as.integer(fx$w_up$shape))
expect_true(max_abs_diff(out_up, fx$w_up) < 1e-5)

ush <- ltx23_upsample1d(ratio = 4L, window_type = "hann")
torch::with_no_grad(out_uph <- ush(fx$w_x))
expect_equal(as.integer(out_uph$shape), as.integer(fx$w_up_hann$shape))
expect_true(max_abs_diff(out_uph, fx$w_up_hann) < 1e-5)

# --- SnakeBeta -----------------------------------------------------------------------

sb <- ltx23_snake_beta(3L)
torch::with_no_grad({
  sb$alpha$copy_(fx$sb.alpha)
  sb$beta$copy_(fx$sb.beta)
  out_sb <- sb(fx$w_x)
})
expect_true(max_abs_diff(out_sb, fx$sb_out) < 1e-5)

# --- Tiny vocoder stage -----------------------------------------------------------------

voc <- load_group(
  ltx23_vocoder(
    in_channels = 8L, hidden_channels = 16L, out_channels = 2L,
    upsample_kernel_sizes = c(4L, 4L), upsample_factors = c(2L, 2L),
    resnet_kernel_sizes = c(3L), resnet_dilations = list(c(1L, 3L)),
    final_bias = FALSE
  ),
  "voc"
)
torch::with_no_grad(out_voc <- voc(fx$voc_x))
expect_equal(as.integer(out_voc$shape), c(1L, 2L, 24L))
expect_true(max_abs_diff(out_voc, fx$voc_out) < 1e-4)

# --- Tiny BWE wrapper --------------------------------------------------------------------

bwe <- load_group(
  ltx23_vocoder_with_bwe(
    in_channels = 8L, hidden_channels = 16L, out_channels = 2L,
    upsample_kernel_sizes = c(4L, 4L), upsample_factors = c(2L, 2L),
    resnet_kernel_sizes = c(3L), resnet_dilations = list(c(1L, 3L)),
    bwe_in_channels = 16L, bwe_hidden_channels = 8L,
    bwe_upsample_kernel_sizes = c(8L, 4L), bwe_upsample_factors = c(4L, 2L),
    bwe_resnet_kernel_sizes = c(3L), bwe_resnet_dilations = list(c(1L, 3L)),
    filter_length = 8L, hop_length = 2L, window_length = 8L,
    num_mel_channels = 8L,
    input_sampling_rate = 100L, output_sampling_rate = 400L
  ),
  "bwe"
)
torch::with_no_grad(out_bwe <- bwe(fx$voc_x))
expect_equal(as.integer(out_bwe$shape), c(1L, 2L, 96L))
expect_true(max_abs_diff(out_bwe, fx$bwe_out) < 1e-4)

# --- Key mappers ----------------------------------------------------------------------------

expect_equal(
  ltx23_map_vocoder_key("vocoder.vocoder.conv_pre.weight"),
  "vocoder.conv_in.weight"
)
expect_equal(
  ltx23_map_vocoder_key("vocoder.vocoder.resblocks.0.acts1.1.downsample.lowpass.filter"),
  "vocoder.resnets.0.acts1.1.downsample.filter"
)
expect_equal(
  ltx23_map_vocoder_key("vocoder.vocoder.ups.2.weight"),
  "vocoder.upsamplers.2.weight"
)
expect_equal(
  ltx23_map_vocoder_key("vocoder.bwe_generator.act_post.act.alpha"),
  "bwe_generator.act_out.act.alpha"
)
expect_equal(
  ltx23_map_vocoder_key("vocoder.mel_stft.stft_fn.forward_basis"),
  "mel_stft.stft_fn.forward_basis"
)
expect_equal(
  ltx23_map_audio_vae_key("audio_vae.decoder.up.1.upsample.conv.conv.weight"),
  "decoder.up.1.upsample.conv.conv.weight"
)
expect_equal(
  ltx23_map_audio_vae_key("audio_vae.per_channel_statistics.mean-of-means"),
  "latents_mean"
)
expect_equal(
  ltx23_map_audio_vae_key("audio_vae.encoder.conv_in.conv.weight"),
  "encoder.conv_in.conv.weight"
)
expect_equal(
  ltx23_map_audio_vae_key("audio_vae.encoder.down.1.downsample.conv.weight"),
  "encoder.down.1.downsample.conv.weight"
)

# --- Encoder: shapes and key-name census -------------------------------------------

enc <- ltx23_audio_encoder(
  base_channels = 8L, in_channels = 2L, num_res_blocks = 2L,
  latent_channels = 4L, ch_mult = c(1L, 2L)
)
enc$eval()
mel_in <- torch::torch_randn(1L, 2L, 12L, 8L)
torch::with_no_grad(out_enc <- enc(mel_in))
# One downsample level: time and mel halved; moments = 2 * latent
expect_equal(as.integer(out_enc$shape), c(1L, 8L, 6L, 4L))

# Parameter names line up with the checkpoint layout
pn <- names(enc$named_parameters())
expect_true("conv_in.conv.weight" %in% pn)
expect_true("down.0.block.0.conv1.conv.weight" %in% pn)
expect_true("down.0.block.1.conv2.conv.bias" %in% pn)
expect_true("down.0.downsample.conv.weight" %in% pn)
expect_true("down.1.block.0.nin_shortcut.conv.weight" %in% pn)
expect_true("mid.block_1.conv1.conv.weight" %in% pn)
expect_true("mid.block_2.conv2.conv.bias" %in% pn)
expect_true("conv_out.conv.weight" %in% pn)
# Pixel norms are parameterless: exactly the conv tensors
expect_true(all(grepl("conv", pn)))

# encode() through the vae wrapper returns mean/logvar of latent size
avae <- ltx23_audio_vae(
  base_channels = 8L, output_channels = 2L, num_res_blocks = 1L,
  latent_channels = 4L, ch_mult = c(1L, 2L), mel_bins = 8L
)
avae$eval()
torch::with_no_grad(m <- avae$encode(torch::torch_randn(1L, 2L, 12L, 8L)))
expect_equal(as.integer(m$mean$shape)[2], 4L)
expect_equal(as.integer(m$logvar$shape)[2], 4L)
