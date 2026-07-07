# Audio conditioning frontend (R/audio_encode_ltx23.R): STFT/mel basis
# constructors, mel geometry, and encode length math.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)
torch::torch_manual_seed(31)

# --- Basis constructors ---------------------------------------------------------------

fb <- diffuseR:::.ltx23_stft_basis(64L)
expect_equal(as.integer(fb$shape), c(66L, 1L, 64L))
# Row 0 is the DC row: pure window (cos(0) = 1)
w <- diffuseR:::.ltx23_hann(64L)
expect_true(as.numeric((fb[1, 1, ] - torch::torch_tensor(w))$abs()$max()) < 1e-6)
# The imaginary DC row is all zeros (-sin(0))
expect_true(as.numeric(fb[34, 1, ]$abs()$max()) < 1e-6)

mel <- diffuseR:::.ltx23_mel_filterbank(16000, 512L, 64L, 0, 8000)
expect_equal(as.integer(mel$shape), c(64L, 257L))
# Every filter is nonnegative with positive mass
expect_true(as.numeric(mel$min()) >= 0)
expect_true(all(as.numeric(mel$sum(dim = 2L)) > 0))

# --- Frontend geometry ---------------------------------------------------------------

frontend <- ltx23_audio_mel_frontend(filter_length = 64L, hop_length = 16L,
  n_mels = 8L, sample_rate = 1600L, fmin = 0, fmax = 800)
x <- torch::torch_randn(2L, 1L, 320L)
torch::with_no_grad(m <- frontend(x))
expect_equal(as.integer(m$shape)[1:2], c(2L, 8L))
expect_true(all(is.finite(as.numeric(m$min()))))
# Log-clamped: never below log(1e-5)
expect_true(as.numeric(m$min()) >= log(1e-5) - 1e-4)

# --- Encode length math ---------------------------------------------------------------

# base_channels 128 so the stats buffer matches the packed feature dim
# (latent_channels * latent_mel_bins = 128), as in the real checkpoint
avae <- ltx23_audio_vae(
  base_channels = 128L, output_channels = 2L, num_res_blocks = 1L,
  latent_channels = 8L, ch_mult = c(1L, 1L, 1L), mel_bins = 64L
)
avae$eval()
small_frontend <- ltx23_audio_mel_frontend()
wav <- matrix(runif(2L * 8000L, -0.5, 0.5), nrow = 2L)
lat <- ltx23_encode_audio(avae, wav, audio_num_frames = 9L,
  frontend = small_frontend)
# Packed [1, L, latent_channels * latent_mel_bins]; L == audio_num_frames
expect_equal(as.integer(lat$shape), c(1L, 9L, 8L * 16L))

# Longer target than the audio covers: right-padded, still exact length
lat2 <- ltx23_encode_audio(avae, wav, audio_num_frames = 25L,
  frontend = small_frontend)
expect_equal(as.integer(lat2$shape)[2], 25L)
