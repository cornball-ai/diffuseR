#' Audio Conditioning Frontend for LTX-2.3
#'
#' Turns user audio into the normalized, packed audio latents the joint
#' denoiser conditions on (lip sync): decode to 16 kHz stereo PCM,
#' log-mel via a causal STFT (filter 1024, hop 160, 64 slaney-normed
#' mel bins to 8 kHz — the checkpoint's preprocessing spec), then the
#' audio VAE encoder in argmax mode. The STFT and mel-filterbank
#' constructors were verified against the checkpoint's stored vocoder
#' bases (identical up to bf16 rounding), so the convention matches
#' training.
#'
#' @name audio_encode_ltx23
NULL

# Periodic Hann window
.ltx23_hann <- function(n) {
    0.5 - 0.5 * cos(2 * pi * (0:(n - 1L)) / n)
}

# Windowed Fourier basis in the layout ltx23_causal_stft expects:
# [n_freqs * 2, 1, filter_length], cos rows then -sin rows, each times
# the periodic Hann window (matches vocoder.mel_stft.stft_fn.forward_basis)
.ltx23_stft_basis <- function(filter_length) {
    nf <- filter_length %/% 2L + 1L
    n <- 0:(filter_length - 1L)
    win <- .ltx23_hann(filter_length)
    basis <- matrix(0, nf * 2L, filter_length)
    for (k in 0:(nf - 1L)) {
        basis[k + 1L, ] <- cos(2 * pi * k * n / filter_length) * win
        basis[nf + k + 1L, ] <- -sin(2 * pi * k * n / filter_length) * win
    }
    torch::torch_tensor(basis, dtype = torch::torch_float32())$unsqueeze(2L)
}

# Slaney-scale, slaney-normed mel filterbank (matches
# vocoder.mel_stft.mel_basis up to bf16 rounding)
.ltx23_mel_filterbank <- function(sample_rate, n_fft, n_mels, fmin, fmax) {
    hz2mel <- function(f) {
        ifelse(f < 1000, f * 3 / 200, 15 + log(f / 1000) / (log(6.4) / 27))
    }
    mel2hz <- function(m) {
        ifelse(m < 15, m * 200 / 3, 1000 * exp((m - 15) * log(6.4) / 27))
    }
    nf <- n_fft %/% 2L + 1L
    fft_freqs <- (0:(nf - 1L)) * sample_rate / n_fft
    hz <- mel2hz(seq(hz2mel(fmin), hz2mel(fmax), length.out = n_mels + 2L))
    W <- matrix(0, n_mels, nf)
    for (i in seq_len(n_mels)) {
        lower <- (fft_freqs - hz[i]) / (hz[i + 1] - hz[i])
        upper <- (hz[i + 2] - fft_freqs) / (hz[i + 2] - hz[i + 1])
        W[i, ] <- pmax(0, pmin(lower, upper)) * 2 / (hz[i + 2] - hz[i])
    }
    torch::torch_tensor(W, dtype = torch::torch_float32())
}

#' Build the 16 kHz log-mel frontend for audio conditioning
#'
#' An \code{\link{ltx23_mel_stft}} whose STFT and mel bases are
#' constructed (not checkpoint-loaded) with the audio VAE's
#' preprocessing spec.
#'
#' @param filter_length,hop_length,n_mels,sample_rate,fmin,fmax The
#'   checkpoint preprocessing parameters (defaults are LTX-2.3's).
#'
#' @return An \code{ltx23_mel_stft} module.
#'
#' @export
ltx23_audio_mel_frontend <- function(filter_length = 1024L,
                                     hop_length = 160L, n_mels = 64L,
                                     sample_rate = 16000L, fmin = 0,
                                     fmax = 8000) {
    frontend <- ltx23_mel_stft(
                               filter_length = filter_length, hop_length = hop_length,
                               window_length = filter_length, num_mel_channels = n_mels
    )
    torch::with_no_grad({
        frontend$stft_fn$forward_basis$copy_(.ltx23_stft_basis(filter_length))
        frontend$mel_basis$copy_(.ltx23_mel_filterbank(sample_rate,
                filter_length, n_mels, fmin, fmax))
    })
    frontend$eval()
    frontend
}

#' Read an audio file as 16 kHz stereo PCM
#'
#' Decodes MP3/WAV/etc. via \code{av} to 16-bit PCM at the target rate
#' and parses the RIFF container in base R.
#'
#' @param path Audio file.
#' @param sample_rate Integer.
#'
#' @return Matrix [2, samples] in [-1, 1].
#'
#' @export
ltx23_read_audio <- function(path, sample_rate = 16000L) {
    if (!requireNamespace("av", quietly = TRUE)) {
        stop("Reading audio requires the 'av' package.")
    }
    wav <- tempfile(fileext = ".wav")
    on.exit(unlink(wav), add = TRUE)
    av::av_audio_convert(path, wav, format = NULL, channels = 2L,
                         sample_rate = sample_rate, verbose = FALSE)

    con <- file(wav, "rb")
    on.exit(close(con), add = TRUE)
    riff <- readBin(con, "raw", 12L)
    stopifnot(rawToChar(riff[1:4]) == "RIFF")
    n_channels <- 2L
    repeat {
        hdr <- readBin(con, "raw", 8L)
        if (length(hdr) < 8L) {
            stop("No data chunk found in ", wav)
        }
        id <- rawToChar(hdr[1:4])
        size <- readBin(hdr[5:8], "integer", 1L, size = 4L, endian = "little")
        if (id == "fmt ") {
            fmt <- readBin(con, "raw", size)
            n_channels <- readBin(fmt[3:4], "integer", 1L, size = 2L,
                                  endian = "little")
        } else if (id == "data") {
            pcm <- readBin(con, "integer", size %/% 2L, size = 2L,
                           signed = TRUE, endian = "little")
            break
        } else {
            invisible(readBin(con, "raw", size + size %% 2L))
        }
    }
    m <- matrix(pcm / 32768, nrow = n_channels)
    if (n_channels == 1L) {
        m <- rbind(m, m)
    }
    m
}

#' Encode audio into normalized, packed conditioning latents
#'
#' Pads or trims the waveform so the latent length equals
#' \code{audio_num_frames} (mel frames \code{4L - 3}, mirroring the
#' decoder's \code{target_frames}), computes the log-mel, encodes in
#' argmax mode, packs, and normalizes with the checkpoint statistics.
#'
#' @param audio_vae An \code{ltx23_audio_vae} (with encoder weights).
#' @param wav Matrix [2, samples] in [-1, 1] at 16 kHz (see
#'   \code{\link{ltx23_read_audio}}).
#' @param audio_num_frames Integer. Target latent length.
#' @param frontend Optional prebuilt \code{\link{ltx23_audio_mel_frontend}}.
#'
#' @return Packed normalized latents [1, audio_num_frames, 128] (float32).
#'
#' @export
ltx23_encode_audio <- function(audio_vae, wav, audio_num_frames,
                               frontend = NULL) {
    frontend <- frontend %||% ltx23_audio_mel_frontend()
    target_mel_frames <- 4L * audio_num_frames - 3L

    dev <- audio_vae$latents_mean$device
    enc_dtype <- audio_vae$encoder$conv_in$conv$weight$dtype
    frontend$to(device = dev)

    x <- torch::torch_tensor(wav, dtype = torch::torch_float32(),
                             device = dev)
    torch::with_no_grad({
        # Stereo channels through the STFT as a batch of mono signals
        mel <- frontend(x$unsqueeze(2L)) # [2, 64, T]
        # -> [1, 2, T, 64]
        mel <- mel$permute(c(1L, 3L, 2L))$unsqueeze(1L)
        t_cur <- mel$shape[3]
        if (t_cur < target_mel_frames) {
            mel <- torch::nnf_pad(mel, c(0L, 0L, 0L, target_mel_frames - t_cur))
        } else if (t_cur > target_mel_frames) {
            mel <- mel$narrow(3L, 1L, target_mel_frames)
        }
        moments <- audio_vae$encode(mel$to(dtype = enc_dtype))
        latents <- moments$mean$to(dtype = torch::torch_float32())
    })
    packed <- ltx23_pack_audio_latents(latents)
    mean <- audio_vae$latents_mean$to(dtype = torch::torch_float32())
    std <- audio_vae$latents_std$to(dtype = torch::torch_float32())
    (packed - mean) / std
}
