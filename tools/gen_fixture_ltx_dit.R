# LTX-2.3 audio-video DiT parity fixture (torch reference, RANDOM-INIT
# weights, SMALL config). Instantiate diffuseR::ltx23_transformer at a
# memory-light config with torch_manual_seed random weights, run the real
# forward under no_grad on random inputs (eager block path, jit disabled),
# and save the module state_dict + the inputs the anvl closure needs
# (hidden states, per-token timesteps, sigmas, precomputed RoPE cos/sin)
# + the reference video/audio outputs to one f32 safetensors. The anvl
# loader reads the SAME state_dict.
#
# Config invariants that make the reference run and stay parity-safe:
#   cross_attention_dim == inner_dim, audio_cross_attention_dim ==
#   audio_inner_dim (prompt-KV modulation + cross-rope shape match); every
#   attention has n_k != d_v so the .ltx23_sdpa scratch aliasing bug
#   (attn_buf shape == out_buf shape) never fires; head dims (8 video,
#   6 audio) differ from every sequence length (12,5,7,4).
#
# Everything is $contiguous() before saving (view-save trap).
#
# Usage: r tools/gen_fixture_ltx_dit.R  [isolate]
#   isolate (default 1): video/audio streams independent (no a2v/v2a).
#   0: full dual-stream with audio<->video cross-attention.

suppressMessages(library(torch))
suppressMessages(library(diffuseR))
options(diffuseR.jit_blocks = FALSE)   # force the eager per-block path

ISOLATE <- TRUE
if (length(argv) >= 1L) ISOLATE <- as.logical(as.integer(argv[[1]]))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir,
                     if (ISOLATE) "ltx_dit_iso.safetensors" else "ltx_dit.safetensors")

# ---- small but architecturally faithful config ----
na <- 3L;  ah <- 8L                 # video: 3 heads x 8 = inner 24
aa <- 2L;  ad <- 6L                 # audio: 2 heads x 6 = inner 12
inner  <- na * ah                   # 24
ainner <- aa * ad                   # 12
in_ch   <- 4L; out_ch   <- 4L
a_in_ch <- 3L; a_out_ch <- 3L
n_layers <- 2L
fps <- 24.0

B <- 1L
nf <- 2L; H <- 3L; W <- 2L          # video grid -> 12 tokens
audio_nf <- 5L                      # audio -> 5 tokens
s_video <- nf * H * W               # 12
s_audio <- audio_nf                 # 5
s_text  <- 7L                       # video text tokens (already at inner)
s_atext <- 4L                       # audio text tokens (already at ainner)

torch_manual_seed(11)
m <- ltx23_transformer(
  in_channels = in_ch, out_channels = out_ch,
  patch_size = 1L, patch_size_t = 1L,
  num_attention_heads = na, attention_head_dim = ah,
  cross_attention_dim = inner,
  audio_in_channels = a_in_ch, audio_out_channels = a_out_ch,
  audio_patch_size = 1L, audio_patch_size_t = 1L,
  audio_num_attention_heads = aa, audio_attention_head_dim = ad,
  audio_cross_attention_dim = ainner,
  num_layers = n_layers
)
m$eval()

# ---- random inputs ----
torch_manual_seed(101)
hidden        <- torch_randn(B, s_video, in_ch)
audio_hidden  <- torch_randn(B, s_audio, a_in_ch)
enc           <- torch_randn(B, s_text, inner)
audio_enc     <- torch_randn(B, s_atext, ainner)
# per-token timesteps (video: [B, s_video], audio: [B, s_audio]); sigma is
# global per batch ([B, 1]). Values in a plausible scaled range.
timestep       <- torch_rand(B, s_video) * 900 + 50
audio_timestep <- torch_rand(B, s_audio) * 900 + 50
sigma          <- torch_rand(B, 1L)
audio_sigma    <- torch_rand(B, 1L)
# text padding masks ([B, S], 1 real / 0 pad) exercise the [B,S] ->
# [B,1,1,S] additive-bias broadcast in the text cross-attention.
enc_mask   <- torch_ones(B, s_text);  enc_mask[, (s_text - 1):s_text] <- 0
aenc_mask  <- torch_ones(B, s_atext); aenc_mask[, s_atext] <- 0

# ---- precompute the RoPE coords + tables exactly as the forward would ----
dev <- torch_device("cpu")
video_coords <- m$rope$prepare_video_coords(B, nf, H, W, dev, fps = fps)
audio_coords <- m$audio_rope$prepare_audio_coords(B, audio_nf, dev)
video_rope <- m$rope(video_coords, device = dev)          # list(cos,sin) [B,na,s_video,ah/2]
audio_rope <- m$audio_rope(audio_coords, device = dev)     # [B,aa,s_audio,ad/2]
vca_rope <- m$cross_attn_rope(video_coords$narrow(2L, 1L, 1L), device = dev)
aca_rope <- m$cross_attn_audio_rope(audio_coords$narrow(2L, 1L, 1L), device = dev)

out <- with_no_grad(m(
  hidden_states = hidden,
  audio_hidden_states = audio_hidden,
  encoder_hidden_states = enc,
  audio_encoder_hidden_states = audio_enc,
  timestep = timestep,
  audio_timestep = audio_timestep,
  sigma = sigma,
  audio_sigma = audio_sigma,
  encoder_attention_mask = enc_mask,
  audio_encoder_attention_mask = aenc_mask,
  video_coords = video_coords,
  audio_coords = audio_coords,
  fps = fps,
  isolate_modalities = ISOLATE,
  use_cross_timestep = FALSE
))

ct <- function(x) x$contiguous()
sd <- lapply(m$state_dict(), ct)
inputs <- list(
  hidden = ct(hidden),
  audio_hidden = ct(audio_hidden),
  enc = ct(enc),
  audio_enc = ct(audio_enc),
  timestep = ct(timestep),
  audio_timestep = ct(audio_timestep),
  sigma = ct(sigma),
  audio_sigma = ct(audio_sigma),
  enc_mask = ct(enc_mask),
  aenc_mask = ct(aenc_mask),
  v_cos = ct(video_rope[[1]]), v_sin = ct(video_rope[[2]]),
  a_cos = ct(audio_rope[[1]]), a_sin = ct(audio_rope[[2]]),
  vca_cos = ct(vca_rope[[1]]), vca_sin = ct(vca_rope[[2]]),
  aca_cos = ct(aca_rope[[1]]), aca_sin = ct(aca_rope[[2]]),
  video_out = ct(out$sample),
  audio_out = ct(out$audio_sample)
)
stopifnot(length(intersect(names(sd), names(inputs))) == 0L)
safetensors::safe_save_file(c(sd, inputs), fixture)

cat(sprintf("fixture: %s (%.2f MB)  isolate=%s\n",
            fixture, file.size(fixture) / 1e6, ISOLATE))
cat(sprintf("config: inner=%d(%dx%d) ainner=%d(%dx%d) layers=%d\n",
            inner, na, ah, ainner, aa, ad, n_layers))
cat(sprintf("seqs: video=%d audio=%d text=%d atext=%d\n",
            s_video, s_audio, s_text, s_atext))
cat(sprintf("video_out %s sd %.4f   audio_out %s sd %.4f\n",
            paste(dim(out$sample), collapse = "x"), out$sample$std()$item(),
            paste(dim(out$audio_sample), collapse = "x"), out$audio_sample$std()$item()))
cat(sprintf("v_cos %s  a_cos %s  vca_cos %s  aca_cos %s\n",
            paste(dim(video_rope[[1]]), collapse = "x"),
            paste(dim(audio_rope[[1]]), collapse = "x"),
            paste(dim(vca_rope[[1]]), collapse = "x"),
            paste(dim(aca_rope[[1]]), collapse = "x")))
