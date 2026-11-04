# General LTX keyframes: frame-0 replacement, appended keyframes at any
# other frame, strengths, and the older single-source arguments as a
# frame-0 keyframe. Tiny random-weight pipeline on the CPU, as in
# test_txt2vid_ltx23.R.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)

# ---- constructor -----------------------------------------------------------
lat <- torch::torch_randn(1L, 128L, 1L, 2L, 2L)
kf <- ltx23_keyframe(lat, frame = 16L, strength = 0.5)
expect_inherits(kf, "ltx23_keyframe")
expect_equal(kf$frame, 16L)
expect_equal(kf$strength, 0.5)
expect_error(ltx23_keyframe(lat, frame = -1L), "non-negative")
expect_error(ltx23_keyframe(lat, strength = 0), "strength")
expect_error(ltx23_keyframe(lat, strength = 1.5), "strength")
expect_error(ltx23_keyframe(list()), "latents, an image path")

# ---- tiny pipeline -------------------------------------------------------
torch::torch_manual_seed(99)
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
src <- readLines(system.file("tinytest", "test_txt2vid_ltx23.R",
                             package = "diffuseR"))
if (!length(src)) {
  src <- readLines("test_txt2vid_ltx23.R")
}
# The VAE, vocoder, pipeline and stub embeds exactly as the pipeline
# test builds them
setup <- src[grep("^vae <- ltx23_video_vae", src):(grep("^res <- txt2vid_ltx2", src) - 1L)]
eval(parse(text = setup))

run <- function(...) {
  txt2vid_ltx2("x", pipe, prompt_embeds = stub_embeds, width = 64L,
               height = 64L, num_frames = 25L, frame_rate = 24, seed = 7L,
               device = "cpu", dtype = "float32", decode_audio = FALSE,
               verbose = FALSE, ...)
}
lat_of <- function(res) {
  diffuseR:::ltx23_unpack_video_latents(res$latents, res$latent_shape[1],
                                        res$latent_shape[2],
                                        res$latent_shape[3])
}
torch::torch_manual_seed(5)
k2 <- torch::torch_randn(1L, 128L, 2L, 2L, 2L)
k1 <- torch::torch_randn(1L, 128L, 1L, 2L, 2L)

# ---- a frame-0 replace keyframe is condition_latents ---------------------
a <- run(condition_latents = k2)
b <- run(keyframes = list(ltx23_keyframe(k2, frame = 0L)))
expect_equal(max(abs(a$video - b$video)), 0)
# a single keyframe need not be wrapped in a list
c1 <- run(keyframes = ltx23_keyframe(k2, frame = 0L))
expect_equal(max(abs(a$video - c1$video)), 0)

# ---- the frame-0 keyframe stays exactly as given ---------------------------
# 25 frames -> 4 latent frames; the last starts at pixel frame 17
both <- run(keyframes = list(ltx23_keyframe(k1, frame = 0L),
                             ltx23_keyframe(k1 * 2, frame = 17L)))
lb <- lat_of(both)
expect_equal(as.numeric((lb$narrow(3L, 1L, 1L) - k1)$abs()$max()), 0)
expect_equal(both$latent_shape, c(4L, 2L, 2L))
expect_equal(dim(both$video), c(25L, 64L, 64L, 3L))
expect_true(max(abs(both$video - run(keyframes = ltx23_keyframe(k1))$video)) > 1e-6)
# a frame-0 keyframe longer than the generation is refused
expect_error(run(keyframes = ltx23_keyframe(torch::torch_randn(1L, 128L, 5L, 2L, 2L))),
             "has 5 latent frames")

# ---- the starting state: replaced tokens, appended tokens, masks -----------
noise <- torch::torch_randn(1L, 128L, 4L, 2L, 2L)
enc_fail <- function(frames) stop("no pixel keyframes here")
st <- diffuseR:::.ltx23_keyframe_state(
  list(ltx23_keyframe(k1, 0L), ltx23_keyframe(k2, 9L, strength = 0.5)),
  4L, 2L, 2L, noise, enc_fail, 64L, 64L, transformer$rope, 24)
# 16 main tokens (4 x 2 x 2), then 8 appended (2 latent frames x 2 x 2)
expect_equal(st$n_main, 16L)
expect_equal(as.integer(st$latents$shape), c(1L, 24L, 128L))
expect_equal(as.numeric(st$conditioning_mask), c(rep(1, 4), rep(0, 12), rep(0.5, 8)))
expect_equal(as.integer(st$extra_coords$shape), c(1L, 3L, 8L, 2L))
# frame-0 tokens start clean; free tokens are pure noise; appended tokens
# start halfway between their keyframe and noise
packed <- diffuseR:::ltx23_pack_video_latents
expect_equal(as.numeric((st$latents[, 1:4, ] - packed(k1))$abs()$max()), 0)
expect_equal(as.numeric((st$latents[, 5:16, ] - packed(noise)[, 5:16, ])$abs()$max()), 0)
expect_equal(as.numeric((st$clean[, 17:24, ] - packed(k2))$abs()$max()), 0)

# ---- appended keyframes: same output geometry, effect on the video ---------
g <- run(keyframes = list(ltx23_keyframe(k1, frame = 0L),
                          ltx23_keyframe(k1 * 2, frame = 12L)))
expect_equal(dim(g$video), c(25L, 64L, 64L, 3L))
expect_equal(g$latent_shape, c(4L, 2L, 2L))
expect_true(max(abs(g$video - run(keyframes = ltx23_keyframe(k1))$video)) > 1e-6)
gp <- run(keyframes = ltx23_keyframe(k1, frame = 12L, strength = 0.6))
expect_true(all(is.finite(gp$video)))
# partial strength at frame 0 (per-token timesteps and the x0 blend)
h0 <- run(keyframes = ltx23_keyframe(k1, frame = 0L, strength = 0.5))
expect_true(all(is.finite(h0$video)))
d_h0 <- as.numeric((lat_of(h0)$narrow(3L, 1L, 1L) - k1)$abs()$mean())
d_free <- as.numeric((lat_of(run())$narrow(3L, 1L, 1L) - k1)$abs()$mean())
expect_true(d_h0 > 0)
expect_true(d_h0 < d_free)

# appended coordinates: [start, end) in seconds offset by the frame, no
# causal first-frame fix, one pixel frame for an image (diffusers
# _prepare_keyframe_coords)
rope <- transformer$rope
co <- diffuseR:::.ltx23_keyframe_coords(rope, 2L, 2L, 2L, 16L, FALSE, 24,
                                        "cpu")
expect_equal(dim(co), c(1L, 3L, 8L, 2L))
expect_equal(as.numeric(co[1, 1, 1, ]) * 24, c(16, 24), tolerance = 1e-6)
expect_equal(as.numeric(co[1, 1, 5, ]) * 24, c(24, 32), tolerance = 1e-6)
co1 <- diffuseR:::.ltx23_keyframe_coords(rope, 1L, 2L, 2L, 12L, TRUE, 24,
                                         "cpu")
expect_equal(as.numeric(co1[1, 1, 1, ]) * 24, c(12, 13), tolerance = 1e-6)
# spatial axes match the generation's own grid
full <- rope$prepare_video_coords(1L, 2L, 2L, 2L, "cpu", fps = 24)
expect_equal(as.numeric((co[, 2:3, , ] - full[, 2:3, , ])$abs()$max()), 0)

# ---- pixel keyframes are encoded at generation time ----------------------
img <- array(runif(64 * 64 * 3), dim = c(64L, 64L, 3L))
pi <- run(keyframes = list(ltx23_keyframe(img, frame = 0L),
                           ltx23_keyframe(img, frame = 17L)))
expect_equal(dim(pi$video), c(25L, 64L, 64L, 3L))
expect_equal(max(abs(run(image = img)$video - run(keyframes = ltx23_keyframe(img))$video)), 0)

# ---- re-entry state (stage 2): base latents, noise_scale ------------------
base <- torch::torch_randn(1L, 128L, 4L, 2L, 2L)
ns <- 0.9
st2 <- diffuseR:::.ltx23_keyframe_state(
  list(ltx23_keyframe(k1, 0L), ltx23_keyframe(k2, 9L, strength = 0.5)),
  4L, 2L, 2L, noise, enc_fail, 64L, 64L, transformer$rope, 24,
  base = base, noise_scale = ns)
expect_equal(as.numeric(st2$conditioning_mask), as.numeric(st$conditioning_mask))
# pinned frame-0 tokens are the keyframe; free tokens re-enter from base
# at noise_scale; appended tokens at (1 - strength) * noise_scale
expect_equal(as.numeric((st2$latents[, 1:4, ] - packed(k1))$abs()$max()), 0)
free <- packed(noise)[, 5:16, ] * ns + packed(base)[, 5:16, ] * (1 - ns)
expect_true(as.numeric((st2$latents[, 5:16, ] - free)$abs()$max()) < 1e-6)
e_mix <- as.numeric(((st2$latents[, 17:24, ] - packed(k2)) /
                       (1 - 0.5) / ns)$abs()$mean())
expect_true(e_mix > 0.5 && e_mix < 1.5)

# ---- two-stage with keyframes --------------------------------------------
torch::torch_manual_seed(3)
up <- ltx23_latent_upsampler(in_channels = 128L, mid_channels = 32L,
                             num_blocks_per_stage = 1L)
up$eval()
run2 <- function(...) {
  txt2vid_ltx2("x", pipe, prompt_embeds = stub_embeds, width = 128L,
               height = 128L, num_frames = 25L, frame_rate = 24, seed = 7L,
               device = "cpu", dtype = "float32", decode_audio = FALSE,
               verbose = FALSE, two_stage = TRUE, upsampler = up, ...)
}
plain2 <- run2()
expect_equal(dim(plain2$video), c(25L, 128L, 128L, 3L))
expect_equal(plain2$latent_shape, c(4L, 4L, 4L))
# latent keyframes are given at stage-1 size and upsampled for stage 2,
# where a pinned frame-0 keyframe stays exactly the upsampled content
l2 <- run2(keyframes = list(ltx23_keyframe(k1, frame = 0L),
                            ltx23_keyframe(k1 * 2, frame = 17L)))
expect_equal(dim(l2$video), c(25L, 128L, 128L, 3L))
expect_equal(l2$latent_shape, c(4L, 4L, 4L))
k1_up <- torch::with_no_grad(
  diffuseR:::.ltx23_upsample_latents(up, k1, pipe$vae, adain_factor = 1))
expect_true(as.numeric((lat_of(l2)$narrow(3L, 1L, 1L) - k1_up)$abs()$max()) < 1e-5)
expect_true(max(abs(l2$video - plain2$video)) > 1e-6)
# pixel keyframes are encoded at each stage's size
img2 <- array(runif(128 * 128 * 3), dim = c(128L, 128L, 3L))
p2 <- run2(keyframes = list(ltx23_keyframe(img2, frame = 0L),
                            ltx23_keyframe(img2, frame = 12L, strength = 0.7)))
expect_equal(dim(p2$video), c(25L, 128L, 128L, 3L))
expect_true(all(is.finite(p2$video)))
expect_equal(max(abs(run2(image = img2)$video -
                     run2(keyframes = ltx23_keyframe(img2))$video)), 0)
# full-size latents are refused (stage 1 needs them at half size)
expect_error(run2(condition_latents = torch::torch_randn(1L, 128L, 1L, 4L, 4L)),
             "condition_latents must be")
expect_error(run2(keyframes = ltx23_keyframe(torch::torch_randn(1L, 128L, 1L, 4L, 4L),
                                             frame = 9L)),
             "keyframe latents must be")
# the temporal upscaler cannot stand in for the spatial one
up_t <- ltx23_latent_upsampler(in_channels = 128L, mid_channels = 32L,
                               num_blocks_per_stage = 1L, temporal = TRUE)
expect_error(txt2vid_ltx2("x", pipe, prompt_embeds = stub_embeds,
                          width = 128L, height = 128L, num_frames = 25L,
                          device = "cpu", dtype = "float32", verbose = FALSE,
                          two_stage = TRUE, upsampler = up_t),
             "spatial upsampler")
# conditioning audio stays frozen through stage 2
wav <- matrix(runif(2L * 16000L, -0.5, 0.5), nrow = 2L)
a1 <- run(audio = wav)
a2 <- run2(audio = wav)
expect_equal(as.numeric((a1$audio_latents - a2$audio_latents)$abs()$max()), 0)

# ---- video-to-video ----------------------------------------------------------
torch::torch_manual_seed(11)
src <- array(runif(25 * 64 * 64 * 3), dim = c(25L, 64L, 64L, 3L))
src_lat <- torch::with_no_grad(diffuseR::ltx23_encode_video_frames(
  pipe$vae, diffuseR::ltx23_preprocess_frames(src, 64L, 64L)))
v5 <- run(video = src, strength = 0.5)
expect_equal(dim(v5$video), c(25L, 64L, 64L, 3L))
expect_true(all(is.finite(v5$video)))
# no audio given and no soundtrack: silence, held fixed and muxed
expect_equal(v5$sample_rate, 16000L)
expect_true(all(v5$audio == 0))
# latents and pixels of the same source give the same result
expect_true(max(abs(run(video = src_lat, strength = 0.5)$video - v5$video)) < 1e-6)
# lower strength stays closer to the source
d_lo <- as.numeric((lat_of(run(video = src_lat, strength = 0.3)) - src_lat)$abs()$mean())
d_hi <- as.numeric((lat_of(run(video = src_lat, strength = 0.9)) - src_lat)$abs()$mean())
expect_true(d_lo < d_hi)
expect_error(run(video = src, strength = 0), "strength")
expect_error(run(video = src, strength = 1.2), "strength")
expect_error(run(video = src[1:9, , , ]), "at least num_frames")
expect_error(run(video = torch::torch_randn(1L, 128L, 2L, 2L, 2L)), "video latents must be")
# with keyframes, and with two-stage (video at full size, encoded at half)
vk <- run(video = src, strength = 0.6,
          keyframes = ltx23_keyframe(src[1, , , ], frame = 0L))
expect_equal(dim(vk$video), c(25L, 64L, 64L, 3L))
src2 <- array(runif(25 * 128 * 128 * 3), dim = c(25L, 128L, 128L, 3L))
v2 <- run2(video = src2, strength = 0.6)
expect_equal(dim(v2$video), c(25L, 128L, 128L, 3L))
expect_true(all(is.finite(v2$video)))
# stage 2 enters at the v2v strength when it is below 0.909375: one
# step from 0.3, the full three from 0.95
s2_steps <- function(s) {
  msgs <- character()
  withCallingHandlers(
    txt2vid_ltx2("x", pipe, prompt_embeds = stub_embeds, width = 128L,
                 height = 128L, num_frames = 25L, seed = 7L, device = "cpu",
                 dtype = "float32", decode_video = FALSE, decode_audio = FALSE,
                 verbose = TRUE, two_stage = TRUE, upsampler = up,
                 video = src2, strength = s),
    message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      invokeRestart("muffleMessage")
    })
  sub("Refining: (\\d+) steps.*", "\\1", grep("^Refining", msgs, value = TRUE))
}
expect_equal(s2_steps(0.3), "1")
expect_equal(s2_steps(0.95), "3")
# a file source: frames read from the start, silent file -> silence
if (requireNamespace("av", quietly = TRUE)) {
  dir <- tempfile("v2v_")
  dir.create(dir)
  pngs <- file.path(dir, sprintf("f%03d.png", 1:30))
  for (i in 1:30) {
    png::writePNG(src[min(i, 25L), , , ], pngs[i])
  }
  mp4 <- file.path(dir, "src.mp4")
  av::av_encode_video(pngs, mp4, framerate = 24, verbose = FALSE)
  vf <- run(video = mp4, strength = 0.5)
  expect_equal(dim(vf$video), c(25L, 64L, 64L, 3L))
  expect_true(all(vf$audio == 0))
  unlink(dir, recursive = TRUE)
}
