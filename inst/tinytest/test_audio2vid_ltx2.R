# audio2vid_ltx2(): the window plan, the latent chain, the audio under
# each window, and resume. A fake generator stands in for LTX: it writes
# a placeholder file and returns packed latents whose values name the
# window and latent frame, so the carried tail can be read back. CPU only.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
    exit_file("torch not fully installed")
}

library(diffuseR)

# --- the plan ---------------------------------------------------------------

w <- ltx2_windows(300)
expect_equal(w$start, c(0L, 72L, 144L, 216L))
expect_equal(w$frames, c(97L, 97L, 97L, 89L))   # 84 left, up to 89
expect_true(all((w$frames - 1L) %% 8L == 0L))
expect_true(w$start[4] + w$frames[4] >= 300)

w <- ltx2_windows(150)
expect_equal(w$start, c(0L, 72L))
expect_equal(w$frames, c(97L, 81L))   # 78 left, up to 81 on the grid

expect_equal(nrow(ltx2_windows(50)), 1L)
expect_equal(ltx2_windows(50)$frames, 57L)
expect_equal(ltx2_windows(98)$frames, c(97L, 33L))   # never shorter than carry + 8
expect_error(ltx2_windows(100, window = 96L), "8k\\+1")
expect_error(ltx2_windows(100, carry = 97L), "carry < window")
expect_error(ltx2_windows(0), "no frames")

# --- a fake generator -------------------------------------------------------

calls <- list()
fake <- function(prompt, width, height, num_frames, frame_rate, audio,
                 filename, verbose, image = NULL, condition_latents = NULL,
                 seed = NULL, ...) {
    j <- length(calls) + 1L
    calls[[j]] <<- list(prompt = prompt, num_frames = num_frames,
                        samples = ncol(audio), image = image,
                        cond = if (is.null(condition_latents)) NULL else
                            as.numeric(condition_latents$flatten()),
                        cond_shape = if (is.null(condition_latents)) NULL else
                            as.integer(condition_latents$shape),
                        seed = seed, extra = list(...))
    writeLines("mp4", filename)
    lf <- (num_frames - 1L) %/% 8L + 1L
    # packed [1, lf * 1 * 1, 128]: every channel of latent frame f holds
    # 1000 * call + f
    v <- rep(1000 * j + seq_len(lf), times = 128L)   # [1, f, c], column-major
    list(latents = torch::torch_tensor(array(v, c(1L, lf, 128L)),
                                       dtype = torch::torch_float32()),
         latent_shape = c(lf, 1L, 1L))
}

rate <- 16000
secs <- 300 / 24                       # 300 frames of narration
audio <- matrix(0, 2L, as.integer(secs * rate))
dir <- tempfile("a2v-")

out <- audio2vid_ltx2(audio, image = "start.png", prompt = "a corn cob talks",
                      output_dir = dir, width = 64L, height = 64L,
                      generate = fake, text_encoder = "te")
expect_equal(nrow(out), 4L)
expect_true(all(out$rendered))
expect_equal(length(calls), 4L)
expect_true(all(file.exists(out$file)))
expect_true(all(file.exists(out$latents)))
expect_true(all(file.exists(sub("\\.mp4$", ".json", out$file))))

# the first window starts from the image, the rest from latents
expect_equal(calls[[1]]$image, "start.png")
expect_null(calls[[1]]$cond)
expect_null(calls[[2]]$image)
# carry 25 -> 4 latent frames: the last four of window 1 (13 latent frames)
expect_equal(calls[[2]]$cond_shape, c(1L, 128L, 4L, 1L, 1L))
expect_equal(unique(calls[[2]]$cond), 1000 + 10:13)
expect_equal(unique(calls[[4]]$cond), 3000 + 10:13)

# every window hears its own span, overlap included
expect_equal(calls[[1]]$samples, round(97 / 24 * rate))
expect_equal(calls[[2]]$samples,
             round((72 + 97) / 24 * rate) - round(72 / 24 * rate))
expect_equal(calls[[4]]$samples, ncol(audio) - round(216 / 24 * rate))
expect_equal(vapply(calls, `[[`, 1L, "num_frames"), c(97L, 97L, 97L, 89L))
expect_equal(calls[[3]]$extra$text_encoder, "te")
expect_null(calls[[1]]$seed)

# --- resume -----------------------------------------------------------------

calls <- list()
out2 <- audio2vid_ltx2(audio, image = "start.png", prompt = "a corn cob talks",
                       output_dir = dir, width = 64L, height = 64L,
                       generate = fake)
expect_equal(length(calls), 0L)
expect_false(any(out2$rendered))

# window 3 lost: windows 1-2 kept, 3 renders from window 2's saved
# latents, and 4 renders again because its conditioning changed
unlink(out$latents[3])
calls <- list()
out3 <- audio2vid_ltx2(audio, image = "start.png", prompt = "a corn cob talks",
                       output_dir = dir, width = 64L, height = 64L,
                       generate = fake)
expect_equal(out3$rendered, c(FALSE, FALSE, TRUE, TRUE))
expect_equal(length(calls), 2L)
expect_equal(unique(calls[[1]]$cond), 2000 + 10:13)   # window 2 of the first run
expect_equal(unique(calls[[2]]$cond), 1000 + 10:13)   # this run's window 3

# a changed prompt for one window re-renders from there on
calls <- list()
out4 <- audio2vid_ltx2(audio, image = "start.png",
                       prompt = c("a corn cob talks", "a corn cob talks",
                                  "a corn cob sings", "a corn cob talks"),
                       output_dir = dir, width = 64L, height = 64L,
                       generate = fake)
expect_equal(out4$rendered, c(FALSE, FALSE, TRUE, TRUE))
expect_equal(calls[[1]]$prompt, "a corn cob sings")

# resume = FALSE renders everything
calls <- list()
out5 <- audio2vid_ltx2(audio, image = "start.png", prompt = "x",
                       output_dir = dir, width = 64L, height = 64L,
                       resume = FALSE, generate = fake, seed = 7L)
expect_true(all(out5$rendered))
expect_equal(vapply(calls, `[[`, 1L, "seed"), 7:10)

# --- on_window: progress, and a stop that leaves resumable windows ---------

seen <- integer(0)
dir2 <- tempfile("a2v-")
calls <- list()
res <- tryCatch(
    audio2vid_ltx2(audio, image = "start.png", prompt = "x", output_dir = dir2,
                   width = 64L, height = 64L, generate = fake,
                   on_window = function(r) {
                       seen <<- c(seen, r$index)
                       if (r$index == 2L) stop("preempted")
                   }),
    error = function(e) conditionMessage(e))
expect_equal(res, "preempted")
expect_equal(seen, 1:2)
calls <- list()
out6 <- audio2vid_ltx2(audio, image = "start.png", prompt = "x",
                       output_dir = dir2, width = 64L, height = 64L,
                       generate = fake)
expect_equal(out6$rendered, c(FALSE, FALSE, TRUE, TRUE))

# --- a caller's plan --------------------------------------------------------

plan <- data.frame(start = c(0L, 120L), frames = c(145L, 113L))
calls <- list()
out7 <- audio2vid_ltx2(matrix(0, 2L, as.integer(10 * rate)), image = "s.png",
                       prompt = c("a", "b"), output_dir = tempfile("a2v-"),
                       width = 64L, height = 64L, windows = plan,
                       generate = fake)
expect_equal(out7$frames, c(145L, 113L))
expect_equal(vapply(calls, `[[`, "", "prompt"), c("a", "b"))

# --- refusals ---------------------------------------------------------------

d <- tempfile("a2v-")
expect_error(audio2vid_ltx2(audio, "s.png", "x", output_dir = d,
                            width = 60L, generate = fake), "multiples of 32")
expect_error(audio2vid_ltx2(audio, "s.png", c("a", "b"), output_dir = d,
                            generate = fake), "one per window")
expect_error(audio2vid_ltx2(matrix(0, 1L, 10L), "s.png", "x", output_dir = d,
                            generate = fake), "\\[2, samples\\]")
expect_error(audio2vid_ltx2(audio, "s.png", "x", output_dir = d,
                            windows = data.frame(start = c(0L, 80L),
                                                 frames = c(97L, 97L)),
                            generate = fake), "the chain needs 72")
expect_error(audio2vid_ltx2(audio, "s.png", "x", output_dir = d,
                            windows = data.frame(start = 0L, frames = 96L),
                            generate = fake), "8k\\+1")
expect_error(audio2vid_ltx2(audio, "s.png", "x", output_dir = d),
             "ltx23_pipeline or a resident")
