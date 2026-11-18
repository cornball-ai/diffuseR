# audio2vid_ltx2.R
# Long audio-driven video: one narration of any length rendered as a
# chain of fixed LTX windows, each continuing the last from its tail
# latents, the way Lightricks' ltx-pipelines lays out long video. Window
# j + 1 starts `carry` frames before window j ends; those frames are
# carried as window j's last denoised latent frames (no decode, no
# re-encode), and every window hears the narration over its whole span,
# overlap included. The windows are written to `output_dir` as they
# finish, each with its tail latents and a record, so a run that stops
# resumes at the first window it does not have.

#' Fixed windows over a narration
#'
#' Lays out windows of \code{window} frames that overlap by \code{carry}
#' frames and cover \code{frames}. Every window but the last is
#' \code{window} frames; the last is sized up to the end on the 8k+1
#' grid and is never shorter than \code{carry + 8}.
#'
#' @param frames Frames to cover.
#' @param window Frames per window, 8k+1 (default 97).
#' @param carry Overlap in frames, 8k+1, at least 9 and below
#'   \code{window} (default 25, four latent frames).
#'
#' @return Data frame with \code{index}, \code{start} (0-based frame)
#'   and \code{frames}.
#'
#' @examples
#' ltx2_windows(300)
#'
#' @export
ltx2_windows <- function(frames, window = 97L, carry = 25L) {
    window <- as.integer(window)
    carry <- as.integer(carry)
    if (!.on_8k1(window) || !.on_8k1(carry) || carry < 9L ||
        carry >= window) {
        stop("window and carry must be 8k+1, with 9 <= carry < window",
             call. = FALSE)
    }
    frames <- as.integer(ceiling(frames))
    if (length(frames) != 1L || is.na(frames) || frames < 1L) {
        stop("no frames to cover", call. = FALSE)
    }
    starts <- 0L
    while (starts[length(starts)] + window < frames) {
        starts <- c(starts, starts[length(starts)] + window - carry)
    }
    n <- length(starts)
    len <- rep(window, n)
    rest <- frames - starts[n]
    last <- 8L * as.integer(ceiling((rest - 1) / 8)) + 1L
    len[n] <- min(window, max(last, if (n > 1L) carry + 8L else 9L))
    data.frame(index = seq_len(n), start = starts, frames = len)
}

#' Long audio-driven video with LTX-2
#'
#' Renders a narration of any length as a chain of LTX windows. The first
#' window starts from \code{image}; each later window is conditioned on
#' the previous window's last \code{carry} frames as latents, and every
#' window is driven by the narration over its own span. Each window is
#' written to \code{output_dir} as it finishes: \code{windowNN.mp4} (the
#' window's frames, its carried head included, with its audio),
#' \code{windowNN.latents.pt} (its tail latents) and \code{windowNN.json}
#' (its plan and prompt). Joining the windows (crossfading each carried
#' head over the previous window's tail) is left to the caller.
#'
#' A second call with the same arguments resumes: a window whose three
#' files exist and whose record matches the plan is kept, and the chain
#' continues from its saved latents. Once one window is rendered again,
#' every later window is too, since its conditioning changed.
#'
#' @param audio The narration: a file path (read with
#'   \code{\link{ltx23_read_audio}}, which needs \code{av}) or a matrix
#'   [2, samples] in [-1, 1] at 16 kHz.
#' @param image Start frame of the first window: a PNG/JPEG path or an
#'   [H, W, 3] array in [0, 1].
#' @param prompt Character. One prompt for every window, or one per
#'   window.
#' @param pipeline An \code{ltx23_pipeline} (from
#'   \code{\link{ltx23_load_pipeline}}) or a resident LTX handle (from
#'   \code{\link{resident_load}}, active).
#' @param output_dir Directory for the windows; created if needed.
#' @param width,height Frame size, multiples of 32.
#' @param window,carry Window length and overlap in frames, 8k+1 (see
#'   \code{\link{ltx2_windows}}). Ignored when \code{windows} is given,
#'   except \code{carry}, which every plan shares.
#' @param windows Optional plan: a data frame with \code{start} and
#'   \code{frames} (8k+1), each window starting \code{carry} frames before
#'   the previous one ends, the first at 0. Default
#'   \code{ltx2_windows(<narration frames>, window, carry)}.
#' @param frame_rate Frames per second.
#' @param seed Optional integer; window j renders with \code{seed + j - 1}.
#' @param resume Logical. Keep windows already on disk that match the plan.
#' @param on_window Optional function called with each window's record (a
#'   list: \code{index}, \code{start}, \code{frames}, \code{file},
#'   \code{latents}, \code{rendered}, \code{seconds}) once it is on disk.
#'   An error raised there stops the run after that window; the windows
#'   written so far stay, so a later call resumes.
#' @param verbose Passed to \code{\link{txt2vid_ltx2}}.
#' @param generate Optional generator \code{function(prompt, ...)} in place
#'   of the one \code{pipeline} implies; for tests and hosts.
#' @param ... Further arguments for \code{\link{txt2vid_ltx2}}
#'   (\code{text_encoder}, \code{tokenizer}, \code{resident}, ...).
#'
#' @return Invisibly, a data frame with one row per window:
#'   \code{index}, \code{start}, \code{frames}, \code{file},
#'   \code{latents}, \code{prompt} and \code{rendered} (FALSE when kept
#'   from an earlier run).
#'
#' @export
audio2vid_ltx2 <- function(audio, image, prompt, pipeline = NULL,
                           output_dir, width = 768L, height = 512L,
                           window = 97L, carry = 25L, windows = NULL,
                           frame_rate = 24, seed = NULL, resume = TRUE,
                           on_window = NULL, verbose = "progress",
                           generate = NULL, ...) {
    carry <- as.integer(carry)
    width <- as.integer(width)
    height <- as.integer(height)
    if (width %% 32L != 0L || height %% 32L != 0L) {
        stop("width and height must be multiples of 32", call. = FALSE)
    }
    gen <- generate %||% .audio2vid_generator(pipeline)
    if (is.character(audio)) {
        audio <- ltx23_read_audio(audio, sample_rate = 16000L)
    }
    if (!is.matrix(audio) || nrow(audio) != 2L || ncol(audio) < 1L) {
        stop("audio must be a file or a [2, samples] matrix at 16 kHz",
             call. = FALSE)
    }
    rate <- 16000
    bed_frames <- ceiling(ncol(audio) / rate * frame_rate - 1e-6)
    if (is.null(windows)) {
        windows <- ltx2_windows(bed_frames, window, carry)
    }
    windows <- .audio2vid_plan(windows, carry)
    n <- nrow(windows)
    if (!is.character(prompt) || anyNA(prompt) ||
        !length(prompt) %in% c(1L, n)) {
        stop("prompt must be one string or one per window (", n, ")",
             call. = FALSE)
    }
    prompts <- rep_len(prompt, n)
    k <- (carry - 1L) %/% 8L + 1L
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    out <- data.frame(index = seq_len(n), start = windows$start,
                      frames = windows$frames,
                      file = file.path(output_dir,
                                       sprintf("window%02d.mp4", seq_len(n))),
                      stringsAsFactors = FALSE)
    out$latents <- sub("\\.mp4$", ".latents.pt", out$file)
    out$prompt <- prompts
    out$rendered <- FALSE
    stale <- !isTRUE(resume)
    tail <- NULL
    for (j in seq_len(n)) {
        t0 <- Sys.time()
        rec <- list(start = out$start[j], frames = out$frames[j],
                    width = width, height = height, carry = carry,
                    frame_rate = frame_rate, prompt = prompts[j],
                    seed = if (is.null(seed)) NULL else as.integer(seed) + j - 1L,
                    samples = ncol(audio))
        json <- sub("\\.mp4$", ".json", out$file[j])
        if (!stale && .audio2vid_have(out$file[j], out$latents[j], json, rec)) {
            tail <- NULL
        } else {
            stale <- TRUE
            if (j > 1L && is.null(tail)) {
                tail <- torch::torch_load(out$latents[j - 1L])
            }
            args <- list(prompt = prompts[j], width = width, height = height,
                         num_frames = out$frames[j], frame_rate = frame_rate,
                         audio = .audio2vid_slice(audio, out$start[j],
                                                  out$frames[j], frame_rate,
                                                  rate),
                         filename = out$file[j], verbose = verbose)
            if (!is.null(rec$seed)) {
                args$seed <- rec$seed
            }
            if (j == 1L) {
                args$image <- image
            } else {
                args$condition_latents <- tail
            }
            res <- do.call(gen, c(args, list(...)))
            if (!file.exists(out$file[j])) {
                stop("window ", j, " produced no file", call. = FALSE)
            }
            # A COPY, not the narrowed view: torch_save() of a view writes
            # the start of its storage, so the saved tail would be the
            # window's first latent frames and a resumed chain would
            # continue from the wrong place.
            tail <- ltx23_tail_latents(res, k)$to(device = "cpu")$clone()
            rm(res)
            torch::torch_save(tail, out$latents[j])
            writeLines(jsonlite::toJSON(rec, auto_unbox = TRUE, null = "null",
                                        digits = NA), json)
            out$rendered[j] <- TRUE
            gc(verbose = FALSE)
        }
        if (is.function(on_window)) {
            on_window(list(index = j, start = out$start[j],
                           frames = out$frames[j], file = out$file[j],
                           latents = out$latents[j],
                           rendered = out$rendered[j],
                           seconds = as.numeric(difftime(Sys.time(), t0,
                                                         units = "secs"))))
        }
    }
    invisible(out)
}

# 8k + 1, one value
.on_8k1 <- function(n) {
    length(n) == 1L && !is.na(n) && n >= 1L && (n - 1L) %% 8L == 0L
}

# The generator a pipeline implies: a resident handle's, or txt2vid_ltx2
# on a loaded pipeline
.audio2vid_generator <- function(pipeline) {
    if (inherits(pipeline, "diffuseR_resident")) {
        if (!identical(pipeline$model, "ltx")) {
            stop("the resident handle holds ", pipeline$model, ", not ltx",
                 call. = FALSE)
        }
        return(function(prompt, ...) resident_generate(pipeline, prompt, ...))
    }
    if (inherits(pipeline, "ltx23_pipeline")) {
        return(function(prompt, ...) txt2vid_ltx2(prompt, pipeline, ...))
    }
    stop("pipeline must be an ltx23_pipeline or a resident ltx handle",
         call. = FALSE)
}

# A plan checked for a chain: integer start/frames, frames 8k+1 and
# longer than the carry, first at 0, each start `carry` frames before
# the previous window ends
.audio2vid_plan <- function(windows, carry) {
    if (!is.data.frame(windows) || !nrow(windows) ||
        !all(c("start", "frames") %in% names(windows))) {
        stop("windows must be a data frame with start and frames",
             call. = FALSE)
    }
    if (!.on_8k1(carry) || carry < 9L) {
        stop("carry must be 8k+1 and at least 9", call. = FALSE)
    }
    start <- as.integer(windows$start)
    frames <- as.integer(windows$frames)
    n <- length(start)
    if (anyNA(start) || anyNA(frames) ||
        !all(vapply(frames, .on_8k1, logical(1)))) {
        stop("every window's frames must be 8k+1", call. = FALSE)
    }
    if (start[1L] != 0L) {
        stop("the first window must start at frame 0", call. = FALSE)
    }
    if (n > 1L) {
        if (any(frames[-1L] <= carry)) {
            stop("every window after the first must be longer than the carry (",
                 carry, ")", call. = FALSE)
        }
        want <- start[-n] + frames[-n] - carry
        if (any(start[-1L] != want)) {
            j <- which(start[-1L] != want)[1L] + 1L
            stop("window ", j, " starts at ", start[j], "; the chain needs ",
                 want[j - 1L], " (the previous end less the carry)",
                 call. = FALSE)
        }
    }
    data.frame(start = start, frames = frames)
}

# The narration under one window, overlap included; the last window may
# reach past the end, and gets what is left
.audio2vid_slice <- function(audio, start, frames, frame_rate, rate) {
    a <- as.integer(round(start / frame_rate * rate)) + 1L
    b <- min(ncol(audio), as.integer(round((start + frames) / frame_rate * rate)))
    if (a > ncol(audio)) {
        stop("a window starts past the end of the audio", call. = FALSE)
    }
    audio[, a:b, drop = FALSE]
}

# TRUE when a window's files exist and its record matches the plan
.audio2vid_have <- function(file, latents, json, rec) {
    if (!file.exists(file) || !file.exists(latents) || !file.exists(json)) {
        return(FALSE)
    }
    old <- tryCatch(jsonlite::fromJSON(json, simplifyVector = TRUE),
                    error = function(e) NULL)
    if (is.null(old)) {
        return(FALSE)
    }
    for (nm in names(rec)) {
        if (!identical(as.character(old[[nm]]), as.character(rec[[nm]]))) {
            return(FALSE)
        }
    }
    TRUE
}
