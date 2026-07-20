# progress.R
#
# Shared denoise-loop progress reporting (issue #30). Three verbosity
# levels: "silent" (nothing), "progress" (a one-line generation summary
# plus a progress bar interactively, or periodic step ticks in captured
# logs), "steps" (the full per-phase chatter the generators already
# emit). Logicals map for backwards compatibility: TRUE = "steps",
# FALSE = "silent".

#' Normalize a verbosity flag
#'
#' @param verbose Logical, or one of "silent", "progress", "steps".
#'   TRUE maps to "steps" and FALSE to "silent".
#'
#' @return One of "silent", "progress", "steps".
#'
#' @keywords internal
.verbosity <- function(verbose) {
    if (isTRUE(verbose)) {
        return("steps")
    }
    if (isFALSE(verbose)) {
        return("silent")
    }
    if (is.character(verbose) && length(verbose) == 1L &&
        verbose %in% c("silent", "progress", "steps")) {
        return(verbose)
    }
    stop("'verbose' must be TRUE, FALSE, \"silent\", \"progress\", or \"steps\"",
         call. = FALSE)
}

# Progress reporter for a denoise loop of n steps: returns tick(i, info)
# and done(). At "progress", interactive sessions get a txtProgressBar
# and non-interactive ones a tick line every ~n/10 steps, so captured
# logs stay readable; label (if any) prints once before the loop. With
# per_step = TRUE the caller supplies its own per-step detail through
# tick(info = ) at the "steps" level instead of the bar/ticks.
.denoise_progress <- function(n, label, verbose, per_step = FALSE) {
    level <- .verbosity(verbose)
    if (level == "silent") {
        return(list(tick = function(i, info = NULL) invisible(NULL),
                    done = function() invisible(NULL)))
    }
    if (level == "progress" && !is.null(label) && nzchar(label)) {
        message(label)
    }
    stepwise <- level == "steps" && per_step
    use_bar <- interactive() && !stepwise
    pb <- if (use_bar) {
        utils::txtProgressBar(min = 0, max = n, style = 3)
    } else {
        NULL
    }
    every <- max(1L, as.integer(ceiling(n / 10)))
    tick <- function(i, info = NULL) {
        if (stepwise) {
            message(sprintf("  step %d/%d%s", i, n,
                            if (is.null(info)) "" else paste0(" ", info)))
        } else if (use_bar) {
            utils::setTxtProgressBar(pb, i)
        } else if (i %% every == 0L || i == n) {
            message(sprintf("  step %d/%d", i, n))
        }
        invisible(NULL)
    }
    done <- function() {
        if (use_bar) {
            close(pb)
        }
        invisible(NULL)
    }
    list(tick = tick, done = done)
}
