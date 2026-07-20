# .verbosity normalization and .denoise_progress output. These tests
# run non-interactively, so the "progress" level takes the periodic
# tick path (the txtProgressBar branch needs interactive()).

verbosity <- diffuseR:::.verbosity
progress <- diffuseR:::.denoise_progress

expect_equal(verbosity(TRUE), "steps")
expect_equal(verbosity(FALSE), "silent")
expect_equal(verbosity("silent"), "silent")
expect_equal(verbosity("progress"), "progress")
expect_equal(verbosity("steps"), "steps")
expect_error(verbosity("loud"))
expect_error(verbosity(1))
expect_error(verbosity(c("progress", "steps")))

run_loop <- function(n, verbose, per_step = FALSE, label = NULL,
                     info = NULL) {
    capture.output({
        pb <- progress(n, label, verbose, per_step = per_step)
        for (i in seq_len(n)) {
            pb$tick(i, info)
        }
        pb$done()
    }, type = "message")
}

# silent: nothing at all
expect_equal(length(run_loop(20, FALSE)), 0L)
expect_equal(length(run_loop(20, "silent", label = "denoise: 20 steps")), 0L)

# progress, non-interactive: label plus a tick every ceiling(n/10) steps
out <- run_loop(20, "progress", label = "denoise: 20 steps")
expect_equal(out[1], "denoise: 20 steps")
expect_equal(out[-1], sprintf("  step %d/20", seq(2, 20, by = 2)))

# the final step always ticks, even off-cadence
out <- run_loop(7, "progress")
expect_true("  step 7/7" %in% out)

# steps with per_step: one line per step carrying the caller's info
out <- run_loop(3, "steps", per_step = TRUE, info = "(sigma 0.9)")
expect_equal(out, sprintf("  step %d/3 (sigma 0.9)", 1:3))

# steps with per_step: label is progress-only, so it does not print
out <- run_loop(3, "steps", per_step = TRUE, label = "stage 1")
expect_equal(out, sprintf("  step %d/3", 1:3))

# steps without per_step (flux-family loops): same ticks as progress
out <- run_loop(20, TRUE)
expect_equal(out, sprintf("  step %d/20", seq(2, 20, by = 2)))
