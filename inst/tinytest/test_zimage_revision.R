# Z-Image's support files resolve by an exact revision, so a fleet node can
# load from a bound snapshot with no refs/ and no network.

library(diffuseR)

# A branch name is refused before any device or file work.
expect_error(zimage_load_pipeline(revision = "main"), pattern = "40-character")
expect_error(diffuseR:::.zimage_cached("vae/config.json", "main"),
             pattern = "40-character")

if (!requireNamespace("hfhub", quietly = TRUE)) {
    exit_file("hfhub not installed")
}

# A cache holding only the snapshot: no refs/main. The revision finds the
# file; without it the lookup reports a missing download.
local({
    root <- tempfile("zimage-hub-")
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    rev <- strrep("ab", 20L)
    snap <- file.path(root, "models--Tongyi-MAI--Z-Image-Turbo", "snapshots",
                      rev, "vae")
    dir.create(snap, recursive = TRUE)
    writeLines("{}", file.path(snap, "config.json"))
    old <- Sys.getenv(c("HUGGINGFACE_HUB_CACHE", "HF_HUB_OFFLINE"),
                      unset = NA)
    on.exit({
        for (k in names(old)) {
            if (is.na(old[[k]])) Sys.unsetenv(k) else do.call(Sys.setenv,
                as.list(old[k]))
        }
    }, add = TRUE)
    Sys.setenv(HUGGINGFACE_HUB_CACHE = root, HF_HUB_OFFLINE = "1")
    got <- diffuseR:::.zimage_cached("vae/config.json", rev)
    expect_equal(normalizePath(got), normalizePath(file.path(snap,
                                                             "config.json")))
    expect_error(diffuseR:::.zimage_cached("vae/config.json"),
                 pattern = "Missing vae/config.json")
})
