# Hosted-artifact policy: exactly the redistributable pair (Apache-2.0,
# ungated), nothing else. flux1 (gated) and ltx (LTX-2 Community
# License) must never appear here.
expect_equal(sort(names(diffuseR:::.prebuilt_nf4_spec)),
             c("flux2", "zimage"))

# Unhosted models refuse without touching the network.
expect_false(diffuseR:::.flux_fetch_prebuilt("flux1", tempfile(),
                                             verbose = FALSE))
expect_false(diffuseR:::.flux_fetch_prebuilt("ltx", tempfile(),
                                             verbose = FALSE))

# .link_or_copy places a file and replaces an existing destination.
src <- tempfile()
writeLines("x", src)
dst <- tempfile()
diffuseR:::.link_or_copy(normalizePath(src), dst)
expect_true(file.exists(dst))
diffuseR:::.link_or_copy(normalizePath(src), dst)
expect_equal(readLines(dst), "x")
unlink(c(src, dst))
