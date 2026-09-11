# .staged_on / .staged_onload (R/staging.R): the checks a phase makes
# before moving a pinned component. Pure fakes, no torch, no GPU: a
# "pair" is anything with $live$device, $live$set_data and $pinned$to,
# which is all the helpers touch.
#
# The case that matters is the PARTIAL one. An onload that dies partway
# leaves the first pairs on the card and the rest on the host. Probing
# the first pair alone called that resident, so every later phase
# skipped the onload and failed on a device mismatch until the process
# ended (the gpuhost's ltx-2.3 entry, 2026-09-10).

library(diffuseR)
staged_on <- diffuseR:::.staged_on
staged_onload <- diffuseR:::.staged_onload
device_spec <- diffuseR:::.device_spec

# A fake pair whose live tensor sits on `type` (and card `index`),
# recording every set_data it receives.
fake_pair <- function(type, index = if (type == "cpu") NULL else 0L) {
    log <- new.env()
    log$calls <- list()
    live <- list(device = list(type = type, index = index),
                 set_data = function(x) {
                     log$calls[[length(log$calls) + 1L]] <- x
                 })
    pinned <- list(to = function(device, non_blocking = FALSE) {
        paste0("copy->", device)
    })
    list(live = live, pinned = pinned, log = log)
}
calls <- function(p) p$log$calls

# Device spec: strings with and without an index, and a torch_device.
expect_equal(device_spec("cuda"), list(type = "cuda", index = NA_integer_))
expect_equal(device_spec("cuda:1"), list(type = "cuda", index = 1L))
expect_equal(device_spec("cpu"), list(type = "cpu", index = NA_integer_))
expect_equal(device_spec(structure(list(type = "cuda", index = 1),
                                   class = "torch_device")),
             list(type = "cuda", index = 1L))
expect_equal(device_spec(structure(list(type = "cuda", index = NULL),
                                   class = "torch_device")),
             list(type = "cuda", index = NA_integer_))

# All on the card: resident.
st <- list(fake_pair("cuda"), fake_pair("cuda"), fake_pair("cuda"))
expect_true(staged_on(st, "cuda"))
expect_true(staged_on(st, "cuda:0"))
expect_false(staged_on(st, "cpu"))

# All on the host: not resident.
st <- list(fake_pair("cpu"), fake_pair("cpu"))
expect_false(staged_on(st, "cuda"))
expect_true(staged_on(st, "cpu"))

# PARTIAL: first pair on the card, second on the host. The first-pair
# probe said "resident"; the whole-set check must not.
st <- list(fake_pair("cuda"), fake_pair("cpu"))
expect_false(staged_on(st, "cuda"))

# THE WRONG CARD is not this card. A request naming an index accepts only
# that index; a request without one accepts any card.
st <- list(fake_pair("cuda", 0L), fake_pair("cuda", 1L))
expect_false(staged_on(st, "cuda:0"))
expect_false(staged_on(st, "cuda:1"))
expect_true(staged_on(st, "cuda"))
expect_true(staged_on(list(fake_pair("cuda", 1L)), "cuda:1"))

# An unreadable pair is not resident.
st <- list(fake_pair("cuda"), list(live = NULL, pinned = NULL))
expect_false(staged_on(st, "cuda"))

# Empty staging holds nothing to move.
expect_true(staged_on(list(), "cuda"))

# Onload is idempotent per pair: resident pairs are untouched, the rest
# are copied. A partial onload is completed, not restarted.
st <- list(fake_pair("cuda"), fake_pair("cpu"),
           fake_pair("cuda"), fake_pair("cpu"))
staged_onload(st, "cuda")
expect_equal(length(calls(st[[1]])), 0L)
expect_equal(calls(st[[2]]), list("copy->cuda"))
expect_equal(length(calls(st[[3]])), 0L)
expect_equal(calls(st[[4]]), list("copy->cuda"))

# A fully resident set is a no-op, whichever spelling names the card.
st <- list(fake_pair("cuda"), fake_pair("cuda"))
staged_onload(st, "cuda:0")
expect_equal(length(calls(st[[1]])), 0L)
expect_equal(length(calls(st[[2]])), 0L)

# A pair on another card IS moved when a particular card was asked for.
st <- list(fake_pair("cuda", 0L), fake_pair("cuda", 1L))
staged_onload(st, "cuda:1")
expect_equal(calls(st[[1]]), list("copy->cuda:1"))
expect_equal(length(calls(st[[2]])), 0L)
