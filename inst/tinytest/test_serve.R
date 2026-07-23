# serve(): HTTP parser, router, auth, and limits - pure R, no sockets,
# no models (the generate closure is faked).

library(diffuseR)

mkreq <- function(method, path, body = "", headers = character(0)) {
  raw_body <- charToRaw(body)
  # NB: paste0(character(0), "\r\n", collapse = "") is "\r\n", not ""
  # (with collapse set, zero-length args become "") - hence the guard
  hdr_block <- if (length(headers)) {
    paste0(paste0(headers, "\r\n"), collapse = "")
  } else {
    ""
  }
  head <- paste0(method, " ", path, " HTTP/1.1\r\n",
                 "Content-Length: ", length(raw_body), "\r\n",
                 hdr_block, "\r\n")
  con <- rawConnection(c(charToRaw(head), raw_body), open = "rb")
  on.exit(close(con))
  diffuseR:::.dserve_read_request(con, max_body = 1024L^2)
}
jbody <- function(resp) jsonlite::fromJSON(as.character(resp$body), simplifyVector = FALSE)

# --- parser ------------------------------------------------------------------------

req <- mkreq("GET", "/health")
expect_equal(req$method, "GET")
expect_equal(req$path, "/health")
expect_equal(length(req$body), 0L)

req <- mkreq("POST", "/x", body = '{"a":1}',
             headers = "Content-Type: application/json")
expect_equal(rawToChar(req$body), '{"a":1}')
expect_equal(req$headers[["content-type"]], "application/json")

# too-large bodies are flagged, not read. (Assign the raw first:
# rawConnection uses deparse(substitute(object)) as the connection
# name, and a long inline expression deparses to length > 1.)
big_bytes <- charToRaw(paste0(
  "POST /x HTTP/1.1\r\nContent-Length: 100\r\n\r\n",
  paste(rep("a", 100), collapse = "")))
con <- rawConnection(big_bytes, open = "rb")
big <- diffuseR:::.dserve_read_request(con, max_body = 4L)
close(con)
expect_true(isTRUE(big$too_large))

# --- router: health, 404, auth -----------------------------------------------------

state <- list(model = "fake", video = FALSE,
              max_pixels = 1024L^2, max_frames = 161L)

r <- diffuseR:::.dserve_route(mkreq("GET", "/health"), state)
expect_equal(r$status, 200L)
expect_equal(jbody(r)$model, "fake")

expect_equal(diffuseR:::.dserve_route(mkreq("GET", "/nope"), state)$status,
             404L)

auth_state <- state
auth_state$token <- "s3cret"
expect_equal(diffuseR:::.dserve_route(mkreq("GET", "/health"),
                                      auth_state)$status, 401L)
ok <- diffuseR:::.dserve_route(
  mkreq("GET", "/health", headers = "Authorization: Bearer s3cret"),
  auth_state)
expect_equal(ok$status, 200L)

# --- image handler: caps, validation, b64 round trip -------------------------------

state$generate <- function(prompt, width, height, seed = NULL, steps = NULL) {
  array(runif(height * width * 3), dim = c(height, width, 3L))
}

bad <- diffuseR:::.dserve_route(
  mkreq("POST", "/v1/images/generations", body = '{"size":"64x64"}'), state)
expect_equal(bad$status, 400L)   # missing prompt

over <- diffuseR:::.dserve_route(
  mkreq("POST", "/v1/images/generations",
        body = '{"prompt":"x","size":"4096x4096"}'), state)
expect_equal(over$status, 400L)
expect_true(grepl("exceeds limits", jbody(over)$error$message))

good <- diffuseR:::.dserve_route(
  mkreq("POST", "/v1/images/generations",
        body = '{"prompt":"x","size":"64x64","seed":1}'), state)
expect_equal(good$status, 200L)
png_bytes <- jsonlite::base64_dec(jbody(good)$data[[1]]$b64_json)
expect_identical(png_bytes[1:4], as.raw(c(0x89, 0x50, 0x4e, 0x47)))
img <- png::readPNG(png_bytes)
expect_equal(dim(img)[1:2], c(64L, 64L))

# wrong endpoint for the hosted model type
expect_equal(diffuseR:::.dserve_route(
  mkreq("POST", "/v1/videos/generations", body = '{"prompt":"x"}'),
  state)$status, 400L)

# --- video caps (no model needed: limits check precedes generation) ----------------

vstate <- list(model = "fake", video = TRUE,
               max_pixels = 1024L^2, max_frames = 161L)
overv <- diffuseR:::.dserve_route(
  mkreq("POST", "/v1/videos/generations",
        body = '{"prompt":"x","width":2048,"height":2048}'), vstate)
expect_equal(overv$status, 400L)
overf <- diffuseR:::.dserve_route(
  mkreq("POST", "/v1/videos/generations",
        body = '{"prompt":"x","num_frames":9999}'), vstate)
expect_equal(overf$status, 400L)
