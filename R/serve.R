# serve.R
# Minimal HTTP server exposing diffuseR over OpenAI-style endpoints.
# Built on base R sockets (serverSocket/socketAccept) so it adds no
# dependencies and runs as a single persistent process: the model loads
# once and stays resident (no fork, so the CUDA context is never
# invalidated). Requests are served one at a time - the natural fit for
# a single-GPU box. Same skeleton as whisper::serve / chatterbox::serve
# (cornball serve ports: whisper 7809, chatterbox 7810, qwen3 7811,
# diffuseR 7812).

# (Per-prompt caches live in server state, created per serve() call:
# a bounded LRU of connector outputs (~9 MB each), never the raw
# Gemma stacks (~0.4 GB each) - a persistent server must not grow
# with prompt count.)

#' Serve diffuseR over HTTP
#'
#' Starts a blocking HTTP server that loads one model and answers
#' OpenAI-style generation requests. Never downloads weights: if the
#' model's artifacts are missing, startup stops with the loader's
#' pointer to the explicit \code{download_*()} function.
#'
#' Endpoints:
#' \itemize{
#'   \item \code{GET /health} - liveness probe, returns
#'     \code{{"status":"ok","model":...}}. The server is
#'     single-threaded, so health only answers between requests.
#'   \item \code{POST /v1/images/generations} - image models. JSON body
#'     \code{{prompt, size, seed, steps}} (\code{size} like
#'     \code{"1024x1024"}; \code{n > 1} is not supported). Returns
#'     \code{{created, data: [{b64_json}]}} with a base64 PNG.
#'   \item \code{POST /v1/videos/generations} - \code{model = "ltx"}
#'     only. JSON body \code{{prompt, width, height, num_frames,
#'     frame_rate, seed}}. Returns raw \code{video/mp4} bytes. Video
#'     generation takes minutes: give your client a matching timeout.
#' }
#'
#' The server is single-threaded and runs until interrupted. Run it
#' under a process supervisor (systemd, tmux); an example unit ships
#' with the package: \code{system.file("diffuser.service",
#' package = "diffuseR")}.
#'
#' Security: base R's \code{serverSocket} binds all interfaces, so the
#' server is reachable by anything that can reach the machine. Keep it
#' behind a firewall or reverse proxy, and/or set \code{token}: when
#' set, every request must carry \code{Authorization: Bearer <token>}
#' or it is refused with 401. Generation size is capped by
#' \code{max_pixels}/\code{max_frames}; oversized requests get 400. A
#' CUDA out-of-memory during a request answers 500 and then exits the
#' process (status 70) so a supervisor restarts it with clean GPU
#' state rather than serving on with stranded components.
#'
#' @param port Integer. TCP port. Default 7812 (cornball serve range: whisper 7809, chatterbox 7810, qwen3 TTS 7811).
#' @param model One of "flux2", "zimage", "flux1" (images) or "ltx"
#'   (video). SD 2.1/SDXL are not served yet.
#' @param device Character. "cuda" or "cpu".
#' @param token Character or NULL. Shared secret; when set, requests
#'   must send \code{Authorization: Bearer <token>}.
#' @param max_pixels Integer. Maximum width x height accepted (images
#'   and video frames). Default 1024^2.
#' @param max_frames Integer. Maximum video frame count. Default 161.
#' @param max_steps Integer. Maximum image inference steps. Default 50.
#' @param max_pixel_frames Numeric. Joint video budget: width x height
#'   x frames must stay under it (NULL = max_pixels x 121, so full-
#'   resolution clips top out at 121 frames and longer clips must
#'   shrink spatially).
#' @param max_prompts Integer. Bound on the per-prompt connector-embed
#'   cache for "ltx" (~9 MB per entry, LRU-evicted). Default 32.
#' @param timeout Integer. Per-connection I/O timeout in seconds.
#' @param max_body Integer. Maximum request body bytes. Default 1 MB
#'   (bodies are JSON).
#' @param warmup Logical. Image models: run one small generation at
#'   startup so the first request doesn't pay tracing and allocator
#'   growth. Ignored for "ltx".
#'
#' @return Does not return normally; runs until interrupted.
#' @export
serve <- function(port = 7812L, model = c("flux2", "zimage", "flux1", "ltx"),
                  device = "cuda", token = NULL, max_pixels = 1024L ^ 2,
                  max_frames = 161L, max_steps = 50L,
                  max_pixel_frames = NULL, max_prompts = 32L, timeout = 300L,
                  max_body = 1024L ^ 2, warmup = TRUE) {
    model <- match.arg(model)
    if (is.null(max_pixel_frames)) {
        # Joint video budget: full max_pixels only up to 121 frames;
        # longer clips must shrink spatially
        max_pixel_frames <- as.numeric(max_pixels) * 121
    }

    message("Loading ", model, " on ", device, " (no downloads; run the ",
            "download_*() function first if artifacts are missing)...")
    state <- .dserve_load(model, device, max_prompts = max_prompts)
    state$token <- token
    state$max_pixels <- as.integer(max_pixels)
    state$max_frames <- as.integer(max_frames)
    state$max_steps <- as.integer(max_steps)
    state$max_pixel_frames <- as.numeric(max_pixel_frames)
    message("Model loaded.")

    if (isTRUE(warmup) && model != "ltx") {
        message("Warming up...")
        tryCatch({
            invisible(state$generate("a red square", width = 256L,
                                     height = 256L, seed = 1L))
            message("Warmup done.")
        }, error = function(e) {
            # A CUDA OOM here strands GPU components before the socket
            # even opens: fail startup so the supervisor sees it
            if (grepl("CUDA out of memory", conditionMessage(e),
                      fixed = TRUE)) {
                stop("warmup hit CUDA out of memory; refusing to serve ",
                     "with stranded GPU state: ", conditionMessage(e),
                     call. = FALSE)
            }
            message("warmup skipped: ", conditionMessage(e))
        })
    }

    srv <- serverSocket(port)
    on.exit(close(srv), add = TRUE)
    message("diffuseR::serve listening on port ", port, " (interrupt to stop)")

    repeat {
        con <- tryCatch(
                        socketAccept(srv, blocking = TRUE, open = "r+b", timeout = timeout),
                        error = function(e) {
            message("accept error: ", conditionMessage(e))
            Sys.sleep(0.5)
            NULL
        }
        )
        if (is.null(con)) {
            next
        }
        resp <- NULL
        tryCatch({
            req <- .dserve_read_request(con, max_body)
            if (!is.null(req)) {
                resp <- tryCatch(
                                 .dserve_route(req, state),
                                 error = function(e) {
                    r <- .dserve_err(500L, conditionMessage(e))
                    if (grepl("CUDA out of memory",
                              conditionMessage(e),
                              fixed = TRUE)) {
                        attr(r, "fatal") <- TRUE
                    }
                    r
                }
                )
                .dserve_send(con, resp$status, resp$content_type, resp$body)
            }
        },
                 error = function(e) message("request error: ", conditionMessage(e)),
                 finally = {
            try(close(con), silent = TRUE)
            gc(verbose = FALSE) # bound dead handles; keep the warm pool
        })
        if (exists("resp", inherits = FALSE) && isTRUE(attr(resp, "fatal"))) {
            message("CUDA out of memory: exiting for a clean supervisor restart")
            quit(save = "no", status = 70L)
        }
    }
}

# Load the served model once; return generate closures + metadata
.dserve_load <- function(model, device, max_prompts = 32L) {
    if (model == "ltx") {
        data_dir <- tools::R_user_dir("diffuseR", "data")
        # Serve a COMPLETE artifact only (a partial/interrupted quantize
        # must not shadow a valid one), and when both precisions are
        # built, follow the machine-aware recommendation - on a card
        # too small for resident nf4, recommend() prescribes streamed
        # fp8 (which is also what download_ltx2() produces by default)
        valid_artifact <- function(d) {
            mp <- file.path(d, "manifest.json")
            file.exists(mp) && isTRUE(tryCatch({
                m <- jsonlite::fromJSON(mp)
                length(m$shards) > 0 &&
                all(file.exists(file.path(d, m$shards)))
            }, error = function(e) FALSE))
        }
        dirs <- c(nf4 = file.path(data_dir, "ltx2.3-nf4"),
                  fp8 = file.path(data_dir, "ltx2.3-fp8"))
        avail <- dirs[vapply(dirs, valid_artifact, logical(1))]
        if (!length(avail)) {
            stop("No complete LTX-2.3 artifact under ", data_dir,
                 "; run download_ltx2() (fp8) or ltx23_quantize_nf4() first.",
                 call. = FALSE)
        }
        pick <- if (length(avail) == 1L) {
            avail[[1]]
        } else {
            prec <- tryCatch(recommend("ltx")$precision,
                             error = function(e) "nf4")
            if (prec %in% names(avail)) {
                avail[[prec]]
            } else {
                avail[[1]]
            }
        }
        pipe <- ltx23_load_pipeline(pick, device = device, verbose = FALSE)
        te_dir <- file.path(data_dir, "gemma3-nf4")
        if (!dir.exists(te_dir)) {
            stop("Gemma3 NF4 artifact not found at ", te_dir,
                 "; run gemma3_quantize_nf4() first (serving loads the ",
                 "quantized encoder, not the raw fp32 weights).",
                 call. = FALSE)
        }
        if (!requireNamespace("hfhub", quietly = TRUE)) {
            stop("Serving ltx requires the hfhub package (tokenizer files).",
                 call. = FALSE)
        }
        tok_dir <- dirname(hfhub::hub_download("Lightricks/LTX-2",
                "tokenizer/tokenizer.json", local_files_only = TRUE))
        te <- load_gemma3_text_encoder(te_dir, device = "cpu", verbose = FALSE)
        tok <- gemma3_tokenizer(tok_dir)
        enc_dev <- if (identical(device, "cuda") &&
            torch::cuda_is_available()) {
            "cuda"
        } else {
            "cpu"
        }
        # Bounded LRU of CONNECTOR outputs (~9 MB each): the raw Gemma
        # stack (~0.4 GB) is encoded, projected, and dropped per prompt
        cache <- new.env(parent = emptyenv())
        order <- character(0)
        embeds <- function(prompt) {
            key <- paste0("p:", prompt)
            e <- cache[[key]]
            if (!is.null(e)) {
                order <<- c(setdiff(order, key), key)
                return(e)
            }
            raw <- encode_with_gemma3(prompt, model = te, tokenizer = tok,
                                      max_sequence_length = 1024L,
                                      device = enc_dev, verbose = FALSE)
            conn <- torch::with_no_grad(pipe$connectors(
                    raw$prompt_embeds$to(device = "cpu",
                        dtype = torch::torch_bfloat16()),
                    raw$prompt_attention_mask$to(device = "cpu")))
            e <- list(video_text_embedding = conn$video_text_embedding,
                      audio_text_embedding = conn$audio_text_embedding,
                      attention_mask = conn$attention_mask)
            rm(raw, conn)
            cache[[key]] <- e
            order <<- c(order, key)
            while (length(order) > max_prompts) {
                rm(list = order[[1]], envir = cache)
                order <<- order[-1]
            }
            gc(verbose = FALSE)
            e
        }
        list(model = model, video = TRUE, pipe = pipe, embeds = embeds,
             device = device)
    } else {
        loader <- switch(model, flux1 = flux_load_pipeline,
                         flux2 = flux2_load_pipeline,
                         zimage = zimage_load_pipeline)
        pipe <- loader(device = device, verbose = FALSE)
        genfn <- switch(model, flux1 = txt2img_flux, flux2 = txt2img_flux2,
                        zimage = txt2img_zimage)
        generate <- function(prompt, width, height, seed = NULL, steps = NULL) {
            args <- list(prompt = prompt, pipeline = pipe,
                         width = as.integer(width),
                         height = as.integer(height), seed = seed,
                         save_file = FALSE, verbose = FALSE)
            if (!is.null(steps)) {
                args$num_inference_steps <- as.integer(steps)
            }
            res <- do.call(genfn, args)
            # The generators return list(image, metadata)
            if (is.list(res) && !is.null(res$image)) {
                res$image
            } else {
                res
            }
        }
        list(model = model, video = FALSE, generate = generate, device = device)
    }
}

.dserve_route <- function(req, state) {
    if (isTRUE(req$too_large)) {
        return(.dserve_err(413L, "request body too large"))
    }
    if (!is.null(state$token)) {
        auth <- req$headers[["authorization"]]
        if (!identical(auth, paste("Bearer", state$token))) {
            return(.dserve_err(401L, "unauthorized"))
        }
    }
    path <- sub("\\?.*$", "", req$path)

    if (identical(req$method, "GET") && path == "/health") {
        return(.dserve_json(list(status = "ok", model = state$model)))
    }
    if (identical(req$method, "POST") && path == "/v1/images/generations") {
        if (isTRUE(state$video)) {
            return(.dserve_err(400L,
                               "this server hosts a video model; POST /v1/videos/generations"))
        }
        return(.dserve_image(req, state))
    }
    if (identical(req$method, "POST") && path == "/v1/videos/generations") {
        if (!isTRUE(state$video)) {
            return(.dserve_err(400L,
                               "this server hosts an image model; POST /v1/images/generations"))
        }
        return(.dserve_video(req, state))
    }
    .dserve_err(404L, "not found")
}

# One length-1 atomic value or the default; JSON arrays/objects sent
# for scalar fields become NA so validation answers 400, not 500
.dserve_scalar <- function(x, default = NULL) {
    if (is.null(x)) {
        return(default)
    }
    if (!is.atomic(x) || length(x) != 1L) {
        return(NA)
    }
    x
}

.dserve_prompt_ok <- function(p) {
    is.atomic(p) && length(p) == 1L && is.character(p) && !is.na(p) &&
    nzchar(p)
}

.dserve_body <- function(req) {
    if (!length(req$body)) {
        return(NULL)
    }
    tryCatch(jsonlite::fromJSON(rawToChar(req$body), simplifyVector = TRUE),
             error = function(e) NULL)
}

.dserve_image <- function(req, state) {
    body <- .dserve_body(req)
    if (is.null(body) || !.dserve_prompt_ok(body$prompt)) {
        return(.dserve_err(400L,
                           "body must be JSON with a single string prompt"))
    }
    n <- .dserve_scalar(body$n, 1L)
    if (is.na(suppressWarnings(as.integer(n))) || as.integer(n) > 1L) {
        return(.dserve_err(400L, "n > 1 is not supported"))
    }
    size <- .dserve_scalar(body$size, "1024x1024")
    if (!is.character(size) || is.na(size)) {
        return(.dserve_err(400L, "size must look like 1024x1024"))
    }
    wh <- suppressWarnings(as.integer(strsplit(size, "x", fixed = TRUE)[[1]]))
    if (length(wh) != 2L || anyNA(wh)) {
        return(.dserve_err(400L, "size must look like 1024x1024"))
    }
    if (anyNA(wh) || any(wh < 16L) || wh[1] * wh[2] > state$max_pixels) {
        return(.dserve_err(400L, sprintf(
                    "request exceeds limits (max %d pixels, min side 16)",
                    state$max_pixels)))
    }
    steps <- .dserve_scalar(body$steps)
    if (!is.null(steps)) {
        steps <- suppressWarnings(as.integer(steps))
        if (is.na(steps) || steps < 1L || steps > state$max_steps) {
            return(.dserve_err(400L, sprintf(
                        "steps must be between 1 and %d", state$max_steps)))
        }
    }
    seed <- .dserve_scalar(body$seed)
    if (!is.null(seed)) {
        seed <- suppressWarnings(as.integer(seed))
        if (is.na(seed)) {
            return(.dserve_err(400L, "seed must be a single integer"))
        }
    }
    img <- state$generate(body$prompt, width = wh[1], height = wh[2],
                          seed = seed, steps = steps)
    png <- png::writePNG(img)
    .dserve_json(list(
                      created = as.integer(Sys.time()),
                      data = list(list(b64_json = jsonlite::base64_enc(png)))
        ))
}

.dserve_video <- function(req, state) {
    body <- .dserve_body(req)
    if (is.null(body) || !.dserve_prompt_ok(body$prompt)) {
        return(.dserve_err(400L,
                           "body must be JSON with a single string prompt"))
    }
    w <- suppressWarnings(as.integer(.dserve_scalar(body$width, 768L)))
    h <- suppressWarnings(as.integer(.dserve_scalar(body$height, 512L)))
    nf <- suppressWarnings(as.integer(.dserve_scalar(body$num_frames, 121L)))
    fr <- suppressWarnings(as.numeric(.dserve_scalar(body$frame_rate, 24)))
    if (anyNA(c(w, h, nf)) || w < 32L || h < 32L || nf < 9L ||
        w * h > state$max_pixels || nf > state$max_frames ||
        as.numeric(w) * h * nf > state$max_pixel_frames) {
        return(.dserve_err(400L, sprintf(
                    "request exceeds limits (max %d pixels, %d frames, %.0f pixel-frames)",
                    state$max_pixels, state$max_frames, state$max_pixel_frames)))
    }
    # frame_rate scales the audio-latent length inversely: bound it
    if (is.na(fr) || fr < 12 || fr > 60) {
        return(.dserve_err(400L, "frame_rate must be between 12 and 60"))
    }
    vseed <- .dserve_scalar(body$seed)
    if (!is.null(vseed)) {
        vseed <- suppressWarnings(as.integer(vseed))
        if (is.na(vseed)) {
            return(.dserve_err(400L, "seed must be a single integer"))
        }
    }
    out <- tempfile(fileext = ".mp4")
    on.exit(unlink(out), add = TRUE)
    txt2vid_ltx2(
                 prompt = body$prompt,
                 pipeline = state$pipe,
                 connector_embeds = state$embeds(body$prompt),
                 width = w,
                 height = h,
                 num_frames = nf,
                 frame_rate = fr,
                 seed = vseed,
                 device = state$device, dtype = "bfloat16",
                 filename = out, verbose = FALSE
    )
    bytes <- readBin(out, "raw", n = file.size(out))
    list(status = 200L, content_type = "video/mp4", body = bytes)
}

# ---- HTTP plumbing (same shape as whisper::serve) ---------------------------

.dserve_read_request <- function(con, max_body) {
    term <- as.raw(c(13L, 10L, 13L, 10L))
    buf <- raw(0)
    max_header <- 65536L
    repeat {
        b <- readBin(con, "raw", n = 1L)
        if (length(b) == 0L) {
            return(NULL)
        }
        buf <- c(buf, b)
        n <- length(buf)
        if (n >= 4L && identical(buf[(n - 3L):n], term)) {
            break
        }
        if (n > max_header) {
            return(NULL)
        }
    }

    lines <- strsplit(rawToChar(buf), "\r\n", fixed = TRUE)[[1]]
    req_line <- strsplit(lines[1], " ", fixed = TRUE)[[1]]
    if (length(req_line) < 2L) {
        return(NULL)
    }
    hdr <- list()
    if (length(lines) > 1L) {
        for (ln in lines[-1L]) {
            if (!nzchar(ln)) {
                next
            }
            pos <- regexpr(":", ln, fixed = TRUE)
            if (pos < 1L) {
                next
            }
            hdr[[tolower(trimws(substr(ln, 1L, pos - 1L)))]] <-
            trimws(substr(ln, pos + 1L, nchar(ln)))
        }
    }
    clen <- suppressWarnings(as.integer(hdr[["content-length"]] %||% "0"))
    if (length(clen) != 1L || is.na(clen) || clen < 0L) {
        clen <- 0L
    }
    if (clen > max_body) {
        return(list(method = req_line[1], path = req_line[2], headers = hdr,
                    body = raw(0), too_large = TRUE))
    }
    body <- raw(0)
    while (length(body) < clen) {
        chunk <- readBin(con, "raw", n = clen - length(body))
        if (length(chunk) == 0L) {
            break
        }
        body <- c(body, chunk)
    }
    list(method = req_line[1], path = req_line[2], headers = hdr, body = body)
}

.dserve_send <- function(con, status, content_type, body) {
    if (is.character(body)) {
        body <- charToRaw(enc2utf8(body))
    }
    reason <- switch(as.character(status), "200" = "OK",
                     "400" = "Bad Request", "401" = "Unauthorized",
                     "404" = "Not Found", "405" = "Method Not Allowed",
                     "413" = "Payload Too Large",
                     "500" = "Internal Server Error", "Unknown")
    head <- paste0(
                   sprintf("HTTP/1.1 %d %s\r\n", status, reason),
                   sprintf("Content-Type: %s\r\n", content_type),
                   sprintf("Content-Length: %d\r\n", length(body)),
                   "Connection: close\r\n\r\n")
    writeBin(c(charToRaw(head), body), con)
    flush(con)
}

.dserve_json <- function(x, status = 200L) {
    list(status = status, content_type = "application/json",
         body = jsonlite::toJSON(x, auto_unbox = TRUE))
}

.dserve_err <- function(status, msg) {
    .dserve_json(list(error = list(message = msg)), status = status)
}
