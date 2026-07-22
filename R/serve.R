# serve.R
# Minimal HTTP server exposing diffuseR over OpenAI-style endpoints.
# Built on base R sockets (serverSocket/socketAccept) so it adds no
# dependencies and runs as a single persistent process: the model loads
# once and stays resident (no fork, so the CUDA context is never
# invalidated). Requests are served one at a time - the natural fit for
# a single-GPU box. Same skeleton as whisper::serve / chatterbox::serve
# (cornball serve ports: whisper 7809, chatterbox 7810, qwen3 7811,
# diffuseR 7812).

# Session memo for per-prompt text embeddings (the expensive part of a
# video request; a track's chunks share one prompt)
.dserve_embed_cache <- new.env(parent = emptyenv())

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
#' @param port Integer. TCP port. Default 7812 (cornball serve range: whisper 7809, chatterbox 7810, qwen3 TTS 7811).
#' @param model One of "flux2", "zimage", "flux1" (images) or "ltx"
#'   (video). SD 2.1/SDXL are not served yet.
#' @param device Character. "cuda" or "cpu".
#' @param timeout Integer. Per-connection I/O timeout in seconds.
#' @param max_body Integer. Maximum request body bytes. Default 1 MB
#'   (bodies are JSON).
#' @param warmup Logical. Image models: run one small generation at
#'   startup so the first request doesn't pay tracing and allocator
#'   growth. Ignored for "ltx".
#'
#' @return Does not return normally; runs until interrupted.
#' @export
serve <- function(port = 7812L,
                  model = c("flux2", "zimage", "flux1", "ltx"),
                  device = "cuda", timeout = 300L,
                  max_body = 1024L^2, warmup = TRUE) {
    model <- match.arg(model)

    message("Loading ", model, " on ", device, " (no downloads; run the ",
            "download_*() function first if artifacts are missing)...")
    state <- .dserve_load(model, device)
    message("Model loaded.")

    if (isTRUE(warmup) && model != "ltx") {
        message("Warming up...")
        tryCatch({
            invisible(state$generate("a red square", width = 256L,
                                     height = 256L, seed = 1L))
            message("Warmup done.")
        }, error = function(e) {
            message("warmup skipped: ", conditionMessage(e))
        })
    }

    srv <- serverSocket(port)
    on.exit(close(srv), add = TRUE)
    message("diffuseR::serve listening on port ", port,
            " (interrupt to stop)")

    repeat {
        con <- tryCatch(
                        socketAccept(srv, blocking = TRUE, open = "r+b",
                                     timeout = timeout),
                        error = function(e) {
                            message("accept error: ", conditionMessage(e))
                            Sys.sleep(0.5)
                            NULL
                        }
        )
        if (is.null(con)) {
            next
        }
        tryCatch({
            req <- .dserve_read_request(con, max_body)
            if (!is.null(req)) {
                resp <- tryCatch(
                                 .dserve_route(req, state),
                                 error = function(e) {
                                     .dserve_err(500L, conditionMessage(e))
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
    }
}

# Load the served model once; return generate closures + metadata
.dserve_load <- function(model, device) {
    if (model == "ltx") {
        pipe <- ltx23_load_pipeline(
            file.path(tools::R_user_dir("diffuseR", "data"), "ltx2.3-nf4"),
            device = device, verbose = FALSE
        )
        te_dir <- file.path(tools::R_user_dir("diffuseR", "data"),
                            "gemma3-nf4")
        if (!dir.exists(te_dir)) {
            stop("Gemma3 NF4 artifact not found at ", te_dir,
                 "; run gemma3_quantize_nf4() first.", call. = FALSE)
        }
        if (!requireNamespace("hfhub", quietly = TRUE)) {
            stop("Serving ltx requires the hfhub package (tokenizer files).",
                 call. = FALSE)
        }
        tok_dir <- dirname(hfhub::hub_download("Lightricks/LTX-2",
                "tokenizer/tokenizer.json", local_files_only = TRUE))
        te <- load_gemma3_text_encoder(te_dir, device = "cpu",
                                       verbose = FALSE)
        tok <- gemma3_tokenizer(tok_dir)
        embeds <- function(prompt) {
            key <- paste0("p:", prompt)
            e <- .dserve_embed_cache[[key]]
            if (is.null(e)) {
                e <- encode_with_gemma3(prompt, model = te, tokenizer = tok,
                    max_sequence_length = 1024L,
                    device = if (torch::cuda_is_available()) "cuda" else "cpu",
                    verbose = FALSE)
                e$prompt_embeds <- e$prompt_embeds$to(device = "cpu")
                e$prompt_attention_mask <-
                e$prompt_attention_mask$to(device = "cpu")
                .dserve_embed_cache[[key]] <- e
            }
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
        generate <- function(prompt, width, height, seed = NULL,
                             steps = NULL) {
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
        list(model = model, video = FALSE, generate = generate,
             device = device)
    }
}

.dserve_route <- function(req, state) {
    if (isTRUE(req$too_large)) {
        return(.dserve_err(413L, "request body too large"))
    }
    path <- sub("\\?.*$", "", req$path)

    if (identical(req$method, "GET") && path == "/health") {
        return(.dserve_json(list(status = "ok", model = state$model)))
    }
    if (identical(req$method, "POST") && path == "/v1/images/generations") {
        if (isTRUE(state$video)) {
            return(.dserve_err(400L, "this server hosts a video model; POST /v1/videos/generations"))
        }
        return(.dserve_image(req, state))
    }
    if (identical(req$method, "POST") && path == "/v1/videos/generations") {
        if (!isTRUE(state$video)) {
            return(.dserve_err(400L, "this server hosts an image model; POST /v1/images/generations"))
        }
        return(.dserve_video(req, state))
    }
    .dserve_err(404L, "not found")
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
    if (is.null(body) || is.null(body$prompt) || !nzchar(body$prompt)) {
        return(.dserve_err(400L, "body must be JSON with a prompt"))
    }
    if (!is.null(body$n) && as.integer(body$n) > 1L) {
        return(.dserve_err(400L, "n > 1 is not supported"))
    }
    size <- if (is.null(body$size)) "1024x1024" else body$size
    wh <- suppressWarnings(as.integer(strsplit(size, "x", fixed = TRUE)[[1]]))
    if (length(wh) != 2L || anyNA(wh)) {
        return(.dserve_err(400L, "size must look like 1024x1024"))
    }
    seed <- if (is.null(body$seed)) NULL else as.integer(body$seed)
    img <- state$generate(body$prompt, width = wh[1], height = wh[2],
                          seed = seed, steps = body$steps)
    png <- png::writePNG(img)
    .dserve_json(list(
        created = as.integer(Sys.time()),
        data = list(list(b64_json = jsonlite::base64_enc(png)))
    ))
}

.dserve_video <- function(req, state) {
    body <- .dserve_body(req)
    if (is.null(body) || is.null(body$prompt) || !nzchar(body$prompt)) {
        return(.dserve_err(400L, "body must be JSON with a prompt"))
    }
    out <- tempfile(fileext = ".mp4")
    on.exit(unlink(out), add = TRUE)
    txt2vid_ltx2(
        prompt = body$prompt,
        pipeline = state$pipe,
        prompt_embeds = state$embeds(body$prompt),
        width = as.integer(body$width %||% 768L),
        height = as.integer(body$height %||% 512L),
        num_frames = as.integer(body$num_frames %||% 121L),
        frame_rate = as.numeric(body$frame_rate %||% 24),
        seed = if (is.null(body$seed)) NULL else as.integer(body$seed),
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
                     "400" = "Bad Request", "404" = "Not Found",
                     "405" = "Method Not Allowed",
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
