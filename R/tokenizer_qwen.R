#' Qwen2 Byte-Level BPE Tokenizer
#'
#' Pure R implementation of the Qwen2 tokenizer (HuggingFace
#' tokenizer.json, BPE model with ByteLevel pre-tokenization), as used by
#' FLUX.2 klein's Qwen3 text encoder. Text is split with the GPT-4-style
#' regex, each pre-token's UTF-8 bytes are mapped through the GPT-2
#' byte-to-unicode table, and rank-based BPE merges produce the ids.
#' Added tokens (\code{<|im_start|>}, \code{<think>}, ...) are split out
#' literally before byte-level encoding.
#'
#' Limitation: the NFC normalizer is not applied (base R has no NFC);
#' input is assumed to already be NFC, which holds for ordinary text.
#'
#' @name tokenizer_qwen
NULL

# GPT-2 byte-to-unicode table: printable bytes map to themselves,
# everything else to 256+n. Returns a character vector indexed by
# byte value + 1.
.qwen_byte_table <- function() {
    bs <- c(33:126, 161:172, 174:255)
    cs <- bs
    n <- 0L
    for (b in 0:255) {
        if (!(b %in% bs)) {
            bs <- c(bs, b)
            cs <- c(cs, 256L + n)
            n <- n + 1L
        }
    }
    out <- character(256)
    out[bs + 1L] <- vapply(cs, intToUtf8, character(1))
    out
}

#' Load a Qwen2 byte-level BPE tokenizer
#'
#' @param tokenizer_path Path to a tokenizer.json (or a directory
#'   containing one).
#'
#' @return A \code{qwen_tokenizer} object.
#'
#' @export
qwen_bpe_tokenizer <- function(tokenizer_path) {
    path <- path.expand(tokenizer_path)
    if (dir.exists(path)) {
        path <- file.path(path, "tokenizer.json")
    }
    if (!file.exists(path)) {
        stop("tokenizer.json not found: ", path)
    }

    tj <- jsonlite::fromJSON(path, simplifyVector = TRUE)
    model <- tj$model
    if (!identical(model$type, "BPE")) {
        stop("Expected a BPE tokenizer, got: ", model$type %||% "none")
    }

    vocab_env <- list2env(as.list(model$vocab), parent = emptyenv())

    merges <- model$merges
    if (is.matrix(merges)) {
        # Pair-format merges [["a", "b"], ...] simplify to a matrix
        keys <- paste(merges[, 1], merges[, 2])
    } else {
        # Legacy "a b" strings
        keys <- merges
    }
    ranks_env <- list2env(stats::setNames(as.list(seq_along(keys)), keys),
                          parent = emptyenv())

    # Pre-tokenization split regex (GPT-4 style)
    split_regex <- NULL
    pres <- tj$pre_tokenizer$pretokenizers
    if (!is.null(pres) && "pattern" %in% names(pres)) {
        split_regex <- pres$pattern$Regex[!is.na(pres$pattern$Regex)][1]
    }
    if (is.null(split_regex) || !nzchar(split_regex)) {
        stop("No Split pre-tokenizer regex found in ", path)
    }

    added <- tj$added_tokens
    added_env <- list2env(stats::setNames(as.list(added$id), added$content),
                          parent = emptyenv())

    structure(
              list(
                   vocab = vocab_env,
                   ranks = ranks_env,
                   split_regex = split_regex,
                   added = added_env,
                   added_contents = added$content[order(-nchar(added$content))],
                   byte_table = .qwen_byte_table(),
                   pad_id = get0("<|endoftext|>", envir = added_env,
                                 ifnotfound = 151643L),
                   path = path
        ),
              class = "qwen_tokenizer"
    )
}

#' @export
print.qwen_tokenizer <- function(x, ...) {
    cat("<qwen_tokenizer>\n")
    cat("  vocab:  ", length(ls(x$vocab)), "+", length(ls(x$added)),
        "added\n")
    cat("  path:   ", x$path, "\n")
    invisible(x)
}

# Rank-based BPE merge loop over byte-unicode symbols
.qwen_bpe_merge <- function(chars, ranks) {
    while (length(chars) > 1L) {
        best_rank <- Inf
        best_i <- 0L
        for (i in seq_len(length(chars) - 1L)) {
            r <- get0(paste(chars[i], chars[i + 1L]), envir = ranks)
            if (!is.null(r) && r < best_rank) {
                best_rank <- r
                best_i <- i
            }
        }
        if (best_i == 0L) {
            break
        }
        chars[best_i] <- paste0(chars[best_i], chars[best_i + 1L])
        chars <- chars[-(best_i + 1L)]
    }
    chars
}

# Encode one plain-text segment (no added tokens inside)
.qwen_encode_segment <- function(text, tokenizer) {
    if (!nzchar(text)) {
        return(integer(0))
    }
    text <- enc2utf8(text)
    m <- gregexpr(tokenizer$split_regex, text, perl = TRUE)[[1]]
    pre_tokens <- regmatches(text, list(m))[[1]]

    ids <- integer(0)
    for (tok in pre_tokens) {
        bytes <- as.integer(charToRaw(tok))
        chars <- tokenizer$byte_table[bytes + 1L]
        chars <- .qwen_bpe_merge(chars, tokenizer$ranks)
        for (piece in chars) {
            id <- get0(piece, envir = tokenizer$vocab)
            if (is.null(id)) {
                stop("Byte-level BPE piece not in vocab: ", piece)
            }
            ids <- c(ids, id)
        }
    }
    as.integer(ids)
}

# Split text on added-token literals (longest first), returning a list
# of list(text=, id=) chunks
.qwen_split_added <- function(text, tokenizer) {
    chunks <- list(list(text = text, id = NA_integer_))
    for (content in tokenizer$added_contents) {
        out <- list()
        for (chunk in chunks) {
            if (!is.na(chunk$id) || !grepl(content, chunk$text, fixed = TRUE)) {
                out[[length(out) + 1L]] <- chunk
                next
            }
            parts <- strsplit(chunk$text, content, fixed = TRUE)[[1]]
            # strsplit drops trailing separators; recover the layout
            n_seps <- lengths(regmatches(chunk$text,
                                         gregexpr(content, chunk$text,
                                                  fixed = TRUE)))
            if (length(parts) == 0L) {
                parts <- ""
            }
            for (i in seq_along(parts)) {
                if (nzchar(parts[i])) {
                    out[[length(out) + 1L]] <- list(text = parts[i],
                                                    id = NA_integer_)
                }
                if (i <= n_seps) {
                    out[[length(out) + 1L]] <- list(
                                                    text = content,
                                                    id = get0(content, envir = tokenizer$added))
                }
            }
        }
        chunks <- out
    }
    chunks
}

#' Encode prompts with the Qwen tokenizer
#'
#' With \code{chat_template = TRUE} (the FLUX.2 klein pipeline behavior)
#' each prompt is wrapped as a single user turn with the generation
#' prompt and a disabled thinking block, matching
#' \code{apply_chat_template(..., add_generation_prompt = TRUE,
#' enable_thinking = FALSE)}. Right-pads with \code{<|endoftext|>}.
#'
#' @param tokenizer A \code{\link{qwen_bpe_tokenizer}}.
#' @param texts Character vector of prompts.
#' @param max_length Integer. Fixed sequence length (klein: 512). NULL
#'   for no truncation/padding.
#' @param chat_template Logical. Wrap in the Qwen3 chat template.
#'
#' @return List with \code{input_ids} and \code{attention_mask} integer
#'   matrices [length(texts), max_length] (ragged lists when
#'   \code{max_length} is NULL). Ids are 0-based.
#'
#' @export
encode_qwen <- function(tokenizer, texts, max_length = 512L,
                        chat_template = TRUE) {
    stopifnot(inherits(tokenizer, "qwen_tokenizer"))

    encode_one <- function(text) {
        if (chat_template) {
            text <- paste0(
                           "<|im_start|>user\n", text, "<|im_end|>\n",
                           "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        }
        ids <- integer(0)
        for (chunk in .qwen_split_added(text, tokenizer)) {
            if (!is.na(chunk$id)) {
                ids <- c(ids, as.integer(chunk$id))
            } else {
                ids <- c(ids, .qwen_encode_segment(chunk$text, tokenizer))
            }
        }
        ids
    }

    all_ids <- lapply(as.character(texts), encode_one)

    if (is.null(max_length)) {
        return(list(
                    input_ids = all_ids,
                    attention_mask = lapply(all_ids, function(x) rep(1L, length(x)))
        ))
    }

    all_ids <- lapply(all_ids, function(ids) {
        if (length(ids) > max_length) ids[seq_len(max_length)] else ids
    })
    n <- length(all_ids)
    input_ids <- matrix(as.integer(tokenizer$pad_id), nrow = n,
                        ncol = max_length)
    attention_mask <- matrix(0L, nrow = n, ncol = max_length)
    for (i in seq_len(n)) {
        len <- length(all_ids[[i]])
        if (len > 0L) {
            input_ids[i, seq_len(len)] <- all_ids[[i]]
            attention_mask[i, seq_len(len)] <- 1L
        }
    }
    list(input_ids = input_ids, attention_mask = attention_mask)
}
