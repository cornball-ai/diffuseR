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

    # Integer-id BPE with no persistent string maps: R gc cost scales
    # with live object count, and a 151k-binding vocab environment made
    # every gc ~17x slower (measured 12 -> 203 ms), which multiplied
    # into minutes across the allocator-triggered gcs of a generation.
    # All lookups compile to three atomic vectors + a 256-int byte table.
    vocab <- unlist(model$vocab) # named int: piece -> id (0-based)
    piece_names <- names(vocab)

    merges <- model$merges
    if (is.matrix(merges)) {
        a <- merges[, 1]
        b <- merges[, 2]
    } else {
        sp <- regexpr(" ", merges, fixed = TRUE)
        a <- substr(merges, 1L, sp - 1L)
        b <- substr(merges, sp + 1L, nchar(merges))
    }
    id_a <- unname(vocab[match(a, piece_names)])
    id_b <- unname(vocab[match(b, piece_names)])
    id_r <- unname(vocab[match(paste0(a, b), piece_names)])
    ok <- !is.na(id_a) & !is.na(id_b) & !is.na(id_r)
    key <- id_a[ok] * 2097152 + id_b[ok] # ids < 2^21: exact in doubles
    rank <- seq_along(id_a)[ok]
    result <- id_r[ok]
    ord <- order(key)

    # Initial ids for the 256 byte-unicode characters
    byte_chars <- .qwen_byte_table()
    byte_ids <- unname(vocab[match(byte_chars, piece_names)])

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
                   merge_key = key[ord],
                   merge_rank = rank[ord],
                   merge_result = result[ord],
                   byte_ids = byte_ids,
                   n_pieces = length(vocab),
                   split_regex = split_regex,
                   added = added_env,
                   added_contents = added$content[order(-nchar(added$content))],
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
    cat("  vocab:  ", x$n_pieces, "+", length(ls(x$added)), "added\n")
    cat("  path:   ", x$path, "\n")
    invisible(x)
}

# Rank-based BPE merge loop over token ids: adjacent-pair keys are
# looked up in the sorted merge table via findInterval (C binary
# search), vectorized across all pairs per iteration
.qwen_bpe_merge_ids <- function(ids, tokenizer) {
    mk <- tokenizer$merge_key
    while (length(ids) > 1L) {
        keys <- ids[-length(ids)] * 2097152 + ids[-1L]
        pos <- findInterval(keys, mk)
        hit <- pos > 0L
        hit[hit] <- mk[pos[hit]] == keys[hit]
        if (!any(hit)) {
            break
        }
        ranks <- rep(Inf, length(keys))
        ranks[hit] <- tokenizer$merge_rank[pos[hit]]
        i <- which.min(ranks)
        ids[i] <- tokenizer$merge_result[pos[i]]
        ids <- ids[-(i + 1L)]
    }
    ids
}

# Encode one plain-text segment (no added tokens inside)
.qwen_encode_segment <- function(text, tokenizer) {
    if (!nzchar(text)) {
        return(integer(0))
    }
    text <- enc2utf8(text)
    m <- gregexpr(tokenizer$split_regex, text, perl = TRUE)[[1]]
    pre_tokens <- regmatches(text, list(m))[[1]]

    out <- lapply(pre_tokens, function(tok) {
        bytes <- as.integer(charToRaw(tok))
        .qwen_bpe_merge_ids(tokenizer$byte_ids[bytes + 1L], tokenizer)
    })
    as.integer(unlist(out))
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
                    gregexpr(content, chunk$text, fixed = TRUE)))
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
#' With \code{chat_template = TRUE} each prompt is wrapped as a single
#' user turn with the generation prompt, matching
#' \code{apply_chat_template(..., add_generation_prompt = TRUE)}. With
#' \code{enable_thinking = FALSE} (the FLUX.2 klein pipeline behavior)
#' the template closes with an empty thinking block; with
#' \code{enable_thinking = TRUE} (the Z-Image pipeline behavior) it ends
#' at the assistant turn. Right-pads with \code{<|endoftext|>}.
#'
#' @param tokenizer A \code{\link{qwen_bpe_tokenizer}}.
#' @param texts Character vector of prompts.
#' @param max_length Integer. Fixed sequence length (klein: 512). NULL
#'   for no truncation/padding.
#' @param chat_template Logical. Wrap in the Qwen3 chat template.
#' @param enable_thinking Logical. Leave the model's thinking enabled
#'   (no empty think block). Default FALSE.
#'
#' @return List with \code{input_ids} and \code{attention_mask} integer
#'   matrices [length(texts), max_length] (ragged lists when
#'   \code{max_length} is NULL). Ids are 0-based.
#'
#' @export
encode_qwen <- function(tokenizer, texts, max_length = 512L,
                        chat_template = TRUE, enable_thinking = FALSE) {
    stopifnot(inherits(tokenizer, "qwen_tokenizer"))

    encode_one <- function(text) {
        if (chat_template) {
            text <- paste0("<|im_start|>user\n", text, "<|im_end|>\n",
                           "<|im_start|>assistant\n",
                           if (!enable_thinking) "<think>\n\n</think>\n\n")
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
