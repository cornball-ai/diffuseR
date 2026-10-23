#' Gemma 4 SentencePiece-style BPE tokenizer
#'
#' Native R port of the HuggingFace \code{tokenizers} pipeline in the
#' Gemma 4 \code{tokenizer.json} shipped with LTX-2.5: added-token
#' literals split out first, the \code{Replace " " -> U+2581} normalizer,
#' characters mapped to vocab ids with \code{<0xHH>} byte fallback, then
#' rank-ordered BPE merges over the whole string (the \code{Split} on
#' " " pre-tokenizer never matches after the normalizer). The
#' post-processor template decides special tokens; for Gemma 4 it adds
#' none, so there is no BOS.
#'
#' @name tokenizer_gemma4
NULL

#' Load the Gemma 4 tokenizer
#'
#' @param source Path to a \code{tokenizer.json}, a directory holding
#'   one, or an LTX-2.5 text encoder \code{.safetensors} file, which
#'   carries the same JSON in its \code{tokenizer_json} tensor.
#'
#' @return A \code{gemma4_tokenizer}.
#'
#' @export
gemma4_tokenizer <- function(source) {
    tj <- jsonlite::fromJSON(.gemma4_tokenizer_json(source),
                             simplifyVector = TRUE)
    model <- tj$model
    if (!identical(model$type, "BPE")) {
        stop("Expected a BPE tokenizer, got: ", model$type %||% "none")
    }
    norm <- tj$normalizer
    if (!identical(norm$type, "Replace") || is.null(norm$pattern$String)) {
        stop("Unsupported normalizer: ", norm$type %||% "none",
             " (expected a Replace of a literal string)")
    }

    # Same integer-id layout as qwen_bpe_tokenizer(): atomic vectors only,
    # since live environment bindings make every R gc slower
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
    ord <- order(key)

    byte_pieces <- sprintf("<0x%02X>", 0:255)
    added <- tj$added_tokens

    structure(
              list(
                   merge_key = key[ord],
                   merge_rank = seq_along(id_a)[ok][ord],
                   merge_result = id_r[ok][ord],
                   piece_names = piece_names,
                   piece_ids = unname(vocab),
                   byte_ids = unname(vocab[match(byte_pieces, piece_names)]),
                   byte_fallback = isTRUE(model$byte_fallback),
                   unk_id = unname(vocab[model$unk_token %||% "<unk>"]),
                   norm_from = norm$pattern$String,
                   norm_to = norm$content,
                   added = list2env(stats::setNames(as.list(added$id),
                                                    added$content),
                                    parent = emptyenv()),
                   added_contents = added$content[order(-nchar(added$content))],
                   template = .gemma4_template_ids(tj$post_processor, added),
                   pad_id = added$id[match("<pad>", added$content)] %||% 0L
        ),
              class = "gemma4_tokenizer"
    )
}

#' @export
print.gemma4_tokenizer <- function(x, ...) {
    cat("<gemma4_tokenizer>\n")
    cat("  vocab:  ", length(x$piece_ids), "pieces,",
        length(x$merge_key), "merges\n")
    cat("  special:", length(x$template$prefix), "prefix /",
        length(x$template$suffix), "suffix ids per sequence\n")
    invisible(x)
}

# The tokenizer.json text from a json path, a directory, or the
# tokenizer_json uint8 tensor of an LTX-2.5 encoder file
.gemma4_tokenizer_json <- function(source) {
    path <- path.expand(source)
    if (dir.exists(path)) {
        path <- file.path(path, "tokenizer.json")
    }
    if (!file.exists(path)) {
        stop("Tokenizer source not found: ", source)
    }
    if (!grepl("\\.safetensors$", path)) {
        return(paste(readLines(path, warn = FALSE, encoding = "UTF-8"),
                     collapse = "\n"))
    }
    handle <- safetensors::safetensors$new(path, framework = "torch")
    if (!"tokenizer_json" %in% handle$keys()) {
        stop("No tokenizer_json tensor in ", path)
    }
    bytes <- as.raw(as.integer(as.array(handle$get_tensor("tokenizer_json"))))
    txt <- rawToChar(bytes)
    Encoding(txt) <- "UTF-8"
    txt
}

# Special-token ids the TemplateProcessing "single" template puts before
# and after the sequence
.gemma4_template_ids <- function(post, added) {
    out <- list(prefix = integer(0), suffix = integer(0))
    single <- post$single
    if (!identical(post$type, "TemplateProcessing") || is.null(single)) {
        return(out)
    }
    seen_seq <- FALSE
    for (i in seq_len(NROW(single))) {
        sp <- single$SpecialToken$id[i] %||% NA
        if (!is.null(single$Sequence) && !is.na(single$Sequence$id[i] %||% NA)) {
            seen_seq <- TRUE
        } else if (!is.na(sp)) {
            id <- added$id[match(sp, added$content)]
            if (seen_seq) {
                out$suffix <- c(out$suffix, id)
            } else {
                out$prefix <- c(out$prefix, id)
            }
        }
    }
    out
}

# Normalize one plain-text segment, map characters to ids (byte fallback
# for pieces missing from the vocab), then apply the ranked merges.
# Shares the merge loop with the Qwen tokenizer.
.gemma4_encode_segment <- function(text, tokenizer) {
    if (!nzchar(text)) {
        return(integer(0))
    }
    text <- gsub(tokenizer$norm_from, tokenizer$norm_to, enc2utf8(text),
                 fixed = TRUE)
    chars <- strsplit(text, "", fixed = TRUE)[[1]]
    ids <- tokenizer$piece_ids[match(chars, tokenizer$piece_names)]
    if (anyNA(ids)) {
        parts <- as.list(ids)
        for (i in which(is.na(ids))) {
            if (tokenizer$byte_fallback) {
                b <- as.integer(charToRaw(chars[i]))
                parts[[i]] <- tokenizer$byte_ids[b + 1L]
            } else {
                parts[[i]] <- tokenizer$unk_id
            }
        }
        ids <- unlist(parts)
    }
    as.integer(.qwen_bpe_merge_ids(ids, tokenizer))
}

#' Encode prompts with the Gemma 4 tokenizer
#'
#' Tokenizes, applies the post-processor's special tokens (none for
#' Gemma 4), truncates to \code{max_length} keeping the start, and pads
#' on the left with \code{<pad>}, as the LTX pipelines do.
#'
#' @param tokenizer A \code{\link{gemma4_tokenizer}}.
#' @param texts Character vector of prompts.
#' @param max_length Integer. Fixed sequence length, or NULL for ragged
#'   unpadded output.
#'
#' @return List with \code{input_ids} and \code{attention_mask}: integer
#'   matrices [length(texts), max_length] (ragged lists when
#'   \code{max_length} is NULL). Ids are 0-based.
#'
#' @export
encode_gemma4 <- function(tokenizer, texts, max_length = 1024L) {
    stopifnot(inherits(tokenizer, "gemma4_tokenizer"))
    encode_one <- function(text) {
        ids <- integer(0)
        for (chunk in .qwen_split_added(text, tokenizer)) {
            if (!is.na(chunk$id)) {
                ids <- c(ids, as.integer(chunk$id))
            } else {
                ids <- c(ids, .gemma4_encode_segment(chunk$text, tokenizer))
            }
        }
        c(tokenizer$template$prefix, ids, tokenizer$template$suffix)
    }
    all_ids <- lapply(as.character(texts), encode_one)

    if (is.null(max_length)) {
        return(list(
                    input_ids = all_ids,
                    attention_mask = lapply(all_ids, function(x) rep(1L, length(x)))
            ))
    }
    n <- length(all_ids)
    input_ids <- matrix(as.integer(tokenizer$pad_id), nrow = n, ncol = max_length)
    attention_mask <- matrix(0L, nrow = n, ncol = max_length)
    for (i in seq_len(n)) {
        ids <- all_ids[[i]]
        if (length(ids) > max_length) {
            ids <- ids[seq_len(max_length)]
        }
        len <- length(ids)
        if (len > 0L) {
            cols <- (max_length - len + 1L):max_length
            input_ids[i, cols] <- ids
            attention_mask[i, cols] <- 1L
        }
    }
    list(input_ids = input_ids, attention_mask = attention_mask)
}
