#' SentencePiece Unigram Tokenizer
#'
#' Pure R implementation of HuggingFace tokenizer.json files with a
#' Unigram model (SentencePiece), as used by T5 - FLUX's second text
#' encoder. Segmentation is Viterbi best-path over the vocab log
#' probabilities (Kudo 2018, arXiv:1804.10959). The normalizer and
#' Metaspace pre-tokenizer settings are read from the file.
#'
#' Limitation: the Precompiled charsmap normalizer (NFKC-style unicode
#' mapping) is approximated by control-whitespace substitution only;
#' ASCII and common latin text tokenizes identically to the reference,
#' exotic unicode may differ.
#'
#' @name tokenizer_unigram
NULL

#' Load a Unigram tokenizer from tokenizer.json
#'
#' @param tokenizer_path Path to a HuggingFace tokenizer.json with a
#'   Unigram model, or a directory containing one.
#'
#' @return A \code{unigram_tokenizer} object.
#'
#' @export
unigram_tokenizer <- function(tokenizer_path) {
    path <- path.expand(tokenizer_path)
    if (dir.exists(path)) {
        path <- file.path(path, "tokenizer.json")
    }
    if (!file.exists(path)) {
        stop("tokenizer.json not found: ", path)
    }

    tj <- jsonlite::fromJSON(path, simplifyVector = FALSE)
    model <- tj$model
    if (is.null(model) || !identical(model$type, "Unigram")) {
        stop("Only Unigram tokenizers are supported (got ",
             model$type %||% "none", "); use bpe_tokenizer() for BPE.")
    }

    pieces <- vapply(model$vocab, function(p) p[[1]], character(1))
    scores <- vapply(model$vocab, function(p) as.numeric(p[[2]]), numeric(1))
    ids0 <- seq_along(pieces) - 1L

    vocab_env <- new.env(parent = emptyenv(), size = length(pieces))
    for (i in seq_along(pieces)) {
        assign(pieces[i], c(ids0[i], scores[i]), envir = vocab_env)
    }

    # Normalizer settings (Sequence or single normalizer)
    norms <- tj$normalizer
    if (!is.null(norms) && identical(norms$type, "Sequence")) {
        norms <- norms$normalizers
    } else if (!is.null(norms)) {
        norms <- list(norms)
    } else {
        norms <- list()
    }
    strip_right <- FALSE
    space_collapse <- NULL
    for (n in norms) {
        if (identical(n$type, "Strip") && isTRUE(n$strip_right)) {
            strip_right <- TRUE
        }
        if (identical(n$type, "Replace") &&
            identical(n$pattern$Regex, " {2,}")) {
            space_collapse <- n$content
        }
    }

    # Metaspace pre-tokenizer (possibly inside a Sequence)
    pre <- tj$pre_tokenizer
    pres <- if (!is.null(pre) && identical(pre$type, "Sequence")) {
        pre$pretokenizers
    } else if (!is.null(pre)) {
        list(pre)
    } else {
        list()
    }
    prepend_scheme <- "never"
    replacement <- "▁"
    for (p in pres) {
        if (identical(p$type, "Metaspace")) {
            replacement <- p$replacement %||% "▁"
            prepend_scheme <- p$prepend_scheme %||%
            (if (isTRUE(p$add_prefix_space)) "always" else "never")
        }
    }

    unk_id <- as.integer(model$unk_id %||% 2L)
    eos <- get0("</s>", envir = vocab_env)
    pad <- get0("<pad>", envir = vocab_env)

    structure(
              list(
                   vocab = vocab_env,
                   n_pieces = length(pieces),
                   max_piece_chars = max(nchar(pieces)),
                   unk_id = unk_id,
                   unk_score = min(scores) - 10.0,
                   eos_id = if (!is.null(eos)) as.integer(eos[1]) else 1L,
                   pad_id = if (!is.null(pad)) as.integer(pad[1]) else 0L,
                   strip_right = strip_right,
                   space_collapse = space_collapse,
                   prepend_scheme = prepend_scheme,
                   replacement = replacement,
                   path = path
        ),
              class = "unigram_tokenizer"
    )
}

#' @export
print.unigram_tokenizer <- function(x, ...) {
    cat("<unigram_tokenizer>\n")
    cat("  pieces:  ", x$n_pieces, "\n")
    cat("  unk/eos/pad:", x$unk_id, x$eos_id, x$pad_id, "\n")
    cat("  path:    ", x$path, "\n")
    invisible(x)
}

# Viterbi best-path segmentation of one pre-token (0-based ids)
.unigram_viterbi <- function(word, tokenizer) {
    n <- nchar(word)
    if (n == 0L) {
        return(integer(0))
    }
    vocab <- tokenizer$vocab
    max_len <- tokenizer$max_piece_chars

    best <- c(0, rep(-Inf, n))
    back_len <- integer(n)
    back_id <- integer(n)

    for (end in seq_len(n)) {
        for (len in seq_len(min(max_len, end))) {
            start <- end - len + 1L
            piece <- substr(word, start, end)
            entry <- get0(piece, envir = vocab)
            if (is.null(entry)) {
                if (len > 1L) {
                    next
                }
                id <- tokenizer$unk_id
                score <- tokenizer$unk_score
            } else {
                id <- entry[1]
                score <- entry[2]
            }
            cand <- best[start] + score
            if (cand > best[end + 1L]) {
                best[end + 1L] <- cand
                back_len[end] <- len
                back_id[end] <- id
            }
        }
    }

    ids <- integer(0)
    pos <- n
    while (pos > 0L) {
        ids <- c(back_id[pos], ids)
        pos <- pos - back_len[pos]
    }
    ids
}

#' Encode text with a Unigram tokenizer
#'
#' Normalizes (strip-right, multi-space collapse, control whitespace to
#' space), applies the Metaspace pre-tokenizer, segments each pre-token
#' by Viterbi over the Unigram scores, fuses consecutive unknowns, and
#' appends EOS. T5 semantics: right padding with \code{<pad>} (id 0),
#' truncation to \code{max_length - 1} before the EOS.
#'
#' @param tokenizer A \code{\link{unigram_tokenizer}}.
#' @param texts Character vector of prompts.
#' @param max_length Integer. Fixed sequence length (NULL for no
#'   truncation/padding).
#' @param add_eos Logical. Append the EOS token.
#' @param pad Logical. Right-pad to \code{max_length}.
#'
#' @return List with \code{input_ids} and \code{attention_mask}, each an
#'   integer matrix [length(texts), max_length] (or ragged lists when
#'   \code{max_length} is NULL). Ids are 0-based (HuggingFace
#'   convention); add 1 for R torch embedding lookups.
#'
#' @export
encode_unigram <- function(tokenizer, texts, max_length = 256L,
                           add_eos = TRUE, pad = TRUE) {
    stopifnot(inherits(tokenizer, "unigram_tokenizer"))
    rep_char <- tokenizer$replacement

    encode_one <- function(text) {
        # Control whitespace to space (charsmap approximation), then the
        # file's normalizer chain
        text <- enc2utf8(text)
        text <- gsub("[\t\n\r\f\v]", " ", text)
        if (tokenizer$strip_right) {
            text <- sub("[ ]+$", "", text)
        }
        if (!is.null(tokenizer$space_collapse)) {
            text <- gsub(" {2,}", tokenizer$space_collapse, text)
        }
        # Metaspace: spaces to the replacement, optional prefix
        text <- gsub(" ", rep_char, text, fixed = TRUE)
        if (tokenizer$prepend_scheme == "always" &&
            !startsWith(text, rep_char)) {
            text <- paste0(rep_char, text)
        }
        if (nchar(text) == 0L) {
            return(integer(0))
        }
        # Split before each replacement char, keeping it attached.
        # (strsplit with a zero-width lookahead detaches the char, so
        # mark boundaries with a control byte instead.)
        marked <- gsub(rep_char, paste0("\u0001", rep_char), text,
                       fixed = TRUE)
        words <- strsplit(marked, "\u0001", fixed = TRUE)[[1]]
        words <- words[nzchar(words)]

        ids <- unlist(lapply(words, .unigram_viterbi, tokenizer = tokenizer))
        ids <- as.integer(ids %||% integer(0))

        # Fuse consecutive unknowns (HF fuse_unk)
        if (length(ids) > 1L) {
            is_unk <- ids == tokenizer$unk_id
            drop <- is_unk & c(FALSE, is_unk[-length(is_unk)])
            ids <- ids[!drop]
        }
        ids
    }

    all_ids <- lapply(as.character(texts), encode_one)

    if (add_eos) {
        keep <- if (is.null(max_length)) Inf else max_length - 1L
        all_ids <- lapply(all_ids, function(ids) {
            if (length(ids) > keep) {
                ids <- ids[seq_len(keep)]
            }
            c(ids, tokenizer$eos_id)
        })
    } else if (!is.null(max_length)) {
        all_ids <- lapply(all_ids, function(ids) {
            if (length(ids) > max_length) ids[seq_len(max_length)] else ids
        })
    }

    if (is.null(max_length) || !pad) {
        masks <- lapply(all_ids, function(ids) rep(1L, length(ids)))
        return(list(input_ids = all_ids, attention_mask = masks))
    }

    n <- length(all_ids)
    input_ids <- matrix(tokenizer$pad_id, nrow = n, ncol = max_length)
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
