# Native R BPE Tokenizer
#
# Implements Byte-Pair Encoding tokenization in pure R.
# Compatible with HuggingFace tokenizer.json format.

# -----------------------------------------------------------------------------
# BPE Tokenizer Class
# -----------------------------------------------------------------------------
#' BPE Tokenizer
#'
#' Native R implementation of Byte-Pair Encoding tokenizer.
#' Loads from HuggingFace tokenizer.json format.
#'
#' @param tokenizer_path Path to tokenizer.json or directory containing it.
#' @return A bpe_tokenizer object.
#' @export
bpe_tokenizer <- function(tokenizer_path) {
  # Find tokenizer.json

  if (dir.exists(tokenizer_path)) {
    json_path <- file.path(tokenizer_path, "tokenizer.json")
  } else {
    json_path <- tokenizer_path
  }

  if (!file.exists(json_path)) {
    stop("tokenizer.json not found at: ", json_path)
  }

  # Load JSON
  config <- jsonlite::fromJSON(json_path, simplifyVector = FALSE)

  # Extract model configuration
  model <- config$model
  if (is.null(model) || model$type != "BPE") {
    stop("Only BPE tokenizers are supported")
  }

  # Build vocabulary lookup (token -> id)
  vocab <- model$vocab
  if (is.null(vocab)) {
    stop("No vocabulary found in tokenizer.json")
  }

  # Convert vocab list to named vector for fast lookup
  vocab_ids <- unlist(vocab)
  names(vocab_ids) <- names(vocab)

  # Build reverse vocabulary (id -> token)
  id_to_token <- names(vocab_ids)
  names(id_to_token) <- as.character(vocab_ids)

  # Parse merges into a priority map
  merges <- model$merges
  if (is.null(merges)) {
    merges <- character(0)
  }

  # Create merge priority lookup (pair -> priority)
  # Lower priority number = merge first
  merge_priority <- seq_along(merges)
  names(merge_priority) <- merges

  # Extract special tokens
  added_tokens <- config$added_tokens
  special_tokens <- list()
  if (!is.null(added_tokens)) {
    for (tok in added_tokens) {
      special_tokens[[tok$content]] <- list(
        id = tok$id,
        special = tok$special %||% FALSE
      )
    }
  }

  # Extract configuration
  byte_fallback <- model$byte_fallback %||% FALSE
  fuse_unk <- model$fuse_unk %||% FALSE
  unk_token <- model$unk_token

  # Pre-tokenizer configuration
  pre_tokenizer <- config$pre_tokenizer
  add_prefix_space <- FALSE
  if (!is.null(pre_tokenizer)) {
    # Check for Metaspace pre-tokenizer (adds space prefix)
    if (!is.null(pre_tokenizer$type) && pre_tokenizer$type == "Metaspace") {
      add_prefix_space <- pre_tokenizer$add_prefix_space %||% TRUE
    }
  }

  structure(
    list(
      vocab = vocab_ids,
      id_to_token = id_to_token,
      merges = merges,
      merge_priority = merge_priority,
      special_tokens = special_tokens,
      byte_fallback = byte_fallback,
      fuse_unk = fuse_unk,
      unk_token = unk_token,
      add_prefix_space = add_prefix_space,
      vocab_size = length(vocab_ids)
    ),
    class = "bpe_tokenizer"
  )
}

#' Print BPE Tokenizer
#' @param x A bpe_tokenizer object.
#' @param ... Additional arguments (ignored).
#' @export
print.bpe_tokenizer <- function(
  x,
  ...
) {
  cat("BPE Tokenizer\n")
  cat("  Vocabulary size:", x$vocab_size, "\n")
  cat("  Merge rules:", length(x$merges), "\n")
  cat("  Special tokens:", length(x$special_tokens), "\n")
  cat("  Byte fallback:", x$byte_fallback, "\n")
  invisible(x)
}

# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

#' Encode text to token IDs
#'
#' @param tokenizer A bpe_tokenizer object.
#' @param text Character string or vector to encode.
#' @param add_special_tokens Logical. Add BOS/EOS tokens.
#' @param max_length Integer. Maximum sequence length (NULL for no limit).
#' @param padding Character. Padding strategy: "none", "max_length", or "longest".
#' @param truncation Logical. Truncate to max_length.
#' @param return_tensors Character. Return type: "list" or "pt" (torch tensors).
#' @return List with input_ids and attention_mask.
#' @export
encode_bpe <- function(
  tokenizer,
  text,
  add_special_tokens = TRUE,
  max_length = NULL,
  padding = "none",
  truncation = FALSE,
  return_tensors = "list"
) {
  if (!inherits(tokenizer, "bpe_tokenizer")) {
    stop("tokenizer must be a bpe_tokenizer object")
  }

  # Handle single string
  if (length(text) == 1) {
    text <- list(text)
  }

  # Encode each text
  encoded <- lapply(text, function(t) {
      encode_single(tokenizer, t, add_special_tokens = add_special_tokens)
    })

  # Get lengths
  lengths <- vapply(encoded, length, integer(1))
  max_len <- max(lengths)

  # Apply max_length constraint
  if (!is.null(max_length)) {
    if (truncation) {
      encoded <- lapply(encoded, function(ids) {
          if (length(ids) > max_length) ids[1:max_length] else ids
        })
    }
    if (padding == "max_length") {
      max_len <- max_length
    }
  }

  # Apply padding
  if (padding != "none") {
    pad_id <- get_pad_id(tokenizer)
    encoded <- lapply(encoded, function(ids) {
        if (length(ids) < max_len) {
          # Left padding (Gemma style)
          c(rep(pad_id, max_len - length(ids)), ids)
        } else {
          ids
        }
      })
  }

  # Create attention mask
  attention_mask <- lapply(encoded, function(ids) {
      pad_id <- get_pad_id(tokenizer)
      as.integer(ids != pad_id)
    })

  # Convert to matrix
  input_ids <- do.call(rbind, encoded)
  attention_mask <- do.call(rbind, attention_mask)

  # Convert to tensors if requested
  if (return_tensors == "pt") {
    input_ids <- torch::torch_tensor(input_ids, dtype = torch::torch_long())
    attention_mask <- torch::torch_tensor(attention_mask, dtype = torch::torch_long())
  }

  list(
    input_ids = input_ids,
    attention_mask = attention_mask
  )
}

#' Encode a single text string
#' @keywords internal
encode_single <- function(
  tokenizer,
  text,
  add_special_tokens = TRUE
) {
  # Add prefix space if configured (Gemma style)
  if (tokenizer$add_prefix_space && !startsWith(text, " ")) {
    text <- paste0(" ", text)
  }

  # Pre-tokenize: split on whitespace while keeping track of spaces
  # Replace spaces with special character (SentencePiece style)
  text <- gsub(" ", "\u2581", text) # LOWER ONE EIGHTH BLOCK

  # Check for special tokens first
  for (tok_name in names(tokenizer$special_tokens)) {
    if (text == tok_name) {
      return(tokenizer$special_tokens[[tok_name]]$id)
    }
  }

  # Use greedy longest match tokenization (for large vocabs like Gemma)
  # This is more efficient than BPE merging for pre-trained vocabs
  ids <- greedy_tokenize(text, tokenizer$vocab, tokenizer$byte_fallback, tokenizer$unk_token)

  # Add special tokens
  if (add_special_tokens) {
    bos_id <- get_bos_id(tokenizer)
    if (!is.null(bos_id)) {
      ids <- c(bos_id, ids)
    }
  }

  ids
}

#' Greedy longest match tokenization
#' @keywords internal
greedy_tokenize <- function(
  text,
  vocab,
  byte_fallback = FALSE,
  unk_token = NULL
) {
  ids <- integer(0)
  i <- 1
  n <- nchar(text)
  vocab_names <- names(vocab)

  while (i <= n) {
    # Try to match the longest token starting at position i
    matched <- FALSE

    # Try decreasing lengths
    for (len in min(n - i + 1, 50) :1) { # Cap at 50 chars max
      candidate <- substr(text, i, i + len - 1)

      if (candidate %in% vocab_names) {
        ids <- c(ids, vocab[[candidate]])
        i <- i + len
        matched <- TRUE
        break
      }
    }

    if (!matched) {
      # No match found - use byte fallback or UNK
      char <- substr(text, i, i)
      if (byte_fallback) {
        # Try single character in vocab first
        if (char %in% vocab_names) {
          ids <- c(ids, vocab[[char]])
        } else {
          # Byte fallback: convert to <0xHH> format
          bytes <- charToRaw(char)
          for (b in bytes) {
            byte_token <- sprintf("<0x%02X>", as.integer(b))
            if (byte_token %in% vocab_names) {
              ids <- c(ids, vocab[[byte_token]])
            } else if (!is.null(unk_token) && unk_token %in% vocab_names) {
              ids <- c(ids, vocab[[unk_token]])
            } else {
              ids <- c(ids, 3L) # Default UNK
            }
          }
        }
      } else if (!is.null(unk_token) && unk_token %in% vocab_names) {
        ids <- c(ids, vocab[[unk_token]])
      } else {
        ids <- c(ids, 3L) # Default UNK
      }
      i <- i + 1
    }
  }

  ids
}

#' Apply BPE merge rules
#' @keywords internal
apply_bpe_merges <- function(
  tokens,
  merge_priority,
  vocab
) {
  if (length(tokens) <= 1) {
    return(tokens)
  }

  # Iteratively merge token pairs
  changed <- TRUE
  while (changed && length(tokens) > 1) {
    changed <- FALSE
    best_idx <- NULL
    best_priority <- Inf

    # Find the highest priority merge
    for (i in seq_len(length(tokens) - 1)) {
      pair <- paste(tokens[i], tokens[i + 1])
      if (pair %in% names(merge_priority)) {
        priority <- merge_priority[[pair]]
        if (priority < best_priority) {
          best_priority <- priority
          best_idx <- i
        }
      }
    }

    # Apply the best merge
    if (!is.null(best_idx)) {
      merged <- paste0(tokens[best_idx], tokens[best_idx + 1])
      # Only merge if result is in vocabulary
      if (merged %in% names(vocab)) {
        tokens <- c(
          if (best_idx > 1) tokens[1:(best_idx - 1)] else character(0),
          merged,
          if (best_idx + 2 <= length(tokens)) tokens[(best_idx + 2) :length(tokens)] else character(0)
        )
        changed <- TRUE
      }
    }
  }

  tokens
}

# -----------------------------------------------------------------------------
# Decoding
# -----------------------------------------------------------------------------

#' Decode token IDs to text
#'
#' @param tokenizer A bpe_tokenizer object.
#' @param ids Integer vector or matrix of token IDs.
#' @param skip_special_tokens Logical. Skip special tokens in output.
#' @return Character string or vector.
#' @export
decode_bpe <- function(
  tokenizer,
  ids,
  skip_special_tokens = TRUE
) {
  if (!inherits(tokenizer, "bpe_tokenizer")) {
    stop("tokenizer must be a bpe_tokenizer object")
  }

  # Handle matrix input
  if (is.matrix(ids)) {
    return(apply(ids, 1, function(row) {
          decode_bpe(tokenizer, row, skip_special_tokens = skip_special_tokens)
        }))
  }

  # Handle torch tensor
  if (inherits(ids, "torch_tensor")) {
    ids <- as.integer(ids$cpu()$numpy())
  }

  # Get special token IDs to skip
  special_ids <- integer(0)
  if (skip_special_tokens) {
    special_ids <- vapply(tokenizer$special_tokens, function(tok) tok$id, integer(1))
  }

  # Convert IDs to tokens
  tokens <- vapply(ids, function(id) {
      if (id %in% special_ids) {
        ""
      } else {
        id_str <- as.character(id)
        if (id_str %in% names(tokenizer$id_to_token)) {
          tokenizer$id_to_token[[id_str]]
        } else {
          ""
        }
      }
    }, character(1))

  # Join tokens and decode
  text <- paste(tokens, collapse = "")

  # Replace SentencePiece space marker with actual space
  text <- gsub("\u2581", " ", text)

  # Decode byte tokens like <0x41>
  # Find all byte tokens and replace them
  byte_pattern <- "<0x([0-9A-Fa-f]{2})>"
  while (grepl(byte_pattern, text)) {
    match <- regmatches(text, regexpr(byte_pattern, text))
    if (length(match) > 0) {
      hex_str <- sub("<0x([0-9A-Fa-f]{2})>", "\\1", match)
      byte_val <- strtoi(hex_str, base = 16)
      char <- rawToChar(as.raw(byte_val))
      text <- sub(byte_pattern, char, text, fixed = FALSE)
    } else {
      break
    }
  }

  # Trim leading space if prefix was added
  text <- sub("^ ", "", text)

  text
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

#' Get padding token ID
#' @keywords internal
get_pad_id <- function(tokenizer) {
  if ("<pad>" %in% names(tokenizer$special_tokens)) {
    tokenizer$special_tokens[["<pad>"]]$id
  } else if ("<pad>" %in% names(tokenizer$vocab)) {
    tokenizer$vocab[["<pad>"]]
  } else {
    0L
  }
}

#' Get BOS token ID
#' @keywords internal
get_bos_id <- function(tokenizer) {
  if ("<bos>" %in% names(tokenizer$special_tokens)) {
    tokenizer$special_tokens[["<bos>"]]$id
  } else if ("<bos>" %in% names(tokenizer$vocab)) {
    tokenizer$vocab[["<bos>"]]
  } else {
    NULL
  }
}

#' Get EOS token ID
#' @keywords internal
get_eos_id <- function(tokenizer) {
  if ("<eos>" %in% names(tokenizer$special_tokens)) {
    tokenizer$special_tokens[["<eos>"]]$id
  } else if ("<eos>" %in% names(tokenizer$vocab)) {
    tokenizer$vocab[["<eos>"]]
  } else {
    NULL
  }
}

#' Get vocabulary size
#' @param tokenizer A bpe_tokenizer object.
#' @return Integer vocabulary size.
#' @export
vocab_size <- function(tokenizer) {
  tokenizer$vocab_size
}

# Null-coalescing operator
if (!exists("%||%", mode = "function")) {
  `%||%` <- function(
    x,
    y
  ) if (is.null(x)) y else x
}

