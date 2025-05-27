
# Helper: get all adjacent symbol pairs
token_get_pairs <- function(symbols) {
  pairs <- character(0)
  if (length(symbols) > 1) {
    for (i in seq_len(length(symbols) - 1)) {
      pairs <- c(pairs, paste(symbols[i], symbols[i+1], sep = " "))
    }
  }
  unique(pairs)
}

# Helper: merge the first occurrence of a given bigram
token_merge_pair_once <- function(symbols, bigram) {
  parts <- strsplit(bigram, " ")[[1]]
  out <- c()
  i <- 1
  while (i <= length(symbols)) {
    if (i < length(symbols) && symbols[i] == parts[1] && symbols[i+1] == parts[2]) {
      out <- c(out, paste0(parts[1], parts[2]))
      i <- i + 2
    } else {
      out <- c(out, symbols[i])
      i <- i + 1
    }
  }
  out
}

#' Tokenize a prompt
#' 
#' @param prompt A character string prompt describing the image to generate.
#' @param merges Path to the merges file (BPE merges).
#' @param vocab_file Path to the vocabulary file (token->id mapping).
#' @param pad_token The token ID used for padding (default is 0).
#``
#' @return A 2D torch tensor of shape c(1, 77) containing the token IDs.
#' @export
CLIPTokenizer <- function(prompt,
                          merges = system.file("tokenizer/merges.txt",
                                               package = "diffuseR"),
                          vocab_file = system.file("tokenizer/vocab.json",
                                                   package = "diffuseR"),
                          pad_token = 0L) {
  # 1. Load merges and build BPE rank map
  merge_lines <- readLines(merges, encoding = "UTF-8")
  merge_lines <- merge_lines[-1]                    # drop header
  merge_lines <- merge_lines[nzchar(merge_lines)]  # drop blanks
  merges_list <- strsplit(merge_lines, " ")
  merge_keys <- sapply(merges_list, paste, collapse = " ")
  bpe_ranks <- stats::setNames(seq_along(merge_keys), merge_keys)
  
  # 2. Load vocabulary JSON (token->id)
  vocab <- jsonlite::fromJSON(vocab_file)
  
  # 3. Prepare text: lowercase and split into words
  text <- tolower(prompt)
  words <- strsplit(text, "\\s+")[[1]]
  
  # 4. BPE tokenization per word
  tokens <- character(0)
  for (word in words) {
    chars <- strsplit(word, "")[[1]]
    symbols <- if (length(chars) > 0) {
      c(
        if (length(chars) > 1) chars[-length(chars)] else character(0),
        paste0(chars[length(chars)], "</w>")
      )
    } else {
      character(0)
    }
    repeat {
      pairs <- token_get_pairs(symbols)
      valid <- intersect(pairs, names(bpe_ranks))
      if (length(valid) == 0) break
      best <- valid[which.min(bpe_ranks[valid])]
      symbols <- token_merge_pair_once(symbols, best)
    }
    tokens <- c(tokens, symbols)
  }
  
  # Truncate tokens to max length
  max_tokens <- 77L - 2L  # 2 for start and end tokens
  if (length(tokens) > max_tokens) {
    truncated_tokens <- tokens[(max_tokens + 1):length(tokens)]
    tokens <- tokens[1:max_tokens]
    warning("Prompt was truncated. Consider shortening it.")
    warning("Dropped prompt: ", paste(truncated_tokens, collapse = " "))
  }
  
  # 5. Map token strings to vocabulary IDs
  ids <- vapply(tokens, function(tok) {
    if (!tok %in% names(vocab)) stop(paste("Unknown token:", tok))
    vocab[[tok]]
  }, integer(1))
  
  # 6. Add special tokens and pad/truncate
  sot_id <- 49406L
  eot_id <- 49407L
  seq_ids <- c(sot_id, ids, eot_id)
  max_len <- 77L
  if (length(seq_ids) < max_len) {
    seq_ids <- c(seq_ids, rep(pad_token, max_len - length(seq_ids)))
  } else if (length(seq_ids) > max_len) {
    seq_ids <- seq_ids[1:max_len]
  }
  
  # 7. Return as 2D torch tensor [1, 77]
  t <- torch::torch_tensor(seq_ids, dtype = torch::torch_long())
  t <- t$unsqueeze(1)  # prepend batch dimension
  t
}
