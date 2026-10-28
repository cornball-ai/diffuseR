#' Encode a prompt for FLUX 3
#'
#' The Qwen3-VL prompt path of the reference pipeline (diffusers
#' Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds, Apache-2.0): the
#' prompt goes through the chat template (one user turn plus the
#' generation prompt, no thinking block), is right-padded to a multiple
#' of \code{pad_multiple} tokens (capped at \code{max_sequence_length}),
#' and the hidden states of \code{out_layers} are concatenated per
#' token. The padding tokens get position id 1, as in the training
#' environment, and the transformer attends to them, so their states are
#' part of the embedding.
#'
#' FLUX 3 expects a full scene caption: a short prompt gives generic
#' output. Expand short requests with an LLM first (BFL publishes the
#' system prompts with flux-3-image-klein-pup).
#'
#' @param prompt Character scalar.
#' @param model A \code{\link{qwen3_encoder}} loaded from FLUX 3's
#'   \code{text_encoder} with \code{\link{load_qwen3_text_encoder}}.
#' @param tokenizer A \code{\link{qwen_bpe_tokenizer}} from the same
#'   directory.
#' @param max_sequence_length Integer. Token cap (8192).
#' @param pad_multiple Integer. Window rounding (80).
#' @param out_layers Integer vector. Hidden-state layers to stack
#'   (4, 8, ..., 32).
#' @param device Character or NULL. Defaults to the model's device.
#'
#' @return \code{list(prompt_embeds, txt_ids)}: embeddings [1, L,
#'   length(out_layers) * hidden] and position ids [L, 4] from
#'   \code{\link{flux3_prepare_text_ids}}.
#'
#' @export
flux3_encode_prompt <- function(prompt, model, tokenizer,
                                max_sequence_length = 8192L,
                                pad_multiple = 80L,
                                out_layers = seq(4L, 32L, by = 4L),
                                device = NULL) {
    if (!is.character(prompt) || length(prompt) != 1L) {
        stop("`prompt` must be a single string.")
    }
    ids <- encode_qwen(tokenizer, prompt, max_length = NULL,
                       chat_template = TRUE, enable_thinking = TRUE)$input_ids[[1]]
    win <- .flux3_text_window(ids, tokenizer$pad_id, max_sequence_length,
                              pad_multiple)

    device <- device %||% model$model$embed_tokens$weight$device
    long <- torch::torch_long()
    input_ids <- torch::torch_tensor(matrix(win$ids + 1L, nrow = 1L),
                                     dtype = long, device = device)
    mask <- torch::torch_tensor(matrix(win$mask, nrow = 1L), dtype = long,
                                device = device)
    states <- torch::with_no_grad(model(input_ids, attention_mask = mask,
                                        out_layers = out_layers,
                                        position_ids = win$positions))
    list(prompt_embeds = torch::torch_cat(states, dim = -1L),
         txt_ids = flux3_prepare_text_ids(length(win$ids), device = device))
}

# Truncate to max_len, right-pad to a multiple of pad_multiple (capped at
# max_len). Padding positions are 1, matching the reference's
# arange(...).masked_fill(attention_mask == 0, 1).
.flux3_text_window <- function(ids, pad_id, max_len, pad_multiple) {
    ids <- utils::head(as.integer(ids), max_len)
    n <- length(ids)
    padded <- min(ceiling(n / pad_multiple) * pad_multiple, max_len)
    pad <- padded - n
    list(ids = c(ids, rep(as.integer(pad_id), pad)),
         mask = c(rep(1L, n), rep(0L, pad)),
         positions = c(seq_len(n) - 1L, rep(1L, pad)))
}
