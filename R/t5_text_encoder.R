#' T5 Text Encoder (T5-v1.1)
#'
#' Fresh R port of the T5 encoder stack from HuggingFace transformers
#' (Apache-2.0, src/transformers/models/t5/modeling_t5.py), as used by
#' FLUX's second text encoder (T5-v1.1-XXL: 24 layers, d_model 4096,
#' 64 heads x d_kv 64, gated-GELU FFN). Distinctives faithfully carried
#' over: RMS layer norms (no mean subtraction), no biases anywhere, no
#' 1/sqrt(d) attention scaling (folded into the weights), and a shared
#' relative position bias computed once from block 1's embedding and
#' added to every layer's attention logits. Module field names mirror
#' the checkpoint keys (minus the \code{encoder.} prefix).
#'
#' FLUX passes no attention mask - padding tokens attend and are
#' attended to - so none is implemented.
#'
#' @name t5_text_encoder
NULL

# Bucketed relative positions (bidirectional): half the buckets split by
# sign, half of those exact small offsets, the rest log-spaced up to
# max_distance. Reference: T5Attention._relative_position_bucket.
.t5_relative_position_bucket <- function(relative_position,
    num_buckets = 32L,
    max_distance = 128L) {
    num_buckets <- num_buckets %/% 2L
    long <- torch::torch_long()
    relative_buckets <- (relative_position > 0)$to(dtype = long)$mul(num_buckets)
    relative_position <- torch::torch_abs(relative_position)

    max_exact <- num_buckets %/% 2L
    is_small <- relative_position < max_exact

    rp_large <- relative_position$to(dtype = torch::torch_float32())$
    div(max_exact)$log()$
    div(log(max_distance / max_exact))$
    mul(num_buckets - max_exact)$
    to(dtype = long)$add(max_exact)
    rp_large <- torch::torch_minimum(
                                     rp_large,
                                     torch::torch_full_like(rp_large, num_buckets - 1L)
    )

    relative_buckets + torch::torch_where(is_small, relative_position, rp_large)
}

# T5 self-attention: no scaling, no biases; the relative position bias
# is added to the logits pre-softmax (softmax in float32)
.t5_attention <- torch::nn_module(
                                  "t5_attention",
                                  initialize = function(d_model, d_kv, num_heads, has_relative_bias = FALSE,
        num_buckets = 32L, max_distance = 128L) {
    inner_dim <- num_heads * d_kv
    self$num_heads <- num_heads
    self$d_kv <- d_kv
    self$num_buckets <- as.integer(num_buckets)
    self$max_distance <- as.integer(max_distance)
    self$q <- torch::nn_linear(d_model, inner_dim, bias = FALSE)
    self$k <- torch::nn_linear(d_model, inner_dim, bias = FALSE)
    self$v <- torch::nn_linear(d_model, inner_dim, bias = FALSE)
    self$o <- torch::nn_linear(inner_dim, d_model, bias = FALSE)
    if (has_relative_bias) {
        self$relative_attention_bias <- torch::nn_embedding(num_buckets,
            num_heads)
    }
},
                                  compute_bias = function(seq_len, device) {
    pos <- torch::torch_arange(start = 0, end = seq_len - 1,
                               dtype = torch::torch_long(), device = device)
    # relative_position[i, j] = j - i
    relative_position <- pos$unsqueeze(1L) - pos$unsqueeze(2L)
    buckets <- .t5_relative_position_bucket(relative_position,
        num_buckets = self$num_buckets,
        max_distance = self$max_distance)
    values <- self$relative_attention_bias(buckets + 1L) # [S, S, H]
    values$permute(c(3L, 1L, 2L))$unsqueeze(1L) # [1, H, S, S]
},
                                  forward = function(x, position_bias) {
    shape <- x$shape
    b <- shape[1]
    s <- shape[2]
    per_head <- c(b, s, self$num_heads, self$d_kv)
    q <- self$q(x)$view(per_head)$transpose(2L, 3L) # [B, H, S, dk]
    k <- self$k(x)$view(per_head)$transpose(2L, 3L)
    v <- self$v(x)$view(per_head)$transpose(2L, 3L)

    scores <- torch::torch_matmul(q, k$transpose(-2L, -1L)) # no 1/sqrt(d)
    scores <- scores + position_bias
    attn <- torch::nnf_softmax(scores$to(dtype = torch::torch_float32()),
                               dim = -1L)$to(dtype = scores$dtype)
    out <- torch::torch_matmul(attn, v)
    out <- out$transpose(2L, 3L)$reshape(c(b, s, -1L))
    self$o(out)
}
)

# layer.0: pre-norm self-attention with residual
.t5_self_attn_layer <- torch::nn_module(
                                        "t5_self_attn_layer",
                                        initialize = function(d_model, d_kv, num_heads, eps,
        has_relative_bias = FALSE, num_buckets = 32L,
        max_distance = 128L) {
    self$SelfAttention <- .t5_attention(d_model, d_kv, num_heads,
                                        has_relative_bias = has_relative_bias,
                                        num_buckets = num_buckets,
                                        max_distance = max_distance)
    self$layer_norm <- ltx23_rms_norm(d_model, eps = eps)
},
                                        forward = function(x, position_bias) {
    x + self$SelfAttention(self$layer_norm(x), position_bias)
}
)

# layer.1: pre-norm gated-GELU feed-forward with residual
.t5_ff_layer <- torch::nn_module(
                                 "t5_ff_layer",
                                 initialize = function(d_model, d_ff, eps) {
    dense <- torch::nn_module(
                              "t5_dense_gated_act_dense",
                              initialize = function(d_model, d_ff) {
        self$wi_0 <- torch::nn_linear(d_model, d_ff, bias = FALSE)
        self$wi_1 <- torch::nn_linear(d_model, d_ff, bias = FALSE)
        self$wo <- torch::nn_linear(d_ff, d_model, bias = FALSE)
    },
                              forward = function(x) {
        h <- torch::nnf_gelu(self$wi_0(x), approximate = "tanh") * self$wi_1(x)
        self$wo(h)
    }
    )
    self$DenseReluDense <- dense(d_model, d_ff)
    self$layer_norm <- ltx23_rms_norm(d_model, eps = eps)
},
                                 forward = function(x) {
    x + self$DenseReluDense(self$layer_norm(x))
}
)

.t5_block <- torch::nn_module(
                              "t5_block",
                              initialize = function(d_model, d_kv, num_heads, d_ff, eps,
        has_relative_bias = FALSE, num_buckets = 32L,
        max_distance = 128L) {
    self$layer <- torch::nn_module_list(list(
            .t5_self_attn_layer(d_model, d_kv, num_heads, eps,
                                has_relative_bias = has_relative_bias,
                                num_buckets = num_buckets,
                                max_distance = max_distance),
            .t5_ff_layer(d_model, d_ff, eps)
        ))
},
                              forward = function(x, position_bias) {
    x <- self$layer[[1]](x, position_bias)
    self$layer[[2]](x)
}
)

#' T5 encoder stack
#'
#' Defaults are the T5-v1.1-XXL configuration used by FLUX.
#'
#' @param vocab_size,d_model,d_kv,num_heads,d_ff,num_layers Integers.
#' @param relative_attention_num_buckets,relative_attention_max_distance
#'   Integers. Relative position bias shape.
#' @param layer_norm_epsilon Numeric.
#'
#' @return Module whose forward(input_ids) (1-based ids [B, S]) returns
#'   the last hidden state [B, S, d_model].
#'
#' @export
t5_encoder <- torch::nn_module(
                               "t5_encoder",
                               initialize = function(vocab_size = 32128L, d_model = 4096L, d_kv = 64L,
        num_heads = 64L, d_ff = 10240L, num_layers = 24L,
        relative_attention_num_buckets = 32L,
        relative_attention_max_distance = 128L,
        layer_norm_epsilon = 1e-6) {
    self$shared <- torch::nn_embedding(vocab_size, d_model)
    self$block <- torch::nn_module_list(
                                        lapply(seq_len(num_layers), function(i) {
        .t5_block(d_model, d_kv, num_heads, d_ff, layer_norm_epsilon,
                  has_relative_bias = (i == 1L),
                  num_buckets = relative_attention_num_buckets,
                  max_distance = relative_attention_max_distance)
    })
    )
    self$final_layer_norm <- ltx23_rms_norm(d_model, eps = layer_norm_epsilon)
},
                               forward = function(input_ids) {
    x <- self$shared(input_ids)
    # Bias comes from block 1 and is shared by every layer
    position_bias <- self$block[[1]]$layer[[1]]$SelfAttention$compute_bias(
        input_ids$shape[2], input_ids$device
    )$to(dtype = x$dtype)
    for (i in seq_along(self$block)) {
        x <- self$block[[i]](x, position_bias)
    }
    self$final_layer_norm(x)
}
)

# Encoder constructor arguments from a transformers config.json
.t5_encoder_args <- function(config) {
    if (is.null(config)) {
        return(list())
    }
    ffp <- config$feed_forward_proj %||% "gated-gelu"
    if (!startsWith(ffp, "gated")) {
        stop("Only gated feed-forward T5 (v1.1) is supported, got: ", ffp)
    }
    args <- list(
                 vocab_size = config$vocab_size,
                 d_model = config$d_model,
                 d_kv = config$d_kv,
                 num_heads = config$num_heads,
                 d_ff = config$d_ff,
                 num_layers = config$num_layers,
                 relative_attention_num_buckets = config$relative_attention_num_buckets,
                 relative_attention_max_distance = config$relative_attention_max_distance
    )
    args <- Filter(function(x) !is.null(x) && length(x) > 0L, args)
    args <- lapply(args, as.integer)
    eps <- config$layer_norm_epsilon
    if (!is.null(eps) && length(eps) == 1L) {
        args$layer_norm_epsilon <- as.numeric(eps)
    }
    args
}

#' Load a T5 encoder from a transformers directory
#'
#' Streams the (possibly sharded) safetensors weights into
#' \code{\link{t5_encoder}}, stripping the \code{encoder.} key prefix
#' and aliasing \code{embed_tokens} to the shared embedding.
#'
#' @param model_path Directory with \code{config.json} and
#'   \code{model*.safetensors} (FLUX.1-schnell's \code{text_encoder_2}).
#' @param device Character. Target device.
#' @param dtype Character. "float32" (CPU default; T5 overflows in
#'   float16) or "bfloat16".
#' @param verbose Logical.
#' @param ... Overrides for \code{\link{t5_encoder}} arguments.
#'
#' @return The loaded \code{t5_encoder} in eval mode.
#'
#' @export
load_t5_text_encoder <- function(model_path, device = "cpu",
                                 dtype = "float32", verbose = TRUE, ...) {
    model_path <- path.expand(model_path)
    config <- NULL
    config_path <- file.path(model_path, "config.json")
    if (file.exists(config_path)) {
        config <- jsonlite::fromJSON(config_path, simplifyVector = TRUE)
    }

    args <- utils::modifyList(.t5_encoder_args(config), list(...))
    model <- do.call(t5_encoder, args)
    model$to(dtype = .flux_dtype(dtype))

    opened <- .flux_open_sharded_dir(model_path, "model")
    ckpt <- structure(
                      list(handle = opened$handle, keys = opened$keys, version = NULL,
                           config = config, path = model_path),
                      class = "ltx23_checkpoint"
    )

    map_key <- function(key) {
        key <- sub("^encoder\\.", "", key)
        if (key == "embed_tokens.weight") {
            key <- "shared.weight"
        }
        key
    }
    res <- ltx23_load_group(ckpt, ckpt$keys, model, map_key = map_key,
                            verbose = verbose)
    if (length(res$unmapped) || length(res$unfilled)) {
        stop("T5 encoder load: ", length(res$unmapped), " unmapped keys, ",
             length(res$unfilled), " unfilled params")
    }

    model$to(device = device)
    model$eval()
    model
}

#' Encode prompts with the T5 encoder
#'
#' Tokenizes with \code{\link{encode_unigram}} (right padding to
#' \code{max_sequence_length}) and runs the encoder. Matching the FLUX
#' reference pipeline, no attention mask is used.
#'
#' @param prompts Character vector.
#' @param model A \code{\link{t5_encoder}}.
#' @param tokenizer A \code{\link{unigram_tokenizer}}.
#' @param max_sequence_length Integer. Fixed token length (schnell: 256).
#' @param device Device for the input ids (defaults to the model's).
#'
#' @return Tensor [length(prompts), max_sequence_length, d_model].
#'
#' @export
encode_with_t5 <- function(prompts, model, tokenizer,
                           max_sequence_length = 256L, device = NULL) {
    enc <- encode_unigram(tokenizer, prompts, max_length = max_sequence_length)
    device <- device %||% model$shared$weight$device
    ids <- torch::torch_tensor(enc$input_ids + 1L,
                               dtype = torch::torch_long(), device = device)
    torch::with_no_grad(model(ids))
}
