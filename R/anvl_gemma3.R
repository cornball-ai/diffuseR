#' Gemma3 rotary tables (host-side, split-half convention)
#'
#' \code{inv_freq = (1 / theta^(arange(0, D-2, 2) / D)) / scaling_factor};
#' cos/sin of the position outer product. Gemma3 uses TWO of these: a
#' global table (theta 1e6, inv_freq divided by the rope_scaling factor)
#' for the every-6th "global" layer, and a local table (theta 1e4, no
#' scaling) for the sliding-window layers. Returned as \code{[S, D/2]}
#' AnvlArrays for the split-half kernel (\code{\link[yunque]{yq_rope_split}}).
#'
#' @param seq_len Integer. Sequence length.
#' @param head_dim Integer. Per-head dim.
#' @param theta Numeric. RoPE base (global 1e6, local 1e4).
#' @param scaling_factor Numeric. Linear RoPE scaling (global 8, local 1);
#'   \code{inv_freq} is divided by it (HuggingFace convention).
#' @param device Character.
#'
#' @return List \code{list(cos, sin)}, each \code{[S, head_dim/2]}, f32.
#'
#' @export
yq_gemma3_rope <- function(seq_len, head_dim, theta = 1e6,
                           scaling_factor = 1.0, device = "cpu") {
    r <- head_dim %/% 2L
    inv_freq <- (1 / theta^((2 * (0:(r - 1L))) / head_dim)) / scaling_factor
    pos <- 0:(seq_len - 1L)
    ang <- outer(pos, inv_freq)                # [S, r]
    list(cos = anvl::nv_array(cos(ang), dtype = "f32", device = device),
         sin = anvl::nv_array(sin(ang), dtype = "f32", device = device))
}

#' Gemma3 additive attention mask (host-side)
#'
#' Causal upper-triangular mask plus per-token padding, as an additive
#' bias \code{[B, 1, S, S]} (0 where attended, \code{neg} where masked),
#' mirroring the reference's \code{(causal + padding) * -1e9}. When
#' \code{window} is set, keys more than \code{window} positions in the
#' past are also masked (the local sliding variant). The reference
#' \code{gemma3_text_model} passes the full (\code{window = NULL}) mask to
#' every layer — its \code{create_sliding_window_mask} is never called —
#' so the encoder uses the full mask; the sliding path is exposed here for
#' completeness. Broadcasts against scores inside \code{\link[yunque]{yq_sdpa}}.
#'
#' @param attention_mask Integer/numeric matrix \code{[B, S]} (1 real,
#'   0 pad), or NULL for causal-only.
#' @param seq_len Integer. Sequence length.
#' @param batch Integer. Batch size.
#' @param window Integer sliding-window size, or NULL for full causal.
#' @param device Character.
#' @param neg Numeric. Per-masked-condition additive bias.
#'
#' @return AnvlArray \code{[B, 1, S, S]}, f32.
#'
#' @export
yq_gemma3_mask <- function(attention_mask, seq_len, batch = 1L,
                           window = NULL, device = "cpu", neg = -1e9) {
    i <- matrix(0:(seq_len - 1L), seq_len, seq_len)              # query rows
    j <- matrix(0:(seq_len - 1L), seq_len, seq_len, byrow = TRUE) # key cols
    causal <- (j > i) * neg                     # future keys masked
    if (!is.null(window)) {
        causal <- causal + ((i - j) > window) * neg  # keys outside the window
    }
    arr <- array(0, dim = c(batch, 1L, seq_len, seq_len))
    for (b in seq_len(batch)) {
        m <- causal
        if (!is.null(attention_mask)) {
            pad <- (1 - attention_mask[b, ]) * neg      # [S] over keys
            m <- m + matrix(pad, seq_len, seq_len, byrow = TRUE)
        }
        arr[b, 1L, , ] <- m
    }
    anvl::nv_array(arr, dtype = "f32", device = device)
}

#' Gemma3 token embedding lookup (host-side gather + sqrt scaling)
#'
#' Gathers rows of the embedding table for the given token ids and scales
#' by \code{sqrt(hidden_size)} (the Gemma embedding scale). The table
#' stays an R matrix (never a resident device tensor); the scaled result
#' is the only thing that crosses to anvl.
#'
#' @param embed R matrix \code{[vocab, hidden]} (from
#'   \code{\link{yq_gemma3_load_weights}}).
#' @param ids Integer matrix \code{[B, S]} of 0-based token ids.
#' @param device Character.
#'
#' @return AnvlArray \code{[B, S, hidden]}, f32.
#'
#' @export
yq_gemma3_embed <- function(embed, ids, device = "cpu") {
    ids <- matrix(as.integer(ids), nrow = nrow(ids))
    b <- nrow(ids); s <- ncol(ids); hidden <- ncol(embed)
    scale <- sqrt(hidden)
    rows <- embed[as.integer(t(ids)) + 1L, , drop = FALSE] * scale  # [B*S, hidden]
    arr <- aperm(array(t(rows), dim = c(hidden, s, b)), c(3L, 2L, 1L))
    anvl::nv_array(arr, dtype = "f32", device = device)
}

# GELU (tanh approximation), matching torch nnf_gelu(approximate = "tanh"):
# 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))).
.yq_gelu_tanh <- function(x) {
    c0 <- sqrt(2 / pi)
    inner <- (x + (x * x * x) * 0.044715) * c0
    (x * (anvl::nv_tanh(inner) + 1)) * 0.5
}

# One Gemma3 decoder layer: Gemma "sandwich" pre/post norms around both
# attention (GQA, per-head q/k RMS norm with (1+weight), split RoPE,
# additive mask) and a GeGLU MLP. Every RMS weight arrives with 1 already
# added at load time, so plain yq_rms_norm reproduces (1 + weight) * x.
.yq_gemma3_layer <- function(num_heads, num_kv, head_dim, eps, precision) {
    inner <- num_heads * head_dim
    groups <- num_heads %/% num_kv
    r <- head_dim %/% 2L

    function(x, cos, sin, mask, w) {
        s <- anvl::shape(x)
        b <- s[1L]; n <- s[2L]

        h <- yunque::yq_rms_norm(x, w$input_ln, eps = eps)
        q <- anvl::nv_reshape(yunque::yq_linear(h, w$q_proj, precision = precision),
                              c(b, n, num_heads, head_dim))
        k <- anvl::nv_reshape(yunque::yq_linear(h, w$k_proj, precision = precision),
                              c(b, n, num_kv, head_dim))
        v <- anvl::nv_reshape(yunque::yq_linear(h, w$v_proj, precision = precision),
                              c(b, n, num_kv, head_dim))
        q <- yunque::yq_rms_norm(q, w$q_norm, eps = eps)   # over head_dim
        k <- yunque::yq_rms_norm(k, w$k_norm, eps = eps)

        perm <- c(1L, 3L, 2L, 4L)              # [B, S, H, D] -> [B, H, S, D]
        q <- anvl::nv_transpose(q, perm)
        k <- anvl::nv_transpose(k, perm)
        v <- anvl::nv_transpose(v, perm)

        cq <- anvl::nv_broadcast_to(anvl::nv_reshape(cos, c(1L, 1L, n, r)),
                                    c(b, num_heads, n, r))
        sq <- anvl::nv_broadcast_to(anvl::nv_reshape(sin, c(1L, 1L, n, r)),
                                    c(b, num_heads, n, r))
        ck <- anvl::nv_broadcast_to(anvl::nv_reshape(cos, c(1L, 1L, n, r)),
                                    c(b, num_kv, n, r))
        sk <- anvl::nv_broadcast_to(anvl::nv_reshape(sin, c(1L, 1L, n, r)),
                                    c(b, num_kv, n, r))
        q <- yunque::yq_rope_split(q, cq, sq)
        k <- yunque::yq_rope_split(k, ck, sk)

        k <- yunque::yq_repeat_kv(k, groups)
        v <- yunque::yq_repeat_kv(v, groups)

        # scale = 1/sqrt(head_dim); the reference uses 1/sqrt(query_pre_attn
        # _scalar) with query_pre_attn_scalar == head_dim (real: 256 == 256).
        attn <- yunque::yq_sdpa(q, k, v, mask = mask, precision = precision)
        attn <- anvl::nv_reshape(anvl::nv_transpose(attn, perm),
                                 c(b, n, inner))
        attn <- yunque::yq_linear(attn, w$o_proj, precision = precision)
        attn <- yunque::yq_rms_norm(attn, w$post_attn_ln, eps = eps)  # post-attn
        x <- x + attn

        h2 <- yunque::yq_rms_norm(x, w$pre_ff_ln, eps = eps)   # pre-feedforward
        mlp <- yunque::yq_linear(
            .yq_gelu_tanh(yunque::yq_linear(h2, w$gate, precision = precision)) *
            yunque::yq_linear(h2, w$up, precision = precision),
            w$down, precision = precision)
        mlp <- yunque::yq_rms_norm(mlp, w$post_ff_ln, eps = eps)  # post-feedforward
        x + mlp
    }
}

#' Gemma3 text encoder forward for LTX-2.3 (anvl port)
#'
#' anvl re-implementation of \code{diffuseR::gemma3_text_model}, returning
#' the stacked hidden states the LTX connectors consume: the scaled
#' embedding output plus every layer's output, with the final layer's
#' output final-RMS-normed (\code{num_layers + 1} states). Mirrors the HF
#' hidden-state semantics: state \code{i} is recorded BEFORE layer \code{i}
#' and the last state is post-final-norm (the un-normed last-layer output
#' never appears). Per-layer the dual RoPE table is selected by the same
#' rule as the reference — the every-\code{pattern}-th layer (global) uses
#' the global table, the rest (sliding) use the local table — and one full
#' causal+padding mask is applied to every layer.
#'
#' @param num_layers Integer. Decoder layers.
#' @param num_heads,num_kv,head_dim Integers. Attention shape.
#' @param eps Numeric. RMS norm epsilon.
#' @param sliding_window_pattern Integer. Global-attention period (6).
#' @param precision Character. Matmul precision.
#'
#' @return Function of (embeds, rope_global, rope_local, mask, w):
#'   \itemize{
#'     \item embeds \code{[B, S, hidden]} from \code{\link{yq_gemma3_embed}}
#'     \item rope_global, rope_local \code{list(cos, sin)} \code{[S, D/2]}
#'       from \code{\link{yq_gemma3_rope}}
#'     \item mask \code{[B, 1, S, S]} from \code{\link{yq_gemma3_mask}}
#'     \item w weights pytree from \code{\link{yq_gemma3_load_weights}}
#'   }
#'   returning \code{[B, S, hidden, num_layers + 1]}.
#'
#' @export
yq_gemma3_encoder <- function(num_layers = 6L, num_heads = 4L, num_kv = 2L,
                              head_dim = 48L, eps = 1e-6,
                              sliding_window_pattern = 6L,
                              precision = "highest") {
    layer <- .yq_gemma3_layer(num_heads, num_kv, head_dim, eps, precision)

    function(embeds, rope_global, rope_local, mask, w) {
        x <- embeds
        states <- vector("list", num_layers + 1L)
        for (i in seq_len(num_layers)) {
            states[[i]] <- x                     # record BEFORE layer i
            # is_sliding == (layer_idx + 1) %% pattern != 0; here i == idx+1
            if (i %% sliding_window_pattern != 0L) {
                rp <- rope_local                 # sliding layer -> local table
            } else {
                rp <- rope_global                # global layer -> global table
            }
            x <- layer(x, rp$cos, rp$sin, mask, w$layers[[i]])
        }
        states[[num_layers + 1L]] <- yunque::yq_rms_norm(x, w$norm, eps = eps)

        nd <- anvl::ndims(x) + 1L                # new trailing axis (stack dim)
        states <- lapply(states, function(s) anvl::nv_unsqueeze(s, nd))
        do.call(anvl::nv_concatenate, c(states, list(dimension = nd)))
    }
}

#' Load Gemma3 text-encoder weights into an anvl pytree
#'
#' Reads the state_dict from a single safetensors file (the parity
#' fixture, or a real checkpoint), transposing 2-D linears to
#' \code{[in, out]} and adding 1 to every RMSNorm weight host-side so
#' plain \code{\link[yunque]{yq_rms_norm}} reproduces Gemma's
#' \code{(1 + weight)} form. The embedding table stays an R matrix for
#' host-side gather (\code{\link{yq_gemma3_embed}}); every other tensor is
#' wrapped as an \code{AnvlArray} on \code{device}.
#'
#' @param path Path to a \code{.safetensors} file holding the state_dict
#'   (keys \code{embed_tokens.weight}, \code{layers.\{i\}.*}, \code{norm.weight};
#'   layer indices 0-based, as HF / R torch emit them).
#' @param num_layers Integer. Decoder layers to load.
#' @param device Character. Target device.
#'
#' @return List \code{list(embed = <R matrix [vocab, hidden]>,
#'   layers = <per-layer weight lists>, norm = <AnvlArray>)}.
#'
#' @export
yq_gemma3_load_weights <- function(path, num_layers = 6L, device = "cpu") {
    st <- yunque::yq_st_open(path)
    on.exit(close(st$con))
    lin <- function(key) anvl::nv_array(yunque::yq_st_read(st, key, transpose = TRUE),
                                        dtype = "f32", device = device)
    # RMSNorm weight with Gemma's +1 folded in.
    nrm <- function(key) anvl::nv_array(yunque::yq_st_read(st, key) + 1,
                                        dtype = "f32", device = device)

    embed <- yunque::yq_st_read(st, "embed_tokens.weight")   # [vocab, hidden]

    layers <- lapply(seq_len(num_layers) - 1L, function(i) {
        p <- sprintf("layers.%d.", i)
        list(
            input_ln     = nrm(paste0(p, "input_layernorm.weight")),
            post_attn_ln = nrm(paste0(p, "post_attention_layernorm.weight")),
            pre_ff_ln    = nrm(paste0(p, "pre_feedforward_layernorm.weight")),
            post_ff_ln   = nrm(paste0(p, "post_feedforward_layernorm.weight")),
            q_proj = lin(paste0(p, "self_attn.q_proj.weight")),
            k_proj = lin(paste0(p, "self_attn.k_proj.weight")),
            v_proj = lin(paste0(p, "self_attn.v_proj.weight")),
            o_proj = lin(paste0(p, "self_attn.o_proj.weight")),
            q_norm = nrm(paste0(p, "self_attn.q_norm.weight")),
            k_norm = nrm(paste0(p, "self_attn.k_norm.weight")),
            gate = lin(paste0(p, "mlp.gate_proj.weight")),
            up   = lin(paste0(p, "mlp.up_proj.weight")),
            down = lin(paste0(p, "mlp.down_proj.weight"))
        )
    })

    list(embed = embed, layers = layers, norm = nrm("norm.weight"))
}
