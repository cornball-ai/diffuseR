#' Stable Diffusion 2.1 CLIP text encoder (anvl port of text_encoder_native)
#'
#' anvl re-implementation of \code{diffuseR::text_encoder_native} for the
#' SD 2.1 conditioning path: the OpenCLIP ViT-H text transformer
#' (\code{embed_dim} 1024, 23 layers, 16 heads, \code{head_dim} 64,
#' context length 77). Learned token + positional embeddings (host-side
#' gather, no RoPE), a stack of pre-LayerNorm transformer layers (standard
#' multi-head self-attention with a causal additive mask, then a
#' GELU MLP), and a final LayerNorm.
#'
#' The SD 2.1 pipeline consumes the \strong{final-LayerNorm last hidden
#' state} (\code{apply_final_ln = TRUE} in
#' \code{diffuseR::sd_pipeline_from_safetensors}), not a penultimate
#' \code{clip_skip} layer. The MLP uses the \strong{tanh-approximation
#' GELU} (\code{diffuseR::text_encoder_native}'s default
#' \code{gelu_type = "tanh"}, since the minimized SD 2.1 config carries no
#' \code{hidden_act}), matched here by \code{.yq_clip_gelu}. LayerNorm is
#' affine with eps 1e-5 (torch \code{nn_layer_norm} default).
#'
#' @name anvl_clip
NULL

# CLIP LayerNorm epsilon (torch nn_layer_norm default; the SD 2.1
# checkpoint was trained with it).
.YQ_CLIP_EPS <- 1e-5

# tanh-approximation GELU, matching torch nnf_gelu(x, approximate = "tanh"):
# 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). All scalars enter
# as ambiguous f32? and defer to x's f32 dtype (compute is f32 throughout).
.yq_clip_gelu <- function(x) {
    x3 <- x * x * x
    inner <- (x + x3 * 0.044715) * sqrt(2 / pi)
    x * (anvl::nv_tanh(inner) + 1) * 0.5
}

# Affine LayerNorm over the last dim: yq_layer_norm (no affine) * w + b,
# with w/b [D] broadcast over [..., D]. (Named distinctly from the UNet's
# .yq_sd_ln to avoid cross-file collisions.)
.yq_clip_ln <- function(x, weight, bias, eps = .YQ_CLIP_EPS) {
    s <- anvl::shape(x)
    yunque::yq_layer_norm(x, eps = eps) *
        anvl::nv_broadcast_to(weight, s) + anvl::nv_broadcast_to(bias, s)
}

# One CLIP transformer layer: pre-norm multi-head self-attention (separate
# biased q/k/v/out projections, causal additive mask, scale 1/sqrt(D)) then
# pre-norm GELU MLP (fc1 -> gelu -> fc2), each residual.
.yq_clip_layer <- function(n_heads, head_dim, eps, precision) {
    inner <- n_heads * head_dim

    function(x, mask, w) {
        s <- anvl::shape(x)
        b <- s[1L]
        n <- s[2L]

        h <- .yq_clip_ln(x, w$ln1_w, w$ln1_b, eps)
        q <- yunque::yq_linear(h, w$q_w, w$q_b, precision = precision)
        k <- yunque::yq_linear(h, w$k_w, w$k_b, precision = precision)
        v <- yunque::yq_linear(h, w$v_w, w$v_b, precision = precision)
        to_heads <- function(t) anvl::nv_transpose(
            anvl::nv_reshape(t, c(b, n, n_heads, head_dim)), c(1L, 3L, 2L, 4L))
        q <- to_heads(q)
        k <- to_heads(k)
        v <- to_heads(v)
        attn <- yunque::yq_sdpa(q, k, v, mask = mask, precision = precision)
        attn <- anvl::nv_reshape(anvl::nv_transpose(attn, c(1L, 3L, 2L, 4L)),
                                 c(b, n, inner))
        x <- x + yunque::yq_linear(attn, w$out_w, w$out_b, precision = precision)

        h2 <- .yq_clip_ln(x, w$ln2_w, w$ln2_b, eps)
        mlp <- yunque::yq_linear(
            .yq_clip_gelu(yunque::yq_linear(h2, w$fc1_w, w$fc1_b, precision = precision)),
            w$fc2_w, w$fc2_b, precision = precision)
        x + mlp
    }
}

#' CLIP causal additive attention mask (host-side)
#'
#' Upper-triangular causal mask as an additive bias \code{[B, 1, S, S]}
#' (0 where attended, a large negative where key j > query i). The SD 2.1
#' pipeline feeds no padding mask (CLIP pads with EOS and causal attention
#' makes trailing pads irrelevant), so this is causal-only. Broadcasts
#' against scores inside \code{\link[yunque]{yq_sdpa}}.
#'
#' @param seq_len Integer. Sequence length.
#' @param batch Integer. Batch size.
#' @param device Character. Target device.
#' @param neg Numeric. Masked-position bias.
#'
#' @return AnvlArray \code{[B, 1, S, S]}, f32.
#'
#' @export
yq_clip_mask <- function(seq_len, batch = 1L, device = "cpu", neg = -3.4e38) {
    causal <- matrix(0, seq_len, seq_len)
    causal[upper.tri(causal)] <- neg            # key j > query i masked
    arr <- array(0, dim = c(batch, 1L, seq_len, seq_len))
    for (b in seq_len(batch)) arr[b, 1L, , ] <- causal
    anvl::nv_array(arr, dtype = "f32", device = device)
}

#' CLIP token + positional embedding lookup (host-side gather)
#'
#' Gathers the token-embedding rows for the given ids and adds the learned
#' positional-embedding rows for positions \code{1..S}. Both tables stay R
#' matrices (never resident device tensors); only the summed
#' \code{[B, S, hidden]} result crosses to anvl. Mirrors
#' \code{diffuseR::text_encoder_native}'s
#' \code{token_embedding(ids + 1) + position_embedding[1:S]}.
#'
#' @param token_embedding R matrix \code{[vocab, hidden]} (from
#'   \code{\link{yq_clip_load_weights}}).
#' @param position_embedding R matrix \code{[max_pos, hidden]}.
#' @param ids Integer matrix \code{[B, S]} of 0-based token ids.
#' @param device Character. Target device.
#'
#' @return AnvlArray \code{[B, S, hidden]}, f32.
#'
#' @export
yq_clip_embed <- function(token_embedding, position_embedding, ids,
                          device = "cpu") {
    ids <- matrix(as.integer(ids), nrow = nrow(ids))
    b <- nrow(ids)
    s <- ncol(ids)
    hidden <- ncol(token_embedding)
    tok_rows <- token_embedding[as.integer(t(ids)) + 1L, , drop = FALSE]  # [B*S, hidden]
    pos_rows <- position_embedding[rep(seq_len(s), times = b), , drop = FALSE]
    combined <- tok_rows + pos_rows
    arr <- aperm(array(t(combined), dim = c(hidden, s, b)), c(3L, 2L, 1L))
    anvl::nv_array(arr, dtype = "f32", device = device)
}

#' Stable Diffusion 2.1 CLIP text encoder forward (anvl)
#'
#' Returns a closure over the static OpenCLIP ViT-H configuration;
#' \code{anvl::jit()} the closure. Runs the 23 transformer layers over
#' pre-embedded tokens and (for SD 2.1) applies the final LayerNorm.
#'
#' @param num_layers Integer. Transformer layers (SD 2.1: 23).
#' @param num_heads Integer. Attention heads (16).
#' @param head_dim Integer. Per-head dimension (64).
#' @param eps Numeric. LayerNorm epsilon.
#' @param apply_final_ln Logical. Apply the final LayerNorm (SD 2.1 uses
#'   the final-LN hidden state; \code{FALSE} gives the pre-final-LN state,
#'   e.g. SDXL's penultimate prompt embeds).
#' @param precision Character. Matmul precision.
#'
#' @return Function of \code{(embeds, mask, w)}:
#'   \itemize{
#'     \item embeds \code{[B, S, hidden]} from \code{\link{yq_clip_embed}}
#'     \item mask \code{[B, 1, S, S]} from \code{\link{yq_clip_mask}}
#'     \item w weights pytree (\code{\link{yq_clip_load_weights}})
#'   }
#'   returning the hidden states \code{[B, S, hidden]}.
#'
#' @export
yq_clip_encoder <- function(num_layers = 23L, num_heads = 16L, head_dim = 64L,
                            eps = 1e-5, apply_final_ln = TRUE,
                            precision = "highest") {
    num_layers <- as.integer(num_layers)
    layer <- .yq_clip_layer(as.integer(num_heads), as.integer(head_dim), eps,
                            precision)

    function(embeds, mask, w) {
        x <- embeds
        for (i in seq_len(num_layers)) {
            x <- layer(x, mask, w$layers[[i]])
        }
        if (apply_final_ln) {
            x <- .yq_clip_ln(x, w$final_ln_w, w$final_ln_b, eps)
        }
        x
    }
}

#' Load SD 2.1 CLIP text-encoder weights into an anvl pytree
#'
#' Reads the diffusers \code{CLIPTextModel} \code{model.safetensors}
#' (F16 upcast to f32) and mirrors its key tree. The token- and
#' positional-embedding tables stay R matrices \code{[rows, hidden]} for
#' host-side gather (\code{\link{yq_clip_embed}}); every other weight is
#' wrapped as an \code{AnvlArray}, with attention / MLP linears transposed
#' to \code{[in, out]} for \code{\link[yunque]{yq_linear}}. With
#' \code{strict = TRUE} every checkpoint key must be consumed exactly once,
#' so a wrong architecture or a missed weight fails loudly.
#'
#' @param path Path to \code{text_encoder/model.safetensors} (or a
#'   directory containing it).
#' @param num_layers Integer. Transformer layers (SD 2.1: 23).
#' @param device Character. Target device.
#' @param strict Logical. Enforce the exact key census.
#'
#' @return Weights pytree for \code{\link{yq_clip_encoder}} (plus
#'   \code{token_embedding} / \code{position_embedding} R matrices for
#'   \code{\link{yq_clip_embed}}).
#'
#' @export
yq_clip_load_weights <- function(path, num_layers = 23L, device = "cpu",
                                 strict = TRUE) {
    if (dir.exists(path)) {
        path <- file.path(path, "model.safetensors")
    }
    st <- yunque::yq_st_open(path)
    on.exit(close(st$con))
    seen <- new.env(parent = emptyenv())
    mark <- function(key) assign(key, TRUE, envir = seen)
    raw <- function(key) {
        mark(key)
        anvl::nv_array(yunque::yq_st_read(st, key), dtype = "f32", device = device)
    }
    lin <- function(key) {
        mark(key)
        anvl::nv_array(yunque::yq_st_read(st, key, transpose = TRUE),
                       dtype = "f32", device = device)
    }
    rmat <- function(key) {
        mark(key)
        yunque::yq_st_read(st, key)              # R matrix [rows, hidden]
    }

    P <- "text_model."
    w <- list(
        token_embedding = rmat(paste0(P, "embeddings.token_embedding.weight")),
        position_embedding = rmat(paste0(P, "embeddings.position_embedding.weight")),
        final_ln_w = raw(paste0(P, "final_layer_norm.weight")),
        final_ln_b = raw(paste0(P, "final_layer_norm.bias"))
    )

    w$layers <- lapply(seq_len(num_layers) - 1L, function(i) {
        p <- sprintf("%sencoder.layers.%d.", P, i)
        list(
            ln1_w = raw(paste0(p, "layer_norm1.weight")),
            ln1_b = raw(paste0(p, "layer_norm1.bias")),
            q_w = lin(paste0(p, "self_attn.q_proj.weight")),
            q_b = raw(paste0(p, "self_attn.q_proj.bias")),
            k_w = lin(paste0(p, "self_attn.k_proj.weight")),
            k_b = raw(paste0(p, "self_attn.k_proj.bias")),
            v_w = lin(paste0(p, "self_attn.v_proj.weight")),
            v_b = raw(paste0(p, "self_attn.v_proj.bias")),
            out_w = lin(paste0(p, "self_attn.out_proj.weight")),
            out_b = raw(paste0(p, "self_attn.out_proj.bias")),
            ln2_w = raw(paste0(p, "layer_norm2.weight")),
            ln2_b = raw(paste0(p, "layer_norm2.bias")),
            fc1_w = lin(paste0(p, "mlp.fc1.weight")),
            fc1_b = raw(paste0(p, "mlp.fc1.bias")),
            fc2_w = lin(paste0(p, "mlp.fc2.weight")),
            fc2_b = raw(paste0(p, "mlp.fc2.bias"))
        )
    })

    if (strict) {
        all_keys <- setdiff(names(st$header), "__metadata__")
        used <- ls(seen)
        unused <- setdiff(all_keys, used)
        extra <- setdiff(used, all_keys)
        if (length(extra)) {
            stop("SD21 CLIP anvl load: ", length(extra),
                 " requested keys not in file, e.g. ",
                 paste(utils::head(extra, 5), collapse = ", "))
        }
        if (length(unused)) {
            stop("SD21 CLIP anvl load: ", length(unused),
                 " checkpoint keys unused, e.g. ",
                 paste(utils::head(unused, 5), collapse = ", "))
        }
    }

    w
}
