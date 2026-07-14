#' FLUX.2 Klein timestep sinusoidal projection (host-side)
#'
#' Parameter-free sinusoidal embedding of \code{timestep * 1000},
#' matching diffusers \code{get_timestep_embedding} with
#' \code{flip_sin_to_cos = TRUE}, \code{downscale_freq_shift = 0}.
#' Computed in base R (deterministic, no weights) and returned as an
#' \code{AnvlArray}, mirroring how the RoPE tables are precomputed
#' outside the model.
#'
#' @param timestep Numeric vector (sigma space, 0-1); scaled by 1000
#'   internally.
#' @param dim Integer. Projection width (klein: 256).
#' @param max_period Numeric. Base period (10000).
#' @param device Character. Target device.
#'
#' @return AnvlArray \code{[length(timestep), dim]}, f32.
#'
#' @export
yq_flux2_time_proj <- function(timestep, dim = 256L, max_period = 10000,
                               device = "cpu") {
    t <- as.numeric(timestep) * 1000
    half <- dim %/% 2L
    exponent <- -log(max_period) * (0:(half - 1L)) / half
    freq <- exp(exponent)
    ang <- outer(t, freq)                     # [N, half]
    emb <- cbind(cos(ang), sin(ang))          # flip_sin_to_cos: cos then sin
    anvl::nv_array(emb, dtype = "f32", device = device)
}

#' FLUX.2 Klein transformer forward (anvl port)
#'
#' anvl re-implementation of \code{diffuseR::flux2_transformer} forward
#' (Flux2Transformer2DModel, klein-4B): x/context embedders, a shared
#' timestep-MLP feeding three modulation projections, a stack of
#' double-stream blocks over separate text/image streams, then
#' single-stream blocks over the concatenated sequence, an
#' adaLN-continuous output norm, and the velocity projection.
#'
#' Returns a closure over the static config; \code{anvl::jit()} it.
#' Weights travel as a named pytree (see \code{yq_flux2_load_weights}).
#' The timestep sinusoid (\code{\link{yq_flux2_time_proj}}) and RoPE
#' tables (built from position ids) are precomputed host-side and passed
#' as inputs, matching the diffusers pipeline boundary.
#'
#' @param num_layers Integer. Double-stream blocks (klein: 5).
#' @param num_single_layers Integer. Single-stream blocks (klein: 20).
#' @param heads Integer. Attention heads (24).
#' @param head_dim Integer. Per-head dim (128).
#' @param mlp_ratio Numeric. FF multiplier (3.0).
#' @param eps Numeric. Norm epsilon.
#' @param precision Character. Matmul precision (see
#'   \code{\link[yunque]{yq_linear}}).
#'
#' @return Function of (latents, text_embeds, time_proj, cos, sin, w):
#'   \itemize{
#'     \item latents \code{[B, S_img, in_channels]}
#'     \item text_embeds \code{[B, S_txt, joint_dim]}
#'     \item time_proj \code{[B, 256]} from \code{yq_flux2_time_proj()}
#'     \item cos, sin \code{[S_txt + S_img, head_dim]} RoPE tables
#'     \item w weights pytree
#'   }
#'   returning velocity \code{[B, S_img, out_channels]}.
#'
#' @export
yq_flux2_transformer <- function(num_layers = 5L, num_single_layers = 20L,
                                 heads = 24L, head_dim = 128L,
                                 mlp_ratio = 3.0, eps = 1e-6,
                                 precision = "highest") {
    double_blk <- yq_flux2_double_block(heads, head_dim, mlp_ratio, eps,
                                        precision)
    single_blk <- yq_flux2_single_block(heads, head_dim, mlp_ratio, eps,
                                         precision)

    function(latents, text_embeds, time_proj, cos, sin, w) {
        x <- yunque::linear(latents, w$x_embedder, precision = precision)
        cc <- yunque::linear(text_embeds, w$context_embedder, precision = precision)
        dim <- anvl::shape(x)[3L]

        temb <- yunque::linear(yunque::silu(yunque::linear(time_proj, w$time_1,
                                            precision = precision)),
                          w$time_2, precision = precision)
        sil <- yunque::silu(temb)
        mod_img <- yunque::linear(sil, w$dsm_img, precision = precision)
        mod_txt <- yunque::linear(sil, w$dsm_txt, precision = precision)
        mod_single <- yunque::linear(sil, w$single_mod, precision = precision)
        msingle <- .yq_mod_split(mod_single, 1L, dim)[[1L]]

        for (i in seq_len(num_layers)) {
            res <- double_blk(x, cc, mod_img, mod_txt, cos, sin, w$double[[i]])
            cc <- res[[1L]]
            x <- res[[2L]]
        }

        s_txt <- anvl::shape(cc)[2L]
        hs <- anvl::nv_concatenate(cc, x, dimension = 2L)
        for (i in seq_len(num_single_layers)) {
            wi <- w$single[[i]]
            hs <- single_blk(hs, msingle[[1L]], msingle[[2L]], msingle[[3L]],
                             cos, sin, wi$qkv, wi$out, wi$norm_q, wi$norm_k)
        }
        s_all <- anvl::shape(hs)[2L]
        hs <- yunque::slice_seq(hs, s_txt + 1L, s_all)

        # adaLN-continuous output norm: scale, shift from silu(temb)
        no <- yunque::linear(yunque::silu(temb), w$norm_out, precision = precision)
        scale <- yunque::slice_lastdim(no, 1L, dim)
        shift <- yunque::slice_lastdim(no, dim + 1L, 2L * dim)
        sh <- anvl::shape(hs)
        s2 <- anvl::shape(scale)
        scale <- anvl::nv_reshape(scale, c(s2[1L], 1L, s2[2L]))
        shift <- anvl::nv_reshape(shift, c(s2[1L], 1L, s2[2L]))
        hs <- yunque::layer_norm(hs, eps = eps) *
        anvl::nv_broadcast_to(scale + 1, sh) +
        anvl::nv_broadcast_to(shift, sh)

        yunque::linear(hs, w$proj_out, precision = precision)
    }
}
