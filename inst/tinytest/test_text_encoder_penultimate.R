# Penultimate hidden-state support for SDXL text encoders. SDXL feeds the
# UNet hidden_states[-2] (the second-to-last block output, pre-final-LN)
# from both encoders, and te2's pooled text_embeds from the full stack.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
    exit_file("torch not fully installed")
}
library(diffuseR)

set.seed(1)
torch::torch_manual_seed(1)

# tiny encoders (embed_dim divisible by num_heads; a pseudo-EOS at the
# highest id so argmax locates the pooled position)
mk1 <- function(pen) {
    m <- text_encoder_native(vocab_size = 50, context_length = 8,
        embed_dim = 32, num_layers = 4, num_heads = 2, mlp_dim = 64,
        apply_final_ln = FALSE, return_penultimate = pen, gelu_type = "quick")
    m$eval()
    m
}
ids <- torch::torch_tensor(matrix(c(40L, 5L, 12L, 49L, 0L, 0L, 0L, 0L),
    nrow = 1), dtype = torch::torch_long())

# --- text_encoder_native: penultimate differs from last, same shape ---
e_last <- mk1(FALSE)
e_pen <- mk1(TRUE)
# copy weights so the two modules are identical apart from the flag
e_pen$load_state_dict(e_last$state_dict())
h_last <- torch::with_no_grad(e_last(ids))
h_pen <- torch::with_no_grad(e_pen(ids))
expect_equal(dim(h_pen), dim(h_last))                       # [1, 8, 32]
expect_true(as.numeric((h_pen - h_last)$abs()$max()) > 1e-4) # genuinely earlier layer

# --- text_encoder2_native: pooled is stack-invariant, hidden is not ---
mk2 <- function(pen) {
    m <- text_encoder2_native(vocab_size = 50, context_length = 8,
        embed_dim = 32, num_layers = 4, num_heads = 2, mlp_dim = 64,
        return_penultimate = pen)
    m$eval()
    m
}
e2_last <- mk2(FALSE)
e2_pen <- mk2(TRUE)
e2_pen$load_state_dict(e2_last$state_dict())
o_last <- torch::with_no_grad(e2_last(ids))
o_pen <- torch::with_no_grad(e2_pen(ids))

# hidden output: penultimate vs last differ
expect_equal(dim(o_pen[[1]]), dim(o_last[[1]]))             # [1, 8, 32]
expect_true(as.numeric((o_pen[[1]] - o_last[[1]])$abs()$max()) > 1e-4)

# pooled output: computed from the full stack, so identical either way
expect_equal(dim(o_pen[[2]]), c(1L, 32L))
expect_true(as.numeric((o_pen[[2]] - o_last[[2]])$abs()$max()) < 1e-5)
