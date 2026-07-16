# Skeleton construction for checkpoint loading
#
# Building a module allocates and initializes every weight, only for
# the loader to overwrite each one from the checkpoint or swap the
# layer for a quantized module. At LTX-2.3 scale that wrote ~90 GB of
# float32 pages before the first checkpoint byte was read (and the
# follow-up bfloat16 cast held both copies alive), OOM-killing a
# 125 GB machine. A skeleton skips the writes: linear and embedding
# weights are allocated with torch_empty() and never touched (the
# pages stay virtual), and tensor creation defaults to the target
# dtype so no post-construction cast pass is needed.

.noinit_state <- new.env(parent = emptyenv())
.noinit_state$active <- FALSE

# Construct ctor(...) as a load skeleton: linear_noinit/embedding_noinit
# weights stay uninitialized and the torch default dtype is `dtype` for
# the duration. Only for modules whose parameters and buffers are all
# filled from a checkpoint afterwards (the group loaders validate
# exactly that); constructor-computed buffers keep their values but
# follow `dtype`, so precision-sensitive constructors (the vocoder's
# filter banks) must not go through here.
.construct_skeleton <- function(ctor, ..., dtype = torch::torch_bfloat16()) {
    old_dtype <- torch::torch_get_default_dtype()
    torch::torch_set_default_dtype(dtype)
    .noinit_state$active <- TRUE
    on.exit(
            {
        torch::torch_set_default_dtype(old_dtype)
        .noinit_state$active <- FALSE
    },
            add = TRUE
    )
    ctor(...)
}

# Drop-in nn_linear that skips reset_parameters() inside
# .construct_skeleton(); bare construction initializes exactly like
# torch::nn_linear, so random-weight forwards in tests are unaffected.
linear_noinit <- torch::nn_module(
                                  "linear_noinit",
                                  initialize = function(in_features, out_features, bias = TRUE) {
    self$in_features <- in_features
    self$out_features <- out_features
    self$weight <- torch::nn_parameter(torch::torch_empty(out_features,
            in_features))
    if (bias) {
        self$bias <- torch::nn_parameter(torch::torch_empty(out_features))
    } else {
        self$bias <- NULL
    }
    if (!.noinit_state$active) {
        self$reset_parameters()
    }
},
                                  reset_parameters = function() {
    torch::nn_init_kaiming_uniform_(self$weight, a = sqrt(5))
    if (!is.null(self$bias)) {
        bound <- 1 / sqrt(self$in_features)
        torch::nn_init_uniform_(self$bias, -bound, bound)
    }
},
                                  forward = function(input) {
    torch::nnf_linear(input, self$weight, self$bias)
}
)

# Drop-in nn_embedding (no padding/norm options) with the same skip.
embedding_noinit <- torch::nn_module(
                                     "embedding_noinit",
                                     initialize = function(num_embeddings, embedding_dim) {
    self$num_embeddings <- num_embeddings
    self$embedding_dim <- embedding_dim
    self$weight <- torch::nn_parameter(torch::torch_empty(num_embeddings,
            embedding_dim))
    if (!.noinit_state$active) {
        torch::nn_init_normal_(self$weight)
    }
},
                                     forward = function(input) {
    torch::nnf_embedding(input, self$weight)
}
)
