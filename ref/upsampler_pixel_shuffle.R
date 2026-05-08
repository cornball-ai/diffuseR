# Converted from PyTorch by pyrotechnics
# Review: indexing (0->1 based), integer literals (add L),
# and block structure (braces may need adjustment)

# import torch
# from einops import rearrange


pixel_shuffle_n_d <- nn_module(
  "PixelShuffleND",

  initialize = function(dims, upscale_factors, int, int] = (2, 2, 2)) {
    assert dims in [1, 2, 3], "dims must be 1, 2, || 3"
    self$dims <- dims
    self$upscale_factors <- upscale_factors
  },

  forward = function(x) {
    if (self$dims == 3) {
        return(rearrange()
            x,
            "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
            p1 <- self$upscale_factors[0],
            p2 <- self$upscale_factors[1],
            p3 <- self$upscale_factors[2],
        )
    } else if (self$dims == 2) {
        return(rearrange()
            x,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1 <- self$upscale_factors[0],
            p2 <- self$upscale_factors[1],
        )
    } else if (self$dims == 1) {
        return(rearrange()
            x,
            "b (c p1) f h w -> b c (f p1) h w",
            p1 <- self$upscale_factors[0],
        )
    } else {
        raise ValueError(sprintf("Unsupported dims: {self$dims}"))
  }

)

