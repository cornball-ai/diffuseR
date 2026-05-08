# Converted from PyTorch by pyrotechnics
# Review: indexing (0->1 based), integer literals (add L),
# and block structure (braces may need adjustment)

# from typing import Optional

# import torch


res_block <- nn_module(
  "ResBlock",

  initialize = function(channels, mid_channels= NULL, dims= 3) {
    if (is.null(mid_channels)) {
        mid_channels <- channels
    conv <- torch.nn_conv2d if dims == 2 else torch.nn_conv3d
    self$conv1 <- conv(channels, mid_channels, kernel_size=3, padding=1)
    self$norm1 <- torch.nn_group_norm(32, mid_channels)
    self$conv2 <- conv(mid_channels, channels, kernel_size=3, padding=1)
    self$norm2 <- torch.nn_group_norm(32, channels)
    self$activation <- torch.nn_silu()
  },

  forward = function(x) {
    residual <- x
    x <- self$conv1(x)
    x <- self$norm1(x)
    x <- self$activation(x)
    x <- self$conv2(x)
    x <- self$norm2(x)
    x <- self$activation(x + residual)
    return(x)
  }

)

