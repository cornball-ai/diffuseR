# Dispatcher argument validation: an unknown model_name must fail fast
# with match.arg's message instead of reaching switch() or a vector
# if() condition (bare txt2img("prompt") used to error with "EXPR must
# be a length 1 vector"; img2img errored on the length-2 default).

expect_error(txt2img("a cat", model_name = "nope"), "should be one of")
expect_error(img2img("cat.png", "a cat", model_name = "nope"),
             "should be one of")
