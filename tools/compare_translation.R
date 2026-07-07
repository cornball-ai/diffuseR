# Structural translation audit for the FLUX port, using treesitR.
#
# Parses the Python reference (diffusers / transformers, Apache-2.0) and
# the R port, extracts numeric literals and torch-op call names per
# pairing, and reports the set differences. The point is a short,
# reviewable drift report - not a zero-diff goal: R-side plumbing
# (seq_len, message, ...) and Python-side plumbing (isinstance, super,
# ...) are filtered, the rest is eyeballed.
#
# Run from the package root:  r tools/compare_translation.R
# Requires: treesitR, the ref/ symlink, and tools/cache/modeling_t5.py
# (curl it from transformers v4.49.0 if absent).

library(treesitR)

DIFFUSERS <- "ref/upstream/diffusers/src/diffusers"

# ---- tree helpers -------------------------------------------------------------

parse_file <- function(path, language) {
    p <- ts_parser_new()
    ts_parser_set_language(p, language)
    src <- paste(readLines(path, warn = FALSE), collapse = "\n")
    ts_tree_root_node(ts_parse(p, src))
}

walk_collect <- function(node, fn) {
    acc <- list()
    recurse <- function(n) {
        r <- fn(n)
        if (!is.null(r)) {
            acc[[length(acc) + 1L]] <<- r
        }
        for (child in ts_node_children(n, named = TRUE)) {
            recurse(child)
        }
    }
    recurse(node)
    acc
}

# Named scopes (class_definition / function_definition) from a Python tree
py_scopes <- function(root, names) {
    walk_collect(root, function(n) {
        if (ts_node_type(n) %in% c("class_definition", "function_definition")) {
            kids <- ts_node_children(n, named = TRUE)
            if (length(kids) && ts_node_type(kids[[1]]) == "identifier" &&
                ts_node_text(kids[[1]]) %in% names) {
                return(n)
            }
        }
        NULL
    })
}

# Numeric literal values under a node (sign-insensitive)
literals <- function(node) {
    vals <- walk_collect(node, function(n) {
        if (ts_node_type(n) %in% c("integer", "float")) {
            txt <- gsub("L$|_", "", ts_node_text(n))
            v <- suppressWarnings(as.numeric(txt))
            if (!is.na(v)) {
                return(v)
            }
        }
        NULL
    })
    sort(unique(unlist(vals)))
}

# Canonical call names under a node: last identifier of the callee,
# stripped of torch_/nnf_ prefixes
call_names <- function(node) {
    names <- walk_collect(node, function(n) {
        if (ts_node_type(n) != "call") {
            return(NULL)
        }
        callee <- ts_node_children(n, named = TRUE)[[1]]
        if (ts_node_type(callee) %in%
            c("attribute", "extract_operator", "namespace_operator")) {
            kids <- ts_node_children(callee, named = TRUE)
            callee <- kids[[length(kids)]]
        }
        if (ts_node_type(callee) != "identifier") {
            return(NULL)
        }
        sub("^torch_|^nnf_", "", ts_node_text(callee))
    })
    sort(unique(unlist(names)))
}

# Plumbing that legitimately exists on only one side
PY_NOISE <- c(
              "super", "range", "len", "isinstance", "hasattr", "getattr", "print",
              "int", "float", "str", "list", "dict", "tuple", "set", "zip",
              "enumerate", "ValueError", "ImportError", "warning", "warn",
              "deprecate", "items", "keys", "values", "pop", "update", "get",
              "append", "join", "startswith", "endswith", "register_to_config",
              "apply_lora_scale", "maybe_allow_in_graph", "ceil", "is_grad_enabled",
              "_gradient_checkpointing_func", "register_buffer", "ModuleList",
              "Module", "signature", "is_torch_npu_available",
              "maybe_adjust_dtype_for_device", "dispatch_attention_fn",
              "set_processor", "processor", "retrieve_timesteps", "register_modules",
              "randn_tensor", "postprocess", "numpy", "maybe_free_model_hooks",
              "MultiPipelineCallbacks", "PipelineCallback", "progress_bar"
)
R_NOISE <- c(
             "c", "list", "seq_len", "seq_along", "lapply", "vapply", "function",
             "paste0", "paste", "message", "stop", "warning", "sprintf", "length",
             "names", "file.path", "is.null", "invisible", "structure", "inherits",
             "getOption", "options", "requireNamespace", "Sys.time", "difftime",
             "Sys.getenv", "Sys.setenv", "close", "txtProgressBar",
             "setTxtProgressBar", "nn_module", "nn_module_list", "gc", "rm",
             "isTRUE", "as.integer", "as.numeric", "nzchar", "nchar", "dirname",
             "path.expand", "file.exists", "dir.exists", "fromJSON", "tryCatch",
             "match.arg", "strrep", "modifyList", "do.call", "Filter", "Negate",
             "grepl", "sub", "gsub", "startsWith", "endsWith", "setdiff", "head",
             "utils", "hub_download", "filename_from_prompt", "save_image",
             "clear_vram", "onload", "offload", "print"
)

report_pair <- function(label, py_nodes, r_files) {
    r_lits <- c()
    r_calls <- c()
    for (f in r_files) {
        root <- parse_file(f, ts_language_r())
        r_lits <- union(r_lits, literals(root))
        r_calls <- union(r_calls, call_names(root))
    }
    py_lits <- sort(unique(unlist(lapply(py_nodes, literals))))
    py_calls <- sort(unique(unlist(lapply(py_nodes, call_names))))

    cat("\n== ", label, " ==\n", sep = "")
    cat("literals only in Python: ",
        paste(setdiff(py_lits, r_lits), collapse = " "), "\n")
    cat("literals only in R:      ",
        paste(setdiff(r_lits, py_lits), collapse = " "), "\n")
    cat("calls only in Python:    ",
        paste(setdiff(setdiff(py_calls, r_calls), PY_NOISE), collapse = " "),
        "\n")
    cat("calls only in R:         ",
        paste(setdiff(setdiff(r_calls, py_calls), R_NOISE), collapse = " "),
        "\n")
}

# ---- pairings -------------------------------------------------------------------

# 1. Transformer stack: blocks, norms, embedders, RoPE
tf <- parse_file(file.path(DIFFUSERS, "models/transformers/transformer_flux.py"),
                 ts_language_python())
norm <- parse_file(file.path(DIFFUSERS, "models/normalization.py"),
                   ts_language_python())
emb <- parse_file(file.path(DIFFUSERS, "models/embeddings.py"),
                  ts_language_python())
act <- parse_file(file.path(DIFFUSERS, "models/attention.py"),
                  ts_language_python())
py_transformer <- c(
                    py_scopes(tf, c("FluxAttnProcessor", "FluxAttention",
                                    "FluxSingleTransformerBlock", "FluxTransformerBlock",
                                    "FluxPosEmbed", "FluxTransformer2DModel",
                                    "_get_qkv_projections")),
                    py_scopes(norm, c("AdaLayerNormZero", "AdaLayerNormZeroSingle",
                                      "AdaLayerNormContinuous")),
                    py_scopes(emb, c("get_1d_rotary_pos_embed", "apply_rotary_emb",
                                     "CombinedTimestepTextProjEmbeddings",
                                     "PixArtAlphaTextProjection", "Timesteps",
                                     "TimestepEmbedding", "get_timestep_embedding")),
                    py_scopes(act, c("FeedForward"))
)
report_pair("transformer stack",
            py_transformer,
            c("R/dit_flux_modules.R", "R/dit_flux.R", "R/rope_flux.R"))

# 2. Pipeline flow
pipe <- parse_file(file.path(DIFFUSERS, "pipelines/flux/pipeline_flux.py"),
                   ts_language_python())
report_pair("pipeline",
            py_scopes(pipe, c("FluxPipeline", "calculate_shift")),
            c("R/txt2img_flux.R"))

# 3. T5 encoder
t5_path <- "tools/cache/modeling_t5.py"
if (file.exists(t5_path)) {
    t5 <- parse_file(t5_path, ts_language_python())
    report_pair("t5 encoder",
                py_scopes(t5, c("T5LayerNorm", "T5DenseGatedActDense",
                                "T5Attention", "T5LayerSelfAttention", "T5LayerFF",
                                "T5Block", "T5Stack", "T5EncoderModel")),
                c("R/t5_text_encoder.R"))
} else {
    cat("\n(t5 encoder pairing skipped: tools/cache/modeling_t5.py missing)\n")
}
