# Memory-mapped safetensors reads for disk-backed staging. Each shard is
# mapped privately (copy-on-write, so nothing can write back into the
# artifact) as one uint8 tensor; a tensor's bytes are a narrow of that
# map, copied to the target device and only then reinterpreted as their
# dtype. R torch has no zero-copy Tensor$view(dtype), and the shards
# the R safetensors writer produces do not align tensor data to the
# element size, so a typed view of the host bytes is not available.
# Measured on an LTX-2.5 shard: 4.5 GB/s to the GPU from the page cache
# (safetensors' get_tensor: 0.68 GB/s), ~1.4 GB/s cold from NVMe, and
# no anonymous host memory: the bytes stay page cache the kernel can
# drop.

.st_dtypes <- function() {
    list(BF16 = torch::torch_bfloat16(), F16 = torch::torch_float16(),
         F32 = torch::torch_float32(), F64 = torch::torch_float64(),
         I64 = torch::torch_long(), I32 = torch::torch_int32(),
         I16 = torch::torch_int16(), I8 = torch::torch_int8(),
         U8 = torch::torch_uint8(), BOOL = torch::torch_bool(),
         F8_E4M3 = torch::torch_float8_e4m3fn(),
         F8_E5M2 = torch::torch_float8_e5m2())
}

# The header of a safetensors file: the byte where tensor data starts,
# and each tensor's dtype, shape and [start, end) data offsets
.st_header <- function(path) {
    con <- file(path, "rb")
    on.exit(close(con))
    n <- readBin(con, "integer", n = 1L, size = 8L, endian = "little")
    hdr <- jsonlite::fromJSON(rawToChar(readBin(con, "raw", n = n)),
                              simplifyVector = FALSE)
    hdr[["__metadata__"]] <- NULL
    list(data_start = 8 + n, tensors = hdr)
}

# A reader over `paths`, or NULL when this torch cannot map files (the
# caller falls back to safetensors' get_tensor). Shards are mapped on
# first use. read(key, device) returns the tensor on `device`.
.mmap_reader <- function(paths) {
    from_file <- get0("torch_from_file", envir = asNamespace("torch"))
    view_copy <- get0("torch_view_copy", envir = asNamespace("torch"))
    if (!is.function(from_file) || !is.function(view_copy)) {
        return(NULL)
    }
    dtypes <- .st_dtypes()
    index <- new.env(parent = emptyenv(), size = 8192L)
    for (i in seq_along(paths)) {
        h <- .st_header(paths[[i]])
        for (k in names(h$tensors)) {
            info <- h$tensors[[k]]
            off <- unlist(info$data_offsets)
            # Byte offset kept as a double: shards can pass 2^31 bytes
            index[[k]] <- list(shard = i,
                               offset = h$data_start + off[[1]],
                               len = as.integer(off[[2]] - off[[1]]),
                               dtype = info$dtype,
                               shape = as.integer(unlist(info$shape)))
        }
    }
    maps <- vector("list", length(paths))
    shard_map <- function(i) {
        if (is.null(maps[[i]])) {
            maps[[i]] <<- from_file(paths[[i]], shared = FALSE,
                                    size = file.size(paths[[i]]),
                                    options = list(dtype = torch::torch_uint8()))
        }
        maps[[i]]
    }
    read <- function(key, device = "cpu") {
        e <- index[[key]]
        if (is.null(e)) {
            stop("Key not found in the mapped shards: ", key, call. = FALSE)
        }
        dtype <- dtypes[[e$dtype]]
        if (is.null(dtype)) {
            stop("Unsupported safetensors dtype ", e$dtype, " for ", key,
                 call. = FALSE)
        }
        if (e$len == 0L) {
            return(torch::torch_empty(e$shape, dtype = dtype, device = device))
        }
        # as_strided takes the offset as int64; narrow() would cap it at
        # R's integer range
        t <- torch::torch_as_strided(shard_map(e$shard), e$len, 1L,
                                     storage_offset = e$offset)$to(device = device)
        # A move to the GPU lands in fresh, aligned storage. On the CPU
        # the bytes are still the map: copy them, so the result is
        # aligned for the dtype view and a write to it cannot change
        # what later reads of the map return
        if (identical(t$device$type, "cpu")) {
            t <- t$clone()
        }
        if (!identical(e$dtype, "U8")) {
            t <- view_copy(t, dtype = dtype)
        }
        if (length(e$shape)) t$view(e$shape) else t$squeeze()
    }
    list(read = read, keys = function() ls(index))
}
