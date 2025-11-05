.require_torch <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("This function requires the 'torch' package. Install it, then run torch::install_torch().",
         call. = FALSE)
  }
  ok <- try(torch::torch_is_installed(), silent = TRUE)
  if (!isTRUE(ok)) {
    stop("LibTorch backend not found. Please run: torch::install_torch()", call. = FALSE)
  }
  invisible(TRUE)
}
