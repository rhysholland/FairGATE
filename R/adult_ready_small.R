#' Adult ready (small sample)
#'
#' A compact (n = 1000) sample of the pre-cleaned Adult dataset for examples
#' and vignettes. All original columns are retained; only rows are subsampled.
#'
#' @details Created from the full CSV by random sampling with a fixed seed:
#' \code{adult_ready_small <- dplyr::slice_sample(big, n = 1000)}.
#'
#' @format A data frame with 1000 rows and K columns (same schema as the full dataset).
#' @source UCI Adult (pre-cleaned in-package).
#' @examples
#' data("adult_ready_small", package = "fairGATE")
#' str(adult_ready_small)
#' head(adult_ready_small)
#' @keywords datasets
"adult_ready_small"
