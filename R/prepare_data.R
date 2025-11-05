#' Prepare Data for GNN Training
#'
#' This function takes a raw dataframe, cleans it, defines the outcome and
#' group variables, and scales the feature matrix. If no `group_mappings`
#' are provided, they are automatically generated from the unique values
#' (or factor levels) of `group_var`.
#'
#' @param data A dataframe containing the raw data.
#' @param outcome_var A string with the column name of the binary outcome (must be 0 or 1).
#' @param group_var A string with the column name of the sensitive attribute.
#' @param group_mappings Optional named list mapping values in `group_var` to numeric codes (e.g., `list("Male" = 0, "Female" = 1)`).
#' @param cols_to_remove A character vector of column names to exclude from the feature matrix (e.g., IDs, highly collinear vars).
#'
#' @return A list containing:
#'   \item{X}{The scaled feature matrix.}
#'   \item{y}{The numeric outcome vector.}
#'   \item{group}{The numeric group vector.}
#'   \item{feature_names}{The names of the features used.}
#'   \item{subject_ids}{A vector of subject IDs, if a 'subjectid' column exists.}
#'   \item{group_mappings}{Added as an attribute for downstream use.}
#'
#' @export
#' @importFrom dplyr select any_of where
#' @importFrom magrittr %>%
#'
#' @examples
#' my_data <- data.frame(
#'   subjectid = 1:10,
#'   remission = sample(0:1, 10, replace = TRUE),
#'   gender = sample(c("M", "F"), 10, replace = TRUE),
#'   feature1 = rnorm(10),
#'   feature2 = rnorm(10)
#' )
#'
#' prepared_data <- prepare_data(
#'   data = my_data,
#'   outcome_var = "remission",
#'   group_var = "gender",
#'   cols_to_remove = c("subjectid")
#' )
prepare_data <- function(data, outcome_var, group_var, group_mappings = NULL, cols_to_remove = NULL) {

  # --- Input Validation ---
  if (!outcome_var %in% names(data)) stop("outcome_var not found in data.")
  if (!group_var %in% names(data)) stop("group_var not found in data.")

  # --- Define Outcome (y) ---
  y <- as.numeric(data[[outcome_var]])
  y[is.na(y)] <- 0  # default NA to 0

  # --- Define Group Vector ---
  group_vec_raw <- trimws(as.character(data[[group_var]]))

  # Auto-generate mapping if not provided
  if (is.null(group_mappings)) {
    levels_use <- if (is.factor(data[[group_var]])) levels(data[[group_var]]) else unique(group_vec_raw)
    codes <- seq_along(levels_use) - 1L
    group_mappings <- as.list(stats::setNames(codes, levels_use))
    message(sprintf("Auto-generated group_mappings: %s", paste(names(group_mappings), collapse = ", ")))
  }

  mapping_vector <- unlist(group_mappings)
  names(mapping_vector) <- names(group_mappings)

  group <- as.numeric(mapping_vector[group_vec_raw])

  if (any(is.na(group))) {
    stop("NA values found in group vector after mapping.\nCheck that all values in your group_var column are present in the group_mappings list.")
  }

  # --- Define Feature Matrix (X) ---
  all_cols_to_remove <- unique(c(outcome_var, group_var, cols_to_remove))
  X <- data %>%
    dplyr::select(-dplyr::any_of(all_cols_to_remove)) %>%
    dplyr::select(dplyr::where(is.numeric))

  feature_names <- colnames(X)
  X <- scale(X)

  # --- Extract Subject IDs if they exist ---
  subject_ids <- if ("subjectid" %in% names(data)) data$subjectid else NULL

  # --- Return ---
  out <- list(
    X = X,
    y = y,
    group = group,
    feature_names = feature_names,
    subject_ids = subject_ids
  )

  # Attach mapping for downstream use (e.g., analyse_experts, plot_sankey)
  attr(out, "group_mappings") <- group_mappings

  return(out)
}
