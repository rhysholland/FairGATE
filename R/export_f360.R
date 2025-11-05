#' Export predictions for IBM Fairness 360
#'
#' Create a CSV (or return a data frame) with columns expected by common
#' IBM Fairness 360 workflows:
#'   subjectid, y_true, y_pred, score, group, group_label, plus optional gate columns.
#'
#' @param gnn_results List returned by train_gnn(); must contain $final_results.
#' @param prepared_data List returned by prepare_data(); used to pull group mappings.
#' @param path Optional file path to write the CSV. If NULL, no file is written.
#' @param include_gate_cols Logical; include gate_prob_expert_* and gate_entropy if available. Default TRUE.
#' @param threshold Numeric between 0 and 1 inclusive; classification threshold for y_pred. Default 0.5.
#' @param verbose Logical; if TRUE, prints progress messages. Default FALSE.
#'
#' @return Invisibly returns the data.frame that is written (or would be written).
#' @export
#'
#' @examples
#' \dontrun{
#' tmp <- file.path(tempdir(), "fairness360_input.csv")
#' export_f360_csv(
#'   gnn_results   = final_fit,
#'   prepared_data = prepared_data_chd,
#'   path          = tmp,
#'   include_gate_cols = TRUE,
#'   threshold     = 0.5,
#'   verbose       = TRUE
#' )
#' }
export_f360_csv <- function(gnn_results,
                            prepared_data,
                            path = NULL,
                            include_gate_cols = TRUE,
                            threshold = 0.5,
                            verbose = FALSE) {

  # ---- Validate inputs ----
  if (is.null(gnn_results) || !is.list(gnn_results)) {
    stop("gnn_results must be a list returned by train_gnn().")
  }
  if (!"final_results" %in% names(gnn_results)) {
    stop("gnn_results$final_results is missing.")
  }
  fr <- gnn_results$final_results

  needed_cols <- c("subjectid", "prob", "true")
  missing_cols <- setdiff(needed_cols, names(fr))
  if (length(missing_cols)) {
    stop("final_results is missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  # ---- Core vectors ----
  subjectid <- fr$subjectid
  score     <- as.numeric(fr$prob)
  y_true    <- as.integer(fr$true)
  if (any(is.na(score))) stop("NA values found in final_results$prob.")
  if (any(is.na(y_true))) stop("NA values found in final_results$true.")

  if (!is.numeric(threshold) || length(threshold) != 1L || threshold < 0 || threshold > 1) {
    stop("threshold must be a single numeric value in [0, 1].")
  }
  y_pred <- as.integer(score >= threshold)

  # ---- Group codes and labels ----
  # Prefer group column from final_results; else try joining from gate_weights by subjectid.
  if ("group" %in% names(fr)) {
    group_codes <- fr$group
  } else if ("gate_weights" %in% names(gnn_results) &&
             !is.null(gnn_results$gate_weights) &&
             "group" %in% names(gnn_results$gate_weights) &&
             "subjectid" %in% names(gnn_results$gate_weights)) {

    if (verbose) message("Joining group codes from gate_weights by subjectid...")
    gw <- gnn_results$gate_weights[, c("subjectid", "group")]
    colnames(gw) <- c("subjectid", "group_gw")
    merged <- merge(fr[, "subjectid", drop = FALSE], gw, by = "subjectid", all.x = TRUE, sort = FALSE)
    group_codes <- merged$group_gw
  } else {
    stop("Could not locate group codes. Ensure either final_results$group exists or gate_weights has subjectid and group.")
  }

  # Coerce to integer codes if possible
  suppressWarnings({
    group_codes_int <- as.integer(as.character(group_codes))
  })
  # If coercion fails (all NA), just keep as character and encode
  if (all(is.na(group_codes_int))) {
    # Build codes from factor levels or appearance order
    grp_chr <- as.character(group_codes)
    levs <- if (is.factor(group_codes)) levels(group_codes) else unique(grp_chr)
    map_lab2code <- stats::setNames(seq_along(levs) - 1L, levs)
    group_codes_int <- unname(map_lab2code[grp_chr])
  }
  if (any(is.na(group_codes_int))) {
    bad <- unique(group_codes[is.na(group_codes_int)])
    stop("Unmapped group codes encountered: ", paste(bad, collapse = ", "))
  }

  # Derive labels using mapping from prepared_data when available
  gm <- attr(prepared_data, "group_mappings")  # expected to be codes -> labels
  group_labels <- if (!is.null(gm) && length(gm)) {
    gm_chr <- as.character(gm)
    names_chr <- names(gm)
    # names are codes; values are labels
    lab <- unname(gm_chr[match(as.character(group_codes_int), names_chr)])
    # fallback to codes if any missing
    lab[is.na(lab)] <- as.character(group_codes_int[is.na(lab)])
    lab
  } else {
    # build labels from observed codes if mapping is not present
    if (verbose) message("No group_mappings attribute found; constructing labels from codes.")
    as.character(group_codes_int)
  }

  # ---- Base output frame ----
  df <- data.frame(
    subjectid   = subjectid,
    y_true      = y_true,
    y_pred      = y_pred,
    score       = score,
    group       = group_codes_int,
    group_label = group_labels,
    stringsAsFactors = FALSE
  )

  # ---- Optional: add gate probabilities / entropy ----
  if (isTRUE(include_gate_cols) &&
      "gate_weights" %in% names(gnn_results) &&
      !is.null(gnn_results$gate_weights)) {

    gw <- gnn_results$gate_weights
    if ("subjectid" %in% names(gw)) {
      keep_cols <- c("subjectid",
                     grep("^gate_prob_expert_", names(gw), value = TRUE),
                     intersect("gate_entropy", names(gw)))
      gw_keep <- gw[, keep_cols, drop = FALSE]
      df <- merge(df, gw_keep, by = "subjectid", all.x = TRUE, sort = FALSE)
    } else if (verbose) {
      message("gate_weights present but missing subjectid; skipping gate columns.")
    }
  }

  # ---- Write CSV if requested ----
  if (!is.null(path)) {
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    utils::write.csv(df, file = path, row.names = FALSE)
    if (verbose) message("Wrote CSV to: ", normalizePath(path, mustWork = FALSE))
  }

  invisible(df)
}
