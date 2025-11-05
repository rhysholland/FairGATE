#' Create a Sankey Plot (robust; aggregates to one row per subject)
#'
#' Visualises routing from Actual Group -> Assigned Expert (2-axis), or
#' Actual Group -> Learned Feature Profile -> Assigned Expert (3-axis)
#' when feature mapping is available.
#'
#' @param prepared_data List from `prepare_data()`; used for group labels and (optionally) feature mapping.
#'                      Needs `feature_names` for 3-axis; `subject_ids` to align subjects to X.
#' @param gnn_results   List from `train_gnn()`; uses $final_results and $gate_weights.
#' @param expert_results Optional list from `analyse_experts()`; used only for picking opposed features in 3-axis.
#' @param top_n_per_side Integer; number of features per side to define Profile A/B (default 2) for 3-axis.
#' @param use_profiles  Logical; try 3-axis when possible (default TRUE). If FALSE, always do 2-axis.
#' @param verbose       Logical; print progress.
#'
#' @return A ggplot object.
#' @export
#' @import ggplot2
#' @import ggalluvial
#' @importFrom dplyr group_by summarise across n left_join mutate select starts_with arrange desc slice ungroup
#' @importFrom rlang .data
plot_sankey <- function(prepared_data,
                        gnn_results,
                        expert_results = NULL,
                        top_n_per_side = 2,
                        use_profiles = TRUE,
                        verbose = FALSE) {

  msg <- function(...) if (isTRUE(verbose)) message(...)

  # ---- Basic checks ----
  fr <- gnn_results$final_results
  gw <- gnn_results$gate_weights
  if (is.null(fr) || !all(c("subjectid","group") %in% names(fr)))
    stop("plot_sankey(): gnn_results$final_results must contain 'subjectid' and 'group'.")
  if (is.null(gw) || !"subjectid" %in% names(gw))
    stop("plot_sankey(): gnn_results$gate_weights must contain 'subjectid'.")

  # ---- 1) Collapse to one row per subject ----
  # Group code per subject (majority vote; tie -> first)
  mode_int <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }
  fr_subj <- fr |>
    dplyr::group_by(.data$subjectid) |>
    dplyr::summarise(group = mode_int(.data$group), .groups = "drop")

  # Mean gate probs per subject across repeats
  prob_cols <- grep("^gate_prob_expert_", names(gw), value = TRUE)
  if (length(prob_cols) < 2L)
    stop("plot_sankey(): need at least two experts (no. of gate_prob_expert_* columns < 2).")

  gw_subj <- gw |>
    dplyr::group_by(.data$subjectid) |>
    dplyr::summarise(dplyr::across(dplyr::starts_with("gate_prob_expert_"), ~ mean(.x, na.rm = TRUE)),
                     .groups = "drop")

  # ---- 2) Map codes -> readable labels using prepare_data() attribute ----
  gm <- attr(prepared_data, "group_mappings")
  fr_subj$Actual_Group <- as.character(fr_subj$group)
  if (!is.null(gm) && length(gm)) {
    flat <- unlist(gm, use.names = TRUE)
    nm   <- as.character(names(flat))
    val  <- as.character(unname(flat))
    # If names are codes and values are labels, keep; else invert
    have_code_in_names <- any(nm %in% fr_subj$Actual_Group)
    have_code_in_vals  <- any(val %in% fr_subj$Actual_Group)
    map_vec <- if (have_code_in_names && !have_code_in_vals) {
      stats::setNames(val, nm)
    } else if (!have_code_in_names && have_code_in_vals) {
      stats::setNames(nm, val)
    } else {
      stats::setNames(val, nm)
    }
    mapped <- unname(map_vec[fr_subj$Actual_Group])
    fr_subj$Actual_Group[!is.na(mapped)] <- mapped[!is.na(mapped)]
  }

  # ---- 3) Assigned expert per subject (argmax of mean probs) ----
  msg("Sankey: deriving assigned expert per subject...")
  max_idx <- apply(gw_subj[, prob_cols, drop = FALSE], 1, function(r) which.max(r))
  assigned_idx <- as.integer(max_idx)

  # Name experts by which group they favour on average (readable legend)
  tmp_means <- sapply(prob_cols, function(pc) {
    tapply(gw_subj[[pc]][match(fr_subj$subjectid, gw_subj$subjectid)],
           fr_subj$Actual_Group, mean, na.rm = TRUE)
  })
  if (is.null(dim(tmp_means))) {
    tmp_means <- matrix(tmp_means, nrow = length(unique(fr_subj$Actual_Group)),
                        dimnames = list(unique(fr_subj$Actual_Group), prob_cols))
  }
  base_names <- rownames(tmp_means)[apply(tmp_means, 2, function(col) {
    i <- which.max(col); ifelse(length(i), i[1], NA_integer_)
  })]
  base_names[is.na(base_names)] <- paste("Expert", seq_along(base_names))
  base_names <- make.unique(base_names, sep = " ")
  expert_labels <- paste0(base_names, " Expert")

  # ---- 4) Assemble per-subject table (aligned sizes) ----
  subj_tbl <- fr_subj |>
    dplyr::left_join(gw_subj, by = "subjectid")

  if (nrow(subj_tbl) != length(assigned_idx))
    stop("Internal size mismatch after aggregation; please report this issue.")

  subj_tbl$Assigned_Expert <- expert_labels[assigned_idx]

  # ---- 5) Decide 2-axis vs 3-axis ----
  can_3_axis <- isTRUE(use_profiles) &&
    !is.null(prepared_data$X) &&
    !is.null(prepared_data$subject_ids) &&
    !is.null(prepared_data$feature_names)

  if (!can_3_axis) {
    msg("Sankey: using 2-axis (group -> expert).")
    counts2 <- subj_tbl |>
      dplyr::group_by(.data$Actual_Group, .data$Assigned_Expert) |>
      dplyr::summarise(N = dplyr::n(), .groups = "drop")

    p <- ggplot2::ggplot(
      counts2,
      ggplot2::aes(axis1 = Actual_Group, axis2 = Assigned_Expert, y = N)
    ) +
      ggalluvial::geom_alluvium(ggplot2::aes(fill = Actual_Group), width = 1/8, alpha = 0.7) +
      ggalluvial::geom_stratum(width = 1/8, fill = "grey90", colour = "black") +
      ggalluvial::stat_stratum(geom = "text", ggplot2::aes(label = after_stat(stratum)), size = 3.5) +
      ggplot2::scale_x_discrete(limits = c("Actual Group", "Assigned Expert")) +
      ggplot2::labs(title = "Patient Routing by Subgroup and Assigned Expert",
                    y = "Number of Patients", fill = "Group") +
      ggplot2::theme_minimal()
    return(p)
  }

  # ---- 6) 3-axis: build simple A/B feature profiles (requires subject alignment) ----
  msg("Sankey: building 3-axis with feature profiles...")
  # Align subjects to rows of X
  idx_map <- match(subj_tbl$subjectid, prepared_data$subject_ids)
  keep <- which(!is.na(idx_map))
  if (length(keep) == 0L) {
    msg("No subject alignment to X; falling back to 2-axis.")
    return(
      ggplot2::ggplot(
        subj_tbl |>
          dplyr::group_by(.data$Actual_Group, .data$Assigned_Expert) |>
          dplyr::summarise(N = dplyr::n(), .groups = "drop"),
        ggplot2::aes(axis1 = Actual_Group, axis2 = Assigned_Expert, y = N)
      ) +
        ggalluvial::geom_alluvium(ggplot2::aes(fill = Actual_Group), width = 1/8, alpha = 0.7) +
        ggalluvial::geom_stratum(width = 1/8, fill = "grey90", colour = "black") +
        ggalluvial::stat_stratum(geom = "text", ggplot2::aes(label = after_stat(stratum)), size = 3.5) +
        ggplot2::scale_x_discrete(limits = c("Actual Group", "Assigned Expert")) +
        ggplot2::labs(title = "Patient Routing by Subgroup and Assigned Expert",
                      y = "Number of Patients", fill = "Group") +
        ggplot2::theme_minimal()
    )
  }

  X <- as.matrix(prepared_data$X)
  fn <- prepared_data$feature_names

  # Choose opposed features from expert_results (fallback to variance)
  choose_feats <- function() {
    if (!is.null(expert_results) && !is.null(expert_results$pairwise_differences)) {
      all_diffs <- tryCatch(do.call(rbind, expert_results$pairwise_differences), error = function(e) NULL)
      if (!is.null(all_diffs) && all(c("feature","difference") %in% names(all_diffs))) {
        A <- unique(all_diffs |>
                      dplyr::arrange(.data$difference) |>
                      dplyr::slice(seq_len(min(top_n_per_side, n()))) |>
                      (\(df) df$feature)())
        B <- unique(all_diffs |>
                      dplyr::arrange(dplyr::desc(.data$difference)) |>
                      dplyr::slice(seq_len(min(top_n_per_side, n()))) |>
                      (\(df) df$feature)())
        return(list(A = intersect(A, fn), B = intersect(B, fn)))
      }
    }
    v <- apply(X, 2, stats::var, na.rm = TRUE)
    ord <- order(v, decreasing = TRUE)
    list(A = fn[ord[seq_len(min(top_n_per_side, length(ord)))]],
         B = fn[rev(ord)[seq_len(min(top_n_per_side, length(ord)))]])
  }
  feats <- choose_feats()
  if (length(feats$A) == 0L && length(feats$B) == 0L) {
    msg("No usable profile features; falling back to 2-axis.")
    counts2 <- subj_tbl |>
      dplyr::group_by(.data$Actual_Group, .data$Assigned_Expert) |>
      dplyr::summarise(N = dplyr::n(), .groups = "drop")
    return(
      ggplot2::ggplot(
        counts2,
        ggplot2::aes(axis1 = Actual_Group, axis2 = Assigned_Expert, y = N)
      ) +
        ggalluvial::geom_alluvium(ggplot2::aes(fill = Actual_Group), width = 1/8, alpha = 0.7) +
        ggalluvial::geom_stratum(width = 1/8, fill = "grey90", colour = "black") +
        ggalluvial::stat_stratum(geom = "text", ggplot2::aes(label = after_stat(stratum)), size = 3.5) +
        ggplot2::scale_x_discrete(limits = c("Actual Group", "Assigned Expert")) +
        ggplot2::labs(title = "Patient Routing by Subgroup and Assigned Expert",
                      y = "Number of Patients", fill = "Group") +
        ggplot2::theme_minimal()
    )
  }

  A_mat <- if (length(feats$A)) X[idx_map[keep], feats$A, drop = FALSE] else matrix(0, nrow = length(keep), ncol = 1)
  B_mat <- if (length(feats$B)) X[idx_map[keep], feats$B, drop = FALSE] else matrix(0, nrow = length(keep), ncol = 1)
  A_score <- rowSums(A_mat, na.rm = TRUE)
  B_score <- rowSums(B_mat, na.rm = TRUE)
  profile <- ifelse(A_score > B_score, "Profile A", "Profile B")

  joined3 <- data.frame(
    Actual_Group    = subj_tbl$Actual_Group[keep],
    Feature_Profile = profile,
    Assigned_Expert = expert_labels[assigned_idx[keep]],
    stringsAsFactors = FALSE
  )

  counts3 <- joined3 |>
    dplyr::group_by(.data$Actual_Group, .data$Feature_Profile, .data$Assigned_Expert) |>
    dplyr::summarise(N = dplyr::n(), .groups = "drop")

  ggplot2::ggplot(
    counts3,
    ggplot2::aes(axis1 = Actual_Group, axis2 = Feature_Profile, axis3 = Assigned_Expert, y = N)
  ) +
    ggalluvial::geom_alluvium(ggplot2::aes(fill = Actual_Group), width = 1/8, alpha = 0.7) +
    ggalluvial::geom_stratum(width = 1/8, fill = "grey90", colour = "black") +
    ggalluvial::stat_stratum(geom = "text", ggplot2::aes(label = after_stat(stratum)), size = 3.5) +
    ggplot2::scale_x_discrete(limits = c("Actual Group", "Learned Feature Profile", "Assigned Expert")) +
    ggplot2::labs(title = "Patient Routing by Subgroup, Feature Profile, and Assigned Expert",
                  y = "Number of Patients", fill = "Group") +
    ggplot2::theme_minimal()
}
