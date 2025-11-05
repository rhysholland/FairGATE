#' Analyse and Visualise Expert Network Specialisation
#'
#' Analyses expert input weights to determine which features are most
#' important per subgroup. For two groups, returns a signed
#' difference plot (GroupB - GroupA). For >2 groups, performs a
#' non-inferential analysis: per-group mean importances, pairwise
#' (B - A) difference tables for all pairs, and a multi-group plot
#' for the features with the largest max-min spread across groups.
#'
#' @param gnn_results A list from `train_gnn()`.
#' @param prepared_data A list from `prepare_data()` (used to retrieve group mappings if not provided).
#' @param group_mappings Optional named mapping from group codes (names) to labels (values).
#' @param top_n_features Integer; number of top features to visualise.
#' @param verbose Logical; print progress messages.
#'
#' @return A list with:
#'   \item{all_weights}{Long table of feature importances by group & repeat}
#'   \item{means_by_group_wide}{Wide table of per-feature mean importance per group}
#'   \item{pairwise_differences}{Named list of B_vs_A difference tables (descriptive)}
#'   \item{difference_plot}{ggplot; only when there are exactly 2 groups}
#'   \item{multi_group_plot}{ggplot; only when there are >2 groups}
#'   \item{top_features_multi}{Long table used for the multi-group plot}
#' @export
#' @import ggplot2
#' @import magrittr
#' @importFrom tibble tibble
#' @importFrom dplyr mutate group_by summarise arrange desc left_join filter slice_max across n
#' @importFrom tidyr pivot_wider pivot_longer
#' @importFrom purrr map_dfr
#' @importFrom utils combn
#' @importFrom rlang .data
analyse_experts <- function(gnn_results,
                            prepared_data,
                            group_mappings = NULL,
                            top_n_features = 10,
                            verbose = FALSE) {

  if (verbose) message("Starting Expert Feature Weight Analysis...")

  weights_list  <- gnn_results$expert_weights
  feature_names <- prepared_data$feature_names

  if (length(weights_list) == 0) stop("No expert weights found in gnn_results$expert_weights.")
  if (is.null(feature_names) || length(feature_names) == 0) {
    stop("prepared_data$feature_names is missing or empty.")
  }

  # ---- obtain/normalise group_mappings to codes -> labels ----
  # 1) pull from prepared_data if not provided
  if (is.null(group_mappings) && !is.null(prepared_data)) {
    group_mappings <- attr(prepared_data, "group_mappings")
    if (verbose && !is.null(group_mappings)) {
      message("Using group_mappings retrieved from prepared_data attributes.")
    }
  }

  # 2) normalise orientation so: names = codes (as character), values = labels (character)
  .normalise_group_mappings <- function(gm, G = NULL) {
    if (is.null(gm) || !length(gm)) {
      if (is.null(G)) return(NULL)
      codes  <- as.character(seq_len(G) - 1L)
      labels <- codes
      names(labels) <- codes
      return(labels)
    }
    nm   <- names(gm)
    vals <- unname(gm)
    to_num <- function(x) suppressWarnings(as.numeric(x))

    nm_num  <- to_num(nm)
    val_num <- to_num(vals)

    # Case A: names are codes (numeric), values are labels (character)
    if (all(!is.na(nm_num)) && any(is.na(val_num))) {
      codes  <- as.character(as.integer(nm_num))
      labels <- as.character(vals)
      names(labels) <- codes
      return(labels)
    }
    # Case B: names are labels (character), values are codes (numeric)
    if (all(is.na(nm_num)) && all(!is.na(val_num))) {
      codes  <- as.character(as.integer(val_num))
      labels <- as.character(nm)
      keep <- !duplicated(codes)
      labels <- labels[keep]; codes <- codes[keep]
      names(labels) <- codes
      return(labels)
    }
    # Fallback: coerce everything to character and hope names are codes
    codes  <- as.character(nm)
    labels <- as.character(vals)
    names(labels) <- codes
    labels
  }

  # infer G from first repeat if needed
  G <- length(weights_list[[1]])
  gm_norm <- .normalise_group_mappings(group_mappings, G)
  if (is.null(gm_norm)) {
    codes  <- as.character(seq_len(G) - 1L)
    labels <- codes
    names(labels) <- codes
    gm_norm <- labels
    if (verbose) message(sprintf("No group_mappings provided; inferring %d groups labelled by codes.", G))
  }

  # ---- infer hidden_dim from element count; handle matrix or vector storage ----
  first_block <- weights_list[[1]][[1]]
  elems <- if (is.matrix(first_block)) length(first_block) else length(first_block)
  hidden_dim <- elems / length(feature_names)
  if (!isTRUE(all.equal(hidden_dim, as.integer(hidden_dim)))) {
    stop("Could not infer hidden_dim: weights length is not a multiple of feature count.")
  }
  hidden_dim <- as.integer(hidden_dim)

  # Helper to get a hidden_dim x p matrix for any storage form
  .to_hidden_by_p <- function(w_vec_or_mat, hidden_dim, p) {
    if (is.matrix(w_vec_or_mat)) {
      # Expecting hidden_dim x p (PyTorch first linear is [hidden, input])
      w <- w_vec_or_mat
      if (!identical(dim(w), c(hidden_dim, p))) {
        # If transposed (p x hidden), fix it
        if (identical(dim(w), c(p, hidden_dim))) w <- t(w)
        else {
          # Fall back to reshape
          w <- matrix(as.numeric(w), nrow = hidden_dim, ncol = p, byrow = TRUE)
        }
      }
      return(w)
    }
    # vector fallback
    matrix(as.numeric(w_vec_or_mat), nrow = hidden_dim, ncol = p, byrow = TRUE)
  }

  # ---- 1) Long table: feature importances per group & repeat ----
  weights_df <- purrr::map_dfr(seq_along(weights_list), function(i) {
    purrr::map_dfr(seq_len(G), function(g) {
      w_mat <- .to_hidden_by_p(weights_list[[i]][[g]], hidden_dim, length(feature_names))
      imp <- apply(w_mat, 2, function(col) mean(abs(col)))
      tibble::tibble(
        feature    = feature_names,
        importance = as.numeric(imp),
        group      = as.character(g - 1L),  # codes are 0..G-1
        iteration  = i
      )
    })
  })

  # Map group codes -> human labels (character), then factor with stable order
  weights_df$group_label <- unname(gm_norm[weights_df$group])
  # ensure character, no list-columns
  weights_df$group_label <- as.character(ifelse(is.na(weights_df$group_label),
                                                weights_df$group, weights_df$group_label))
  level_order <- unique(weights_df$group_label)
  weights_df$group_label <- factor(weights_df$group_label, levels = level_order)

  # ---- 2) Per-feature mean importances by group (across repeats) ----
  means_by_group <- weights_df %>%
    dplyr::group_by(.data$group_label, .data$feature) %>%
    dplyr::summarise(avg_importance = mean(.data$importance, na.rm = TRUE), .groups = "drop")

  means_by_group_wide <- means_by_group %>%
    tidyr::pivot_wider(names_from = .data$group_label, values_from = .data$avg_importance)

  # ---- 3) Pairwise (B - A) difference tables for all pairs (descriptive) ----
  group_levels <- as.character(levels(weights_df$group_label))
  group_pairs  <- if (length(group_levels) >= 2) utils::combn(group_levels, 2, simplify = FALSE) else list()
  pairwise_results <- list()
  if (length(group_pairs)) {
    for (pair in group_pairs) {
      A <- pair[1]; B <- pair[2]
      diff_tbl <- means_by_group %>%
        dplyr::filter(.data$group_label %in% c(A, B)) %>%
        tidyr::pivot_wider(names_from = .data$group_label, values_from = .data$avg_importance) %>%
        dplyr::mutate(difference = .data[[B]] - .data[[A]]) %>%
        dplyr::arrange(dplyr::desc(abs(.data$difference)))
      pairwise_results[[paste0(B, "_vs_", A)]] <- diff_tbl
    }
  }

  # ---- 4) Visuals ----
  difference_plot <- NULL
  multi_group_plot <- NULL
  top_features_multi <- NULL

  if (length(group_levels) == 2) {
    # Binary: signed-difference plot (B - A)
    A <- group_levels[1]; B <- group_levels[2]
    diff_table <- pairwise_results[[paste0(B, "_vs_", A)]]
    top_features_data <- diff_table %>%
      dplyr::slice_max(order_by = abs(.data$difference), n = top_n_features)

    difference_plot <- ggplot(top_features_data,
                              aes(x = .data$difference, y = reorder(.data$feature, .data$difference))) +
      geom_col(aes(fill = .data$difference > 0), show.legend = FALSE) +
      scale_fill_manual(values = c("TRUE" = "#F8766D", "FALSE" = "#00BFC4")) +
      labs(
        title = "Top Features by Expert Weight Difference",
        subtitle = paste("Positive = more important in", B, "expert"),
        x = paste0("Importance Difference (", B, " - ", A, ")"),
        y = "Feature"
      ) +
      theme_minimal()

  } else if (length(group_levels) > 2) {
    # Multi-group: rank features by spread = max(mean) - min(mean)
    spread_tbl <- means_by_group_wide %>%
      dplyr::mutate(
        spread = apply(dplyr::across(-.data$feature), 1,
                       function(r) max(r, na.rm = TRUE) - min(r, na.rm = TRUE))
      ) %>%
      dplyr::arrange(dplyr::desc(.data$spread))

    top_multi <- spread_tbl %>%
      dplyr::slice_head(n = top_n_features) %>%
      tidyr::pivot_longer(cols = -c(.data$feature, .data$spread),
                          names_to = "group_label", values_to = "avg_importance") %>%
      dplyr::mutate(feature = factor(.data$feature, levels = rev(unique(.data$feature))))

    top_features_multi <- top_multi

    multi_group_plot <- ggplot(top_multi,
                               aes(x = .data$avg_importance, y = .data$feature, fill = .data$group_label)) +
      geom_col(position = "dodge") +
      labs(
        title    = "Top Features by Max-Min Spread Across Subgroups",
        subtitle = "Non-inferential: mean expert-input importance per group",
        x = "Mean importance (|weights|, averaged over hidden units & repeats)",
        y = "Feature", fill = "Group"
      ) +
      theme_minimal()
  }

  if (verbose) message("Feature importance summaries computed.")

  list(
    all_weights             = weights_df,
    means_by_group_wide     = means_by_group_wide,
    pairwise_differences    = pairwise_results,
    difference_plot         = difference_plot,
    multi_group_plot        = multi_group_plot,
    top_features_multi      = top_features_multi
  )
}
