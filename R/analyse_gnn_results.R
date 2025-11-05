#' Analyse and Visualise GNN Results
#'
#' Generates ROC/Calibration outputs and subgroup gate analyses.
#'
#' @param gnn_results List from `train_gnn()` (expects $final_results, $performance_summary, $gate_weights).
#' @param prepared_data List from `prepare_data()` (used to retrieve group mappings if not provided).
#' @param group_mappings Optional named mapping from group codes (names) to labels (values).
#' @param create_roc_plot Logical; return ROC ggplot (default TRUE).
#' @param create_calibration_plot Logical; return calibration ggplot (default TRUE).
#' @param analyse_gate_weights Logical; analyse gate weights across groups (default TRUE).
#' @param analyse_gate_entropy Logical; analyse gate entropy across groups (default TRUE).
#' @param verbose Logical; print progress messages.
#' @param nonparametric Logical; if TRUE and k>2, use Kruskal-Wallis + Wilcoxon instead of ANOVA + Tukey.
#'
#' @return A list of ggplots, test results, and summary tables.
#' @export
#' @import ggplot2
#' @import magrittr
#' @importFrom pROC roc auc
#' @importFrom dplyr mutate group_by summarise filter n left_join across bind_rows
#' @importFrom tidyselect all_of
#' @importFrom stats t.test aov TukeyHSD kruskal.test pairwise.wilcox.test sd model.frame
#' @importFrom rlang .data
analyse_gnn_results <- function(gnn_results,
                                prepared_data,
                                group_mappings = NULL,
                                create_roc_plot = TRUE,
                                create_calibration_plot = TRUE,
                                analyse_gate_weights = TRUE,
                                analyse_gate_entropy = TRUE,
                                verbose = FALSE,
                                nonparametric = FALSE) {

  # ---- Validate inputs ----
  required <- c("final_results", "performance_summary", "gate_weights")
  if (!all(required %in% names(gnn_results))) {
    stop("gnn_results object is missing required components: ",
         paste(setdiff(required, names(gnn_results)), collapse = ", "))
  }

  out <- list()
  results_all <- gnn_results$final_results
  perf_tbl    <- gnn_results$performance_summary
  gate_data   <- gnn_results$gate_weights

  # ---- Try to pull group_mappings automatically ----
  if (is.null(group_mappings) && !is.null(prepared_data)) {
    group_mappings <- attr(prepared_data, "group_mappings")
    if (verbose && !is.null(group_mappings)) {
      message("Using group_mappings retrieved from prepared_data attributes.")
    }
  }

  # ---- Normalise group_mappings so that names = codes (as character), values = labels ----
  # Accepts either codes->labels (names numeric/values character) or labels->codes (names character/values numeric).
  .normalise_group_mappings <- function(gm, observed_codes) {
    if (is.null(gm) || !length(gm)) return(NULL)

    nm <- names(gm)
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
      # If duplicate codes map to multiple labels, keep first occurrence
      m <- !duplicated(codes)
      labels <- labels[m]; codes <- codes[m]
      names(labels) <- codes
      return(labels)
    }

    # Fallback: build mapping from observed codes with string labels "Group <code>"
    oc <- sort(unique(as.character(observed_codes)))
    lbl <- paste0("Group ", oc)
    names(lbl) <- oc
    lbl
  }

  # Build group_label in gate_data with stable levels carried through
  observed_codes <- gate_data$group
  gm_norm <- .normalise_group_mappings(group_mappings, observed_codes)

  if (!is.null(gm_norm)) {
    # map using codes as character
    keys <- as.character(observed_codes)
    labs <- unname(gm_norm[keys])
    labs[is.na(labs)] <- keys[is.na(labs)]  # fallback pass-through
    # Set levels in order of sorted unique observed codes to keep label order stable
    level_order <- unname(gm_norm[sort(unique(as.character(observed_codes)))])
    level_order[is.na(level_order)] <- sort(unique(as.character(observed_codes)))[is.na(level_order)]
    gate_data$group_label <- factor(labs, levels = unique(level_order))
  } else {
    # No mapping: make a label from the code itself
    gate_data$group_label <- factor(as.character(observed_codes),
                                    levels = sort(unique(as.character(observed_codes))))
  }

  # ---- Extract AUC & Brier ----
  get_metric <- function(tbl, name) {
    if (!all(c("Metric", "Value") %in% names(tbl))) return(NA_real_)
    v <- tbl$Value[tbl$Metric == name]
    if (length(v)) as.numeric(v[1]) else NA_real_
  }
  auc_gated   <- get_metric(perf_tbl, "AUC")
  brier_gated <- get_metric(perf_tbl, "Brier Score")

  out$metrics_table <- perf_tbl
  out$auc   <- auc_gated
  out$brier <- brier_gated

  # ---- Helper: two-level vs multi-level tests (NA- and level-safe) ----
  .two_or_multi_test <- function(formula, data, pairwise = TRUE, nonparametric = FALSE) {
    mf <- stats::model.frame(formula, data = data, na.action = stats::na.omit)
    if (!nrow(mf)) {
      return(list(method = "No data after NA removal", omnibus = NA, pairwise = NULL))
    }
    g <- factor(mf[[2]])
    if (nlevels(g) < 2L) {
      return(list(method = "Insufficient groups after NA filtering", omnibus = NA, pairwise = NULL))
    }
    if (nlevels(g) == 2L) {
      return(list(method = "Welch two-sample t-test (2 groups)",
                  omnibus = stats::t.test(formula, data = mf),
                  pairwise = NULL))
    }
    if (isTRUE(nonparametric)) {
      return(list(method = "Kruskal-Wallis (multi-group) + pairwise Wilcoxon (BH)",
                  omnibus = stats::kruskal.test(formula, mf),
                  pairwise = stats::pairwise.wilcox.test(mf[[1]], g, p.adjust.method = "BH")))
    }
    fit <- stats::aov(formula, data = mf)
    list(method = "One-way ANOVA (multi-group) + Tukey HSD",
         omnibus = summary(fit),
         pairwise = stats::TukeyHSD(fit))
  }

  # Early guard: if overall only one subgroup, skip comparative analyses but still produce ROC/Calibration
  if (nlevels(gate_data$group_label) < 2L) {
    if (verbose) message("Only one subgroup present overall; skipping gate-weight and entropy comparisons.")
    analyse_gate_weights <- FALSE
    analyse_gate_entropy <- FALSE
  }

  # ---- 1) ROC ----
  if (create_roc_plot || isTRUE(TRUE)) {
    if (verbose) message("Generating ROC artefacts...")
    roc_obj <- pROC::roc(response = results_all$true,
                         predictor = results_all$prob,
                         quiet = TRUE, levels = c(0, 1))
    roc_df <- data.frame(
      fpr = 1 - roc_obj$specificities,
      tpr = roc_obj$sensitivities,
      threshold = roc_obj$thresholds
    )
    out$roc_curve <- roc_df

    if (create_roc_plot) {
      out$roc_plot <- ggplot(roc_df, aes(x = .data$fpr, y = .data$tpr)) +
        geom_path(linewidth = 1) +
        geom_abline(linetype = "dashed") +
        labs(
          title = "ROC Curve",
          subtitle = paste("AUC =", round(auc_gated, 3)),
          x = "False Positive Rate (1 - Specificity)",
          y = "True Positive Rate (Sensitivity)"
        ) + theme_minimal()
    }
  }

  # ---- 2) Calibration ----
  if (create_calibration_plot || isTRUE(TRUE)) {
    if (verbose) message("Generating Calibration artefacts...")
    calib_tbl <- results_all %>%
      dplyr::mutate(prob_bin = cut(prob, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)) %>%
      dplyr::group_by(prob_bin) %>%
      dplyr::summarise(
        mean_predicted_prob = mean(prob),
        observed_proportion = mean(true),
        .groups = "drop"
      ) %>%
      dplyr::filter(!is.na(prob_bin))
    out$calibration_table <- calib_tbl

    if (create_calibration_plot) {
      out$calibration_plot <- ggplot(calib_tbl, aes(x = .data$mean_predicted_prob,
                                                    y = .data$observed_proportion)) +
        geom_abline(linetype = "dashed") +
        geom_line(linewidth = 1) +
        geom_point(size = 2) +
        coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
        labs(
          title = "Calibration Plot",
          subtitle = paste("Brier Score =", round(brier_gated, 3)),
          x = "Mean Predicted Probability",
          y = "Observed Proportion of Remission"
        ) + theme_minimal()
    }
  }

  # ---- 3) Gate weight analysis ----
  if (analyse_gate_weights) {
    if (verbose) message("Analysing gate weights across groups...")
    expert_cols <- grep("^gate_prob_expert_", names(gate_data), value = TRUE)

    # Keep only experts with enough finite data to compare (avoids degenerate contrasts)
    expert_cols <- Filter(function(col) {
      v <- gate_data[[col]]
      sum(is.finite(v)) >= 10  # arbitrary but protective
    }, expert_cols)

    if (!length(expert_cols)) {
      warning("No suitable gate_prob_expert_* columns with enough data; skipping gate weight analysis.")
    } else {
      e1 <- expert_cols[1]
      out$gate_density_plot <- ggplot(gate_data, aes(x = .data[[e1]], fill = .data$group_label)) +
        geom_density(alpha = 0.7) +
        labs(
          title = "Gate Weight Distribution by Subgroup",
          subtitle = sprintf("Routing preference for '%s'", e1),
          x = sprintf("Gate Weight for '%s'", e1),
          y = "Density",
          fill = "Group"
        ) + theme_minimal()

      tests <- lapply(expert_cols, function(col) {
        .two_or_multi_test(stats::as.formula(paste0(col, " ~ group_label")),
                           gate_data, pairwise = TRUE, nonparametric = nonparametric)
      })
      names(tests) <- expert_cols
      out$gate_weight_tests <- tests
    }
  }

  # ---- 4) Gate entropy ----
  if (analyse_gate_entropy) {
    if (verbose) message("Analysing gate entropy across groups...")
    if (!"gate_entropy" %in% names(gate_data)) {
      warning("gate_entropy not found in gate_weights; skipping entropy analysis.")
    } else {
      out$entropy_density_plot <- ggplot(gate_data, aes(x = .data$gate_entropy,
                                                        fill = .data$group_label)) +
        geom_density(alpha = 0.7) +
        labs(
          title = "Gate Entropy Distribution by Subgroup",
          subtitle = "Entropy quantifies routing uncertainty (0 = decisive)",
          x = "Gate Entropy", y = "Density", fill = "Group"
        ) + theme_minimal()

      out$gate_entropy_test <- .two_or_multi_test(
        gate_entropy ~ group_label, gate_data, pairwise = TRUE, nonparametric = nonparametric
      )
    }
  }

  # Carry labels through in outputs for downstream consumers
  out$group_levels <- levels(gate_data$group_label)

  if (verbose) message("Analysis complete.")
  out
}
