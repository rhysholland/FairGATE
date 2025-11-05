#' Train and Evaluate the Gated Neural Network (robust splits + safe ROC)
#'
#' Trains a subgroup-aware gated neural network with a fairness-constrained loss,
#' optionally performs hyperparameter tuning with lightweight budgets, and returns
#' predictions, gate/expert weights, and summary metrics. Designed to be CRAN-safe:
#' - No background installs, no saving unless requested
#' - CPU-only torch, capped threads to avoid oversubscription
#'
#' @param prepared_data List from `prepare_data()` containing:
#'   \itemize{
#'     \item `X` (matrix/data.frame of numeric features)
#'     \item `y` (numeric 0/1)
#'     \item `group` (numeric codes for sensitive subgroup)
#'     \item `feature_names` (character vector; optional)
#'     \item `subject_ids` (vector; optional)
#'   }
#' @param hyper_grid data.frame with columns: `lr`, `hidden_dim`, `dropout_rate`, `lambda`, `temperature`.
#' @param num_repeats Integer (>=1). Repeated train/test splits for the **final** model (and for tuning if `tune_repeats` is not set).
#' @param epochs Integer (>=1). Training epochs per run for the **final** model (and for tuning if `tune_epochs` is not set).
#' @param output_dir Directory to write csv/rds if `save_outputs = TRUE`. Defaults to `tempdir()`.
#' @param run_tuning Logical. If `TRUE`, runs a grid search using `hyper_grid` and picks best by mean AUC.
#' @param best_params data.frame/list with `lr`, `hidden_dim`, `dropout_rate`, `lambda`, `temperature` if `run_tuning = FALSE`.
#' @param save_outputs Logical. If `TRUE`, writes CSV/RDS outputs to `output_dir`. Default `FALSE`.
#' @param seed Optional integer seed to make data splits reproducible. If `NULL`, current RNG state is respected.
#' @param verbose Logical. Print progress messages. Default `FALSE`.
#' @param tune_repeats Integer (>=1). Repeats per combo **during tuning only**. Defaults to `min(5, num_repeats)`.
#' @param tune_epochs Integer (>=1). Epochs per run **during tuning only**. Defaults to `min(epochs, 100)`.
#'
#' @return A list with:
#'   \itemize{
#'     \item `final_results` (tibble: subjectid, true, prob, group, iteration)
#'     \item `gate_weights` (tibble with gate probabilities & entropy per subject/iteration)
#'     \item `expert_weights` (list of expert input-layer weight matrices per repeat)
#'     \item `performance_summary` (tibble with AUC and Brier)
#'     \item `aif360_data` (tibble for fairness metric tooling)
#'     \item `tuning_results` (tibble or message when tuning skipped)
#'   }
#'
#' @export
#' @importFrom pROC roc auc
#' @importFrom dplyr bind_rows slice_max select
#' @importFrom readr write_csv
#' @importFrom tibble tibble as_tibble
#' @importFrom utils capture.output tail flush.console
train_gnn <- function(prepared_data,
                      hyper_grid,
                      num_repeats = 20,
                      epochs = 300,
                      output_dir = tempdir(),
                      run_tuning = TRUE,
                      best_params = NULL,
                      save_outputs = FALSE,
                      seed = NULL,
                      verbose = FALSE,
                      tune_repeats = NULL,
                      tune_epochs  = NULL) {

  # -------- Torch availability (runtime-only; CRAN-safe) --------
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("The 'torch' package is required but not installed. Install it, then run torch::install_torch() outside CRAN checks.")
  }

  # Cap threads to avoid deadlocks / oversubscription (esp. on macOS / CI)
  try({
    torch::torch_set_num_threads(1L)
    Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
  }, silent = TRUE)

  # create output dir only if user explicitly requests saving
  if (isTRUE(save_outputs) && !dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  # -------- Seed handling (respect current RNG unless user sets one) --------
  if (!is.null(seed)) {
    if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed)) {
      stop("`seed` must be a single finite numeric value or NULL.")
    }
    seed <- as.integer(seed)
    if (is.na(seed)) stop("`seed` must be coercible to an integer.")
  }
  base_seed <- seed
  seed_counter <- -1L
  next_split_seed <- function() {
    if (is.null(base_seed)) return(NULL)
    seed_counter <<- seed_counter + 1L
    base_seed + seed_counter
  }

  # -------- Unpack & validate --------
  X <- prepared_data$X
  y <- as.numeric(prepared_data$y)        # expected 0/1
  group <- as.numeric(prepared_data$group)
  feature_names <- prepared_data$feature_names
  subject_ids <- prepared_data$subject_ids

  if (!is.matrix(X) && !is.data.frame(X)) stop("prepared_data$X must be a matrix or data.frame.")
  X <- as.matrix(X)
  if (!all(is.finite(X))) stop("prepared_data$X contains non-finite values. Please clean or impute.")
  if (any(!y %in% c(0, 1))) stop("prepared_data$y must be binary 0/1.")
  if (length(y) != nrow(X)) stop("Length of y must equal nrow(X).")
  if (length(group) != nrow(X)) stop("Length of group must equal nrow(X).")

  # Drop rows with NA across X/y/group to avoid downstream NA headaches
  if (anyNA(X) || anyNA(y) || anyNA(group)) {
    keep <- complete.cases(X) & !is.na(y) & !is.na(group)
    if (isTRUE(verbose)) message(sprintf("[prepare] Dropped %d rows with NA in X/y/group.", sum(!keep)))
    X <- X[keep, , drop = FALSE]
    y <- y[keep]
    group <- group[keep]
    if (!is.null(subject_ids)) subject_ids <- subject_ids[keep]
  }

  if (length(unique(y)) < 2L) stop("Only one class present in y; cannot train a classifier.")
  n <- nrow(X); input_dim <- ncol(X)

  # -------- Stratified split ensuring both classes in TEST --------
  stratified_split_both_in_test <- function(y_vec, test_prop = 0.20, seed_local = NULL) {
    if (!is.null(seed_local)) {
      seed_local <- as.integer(seed_local)
      if (is.na(seed_local)) stop("seed must be coercible to integer.")
      restore_rng <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        old_seed <- get(".Random.seed", envir = .GlobalEnv)
        function() assign(".Random.seed", old_seed, envir = .GlobalEnv)
      } else {
        function() if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv)
      }
      on.exit(restore_rng(), add = TRUE)
      set.seed(seed_local)
    }
    idx0 <- which(y_vec == 0); idx1 <- which(y_vec == 1)
    n0_test <- max(1L, floor(length(idx0) * test_prop))
    n1_test <- max(1L, floor(length(idx1) * test_prop))
    test <- sort(c(sample(idx0, n0_test), sample(idx1, n1_test)))
    train <- setdiff(seq_along(y_vec), test)
    list(train = train, test = test)
  }

  # -------- Safe AUC wrapper (guards for degenerate cases) --------
  safe_auc <- function(y_true, probs) {
    keep <- is.finite(probs) & !is.na(y_true)
    y_ <- y_true[keep]; p_ <- probs[keep]
    if (length(y_) == 0L) return(NA_real_)
    if (length(unique(y_)) < 2L) return(NA_real_)
    if (length(unique(round(p_, 6))) <= 1L) return(NA_real_)
    ro <- pROC::roc(response = y_, predictor = p_, quiet = TRUE, levels = c(0, 1))
    as.numeric(pROC::auc(ro))
  }

  # -------- Model definition --------
  gated_model <- torch::nn_module(
    "GatedModel",
    initialize = function(input_dim, hidden_dim, num_groups, dropout_p = 0.5) {
      self$gate_layer <- torch::nn_linear(input_dim, num_groups)
      self$subgroup_layers <- torch::nn_module_list(lapply(seq_len(num_groups), function(i) {
        torch::nn_sequential(
          torch::nn_linear(input_dim, hidden_dim),
          torch::nn_batch_norm1d(hidden_dim),
          torch::nn_relu(),
          torch::nn_dropout(p = dropout_p),
          torch::nn_linear(hidden_dim, hidden_dim),
          torch::nn_relu(),
          torch::nn_linear(hidden_dim, 1)
        )
      }))
    },
    forward = function(x, temperature_val = 0.5) {
      gate_weights    <- torch::nnf_softmax(self$gate_layer(x) / temperature_val, dim = 2)
      subgroup_logits <- lapply(self$subgroup_layers, function(layer) layer(x)$squeeze(2))
      subgroup_logits <- torch::torch_stack(subgroup_logits, dim = 2)
      output_logits   <- torch::torch_sum(gate_weights * subgroup_logits, 2)
      attr(output_logits, "gate_weights") <- gate_weights
      output_logits
    }
  )

  # -------- Fairness-constrained loss (robust to missing groups) --------
  fairness_constrained_loss <- function(y_logits, y_true, groups, lambda = 1.0) {
    bce <- torch::nnf_binary_cross_entropy_with_logits(y_logits, y_true)
    if (lambda == 0) return(bce)

    y_prob <- torch::torch_sigmoid(y_logits)
    ug <- sort(unique(as.integer(as.array(groups))))
    soft_tprs <- list(); soft_fprs <- list()
    for (g in ug) {
      m <- (groups == g); if (torch::torch_sum(m)$item() == 0) next
      yp <- y_prob[m]; yt <- y_true[m]
      p <- torch::torch_sum(yt); n <- torch::torch_sum(1 - yt)
      if (p$item() > 0 && n$item() > 0) {
        soft_tprs[[as.character(g)]] <- torch::torch_sum(yp * yt) / (p + 1e-7)
        soft_fprs[[as.character(g)]] <- torch::torch_sum(yp * (1 - yt)) / (n + 1e-7)
      }
    }
    if (length(soft_tprs) < 2L || length(soft_fprs) < 2L) return(bce)
    tpr_v <- torch::torch_var(torch::torch_stack(soft_tprs))
    fpr_v <- torch::torch_var(torch::torch_stack(soft_fprs))
    bce + lambda * (tpr_v + fpr_v)
  }

  # -------- Prepare tensors (CPU; no data loaders to keep CRAN-safe) --------
  num_groups <- length(unique(group))
  x_all   <- torch::torch_tensor(X, dtype = torch::torch_float())
  y_all_t <- torch::torch_tensor(as.numeric(y), dtype = torch::torch_float())
  g_all_t <- torch::torch_tensor(as.integer(group), dtype = torch::torch_long())

  # -------- Tuning budgets (key change to keep tuning fast) --------
  if (is.null(tune_repeats)) tune_repeats <- max(1L, min(5L, as.integer(num_repeats)))
  if (is.null(tune_epochs))  tune_epochs  <- max(1L, min(100L, as.integer(epochs)))

  # -------- Hyperparameter tuning (optional) --------
  tuning_results <- tibble::tibble()
  if (isTRUE(run_tuning)) {
    if (missing(hyper_grid) || is.null(hyper_grid) || nrow(hyper_grid) == 0L) {
      stop("run_tuning=TRUE but `hyper_grid` is missing/empty.")
    }
    need_cols <- c("lr", "hidden_dim", "dropout_rate", "lambda", "temperature")
    if (!all(need_cols %in% names(hyper_grid))) {
      stop("`hyper_grid` must include columns: lr, hidden_dim, dropout_rate, lambda, temperature")
    }
    if (isTRUE(verbose)) message("Starting hyperparameter tuning with ", nrow(hyper_grid), " combinations...")

    for (j in seq_len(nrow(hyper_grid))) {
      # robust scalar extraction (in case expand.grid gave factors)
      p_lr   <- as.numeric(hyper_grid[j, "lr", drop = TRUE])
      p_hid  <- as.integer(hyper_grid[j, "hidden_dim", drop = TRUE])
      p_drop <- as.numeric(hyper_grid[j, "dropout_rate", drop = TRUE])
      p_lam  <- as.numeric(hyper_grid[j, "lambda", drop = TRUE])
      p_temp <- as.numeric(hyper_grid[j, "temperature", drop = TRUE])

      if (isTRUE(verbose)) {
        message(sprintf("Combo %d/%d: lr=%.4g | hidden=%d | dropout=%.2f | lambda=%.2f | T=%.2f | reps=%d | epochs=%d",
                        j, nrow(hyper_grid), p_lr, p_hid, p_drop, p_lam, p_temp, tune_repeats, tune_epochs))
      }

      aucs <- numeric(0)
      for (r in seq_len(as.integer(tune_repeats))) {
        if (interactive()) utils::flush.console()
        split_seed <- next_split_seed()
        if (!is.null(split_seed)) try(torch::torch_manual_seed(split_seed), silent = TRUE)

        sp <- stratified_split_both_in_test(y, test_prop = 0.20, seed_local = split_seed)
        tr <- sp$train; te <- sp$test

        xtr <- x_all[tr, , drop = FALSE]
        ytr <- y_all_t[tr]
        gtr <- g_all_t[tr]
        xte <- x_all[te, , drop = FALSE]
        yte <- y[te]

        net <- gated_model(input_dim, p_hid, num_groups, dropout_p = p_drop)
        opt <- torch::optim_adam(net$parameters, lr = p_lr)

        for (ep in seq_len(as.integer(tune_epochs))) {
          net$train(); opt$zero_grad()
          logits <- net(xtr, temperature_val = p_temp)
          ytr_use <- if (length(logits$size()) == 1L) ytr else ytr$unsqueeze(2)
          loss <- fairness_constrained_loss(logits, ytr_use, gtr, lambda = p_lam)
          if (!is.finite(as.numeric(loss$item()))) {  # guard against NaN
            if (isTRUE(verbose)) message("  Loss became non-finite; aborting this repeat.")
            break
          }
          loss$backward(); opt$step()
        }

        net$eval()
        probs <- torch::with_no_grad({
          as.numeric(torch::torch_sigmoid(net(xte, temperature_val = p_temp)))
        })
        probs[!is.finite(probs)] <- NA_real_
        probs <- pmin(pmax(probs, 1e-6), 1 - 1e-6)

        aucs <- c(aucs, safe_auc(yte, probs))
      }

      mean_auc <- mean(aucs, na.rm = TRUE)
      tuning_results <- dplyr::bind_rows(
        tuning_results,
        tibble::as_tibble(list(
          lr = p_lr, hidden_dim = p_hid, dropout_rate = p_drop, lambda = p_lam, temperature = p_temp,
          mean_auc = mean_auc
        ))
      )
      if (isTRUE(verbose)) message("Mean AUC: ", ifelse(is.finite(mean_auc), sprintf("%.4f", mean_auc), "NA"))
    }

    best_params <- tuning_results %>% dplyr::slice_max(order_by = mean_auc, n = 1)
    if (isTRUE(verbose)) {
      message("Best Performing Combination Found")
      utils::capture.output(best_params) |> paste(collapse = "\n") |> message()
    }
  } else {
    if (is.null(best_params)) stop("If run_tuning = FALSE, `best_params` must be provided.")
    best_params <- as.data.frame(best_params)
    need_cols <- c("lr", "hidden_dim", "dropout_rate", "lambda", "temperature")
    if (!all(need_cols %in% names(best_params))) {
      stop("`best_params` must include: lr, hidden_dim, dropout_rate, lambda, temperature")
    }
    if (isTRUE(verbose)) {
      message("Skipping tuning. Using provided parameters.")
      utils::capture.output(best_params[1, need_cols]) |> paste(collapse = "\n") |> message()
    }
  }

  # -------- Final run with best params (uses full budgets) --------
  bp <- best_params
  final_lr          <- as.numeric(bp$lr[1])
  final_hidden_dim  <- as.integer(bp$hidden_dim[1])
  final_dropout     <- as.numeric(bp$dropout_rate[1])
  final_lambda      <- as.numeric(bp$lambda[1])
  final_temperature <- as.numeric(bp$temperature[1])

  results_list <- list()
  gate_weights_list <- list()
  expert_weights_list <- list()

  for (r in seq_len(as.integer(num_repeats))) {
    split_seed <- next_split_seed()
    if (!is.null(split_seed)) try(torch::torch_manual_seed(split_seed), silent = TRUE)

    sp <- stratified_split_both_in_test(y, test_prop = 0.20, seed_local = split_seed)
    tr <- sp$train; te <- sp$test

    xtr <- x_all[tr, , drop = FALSE]
    ytr <- y_all_t[tr]
    gtr <- g_all_t[tr]

    xte <- x_all[te, , drop = FALSE]
    yte <- y[te]
    gte <- group[te]
    subj_te <- if (!is.null(subject_ids)) subject_ids[te] else seq_along(yte)

    net <- gated_model(input_dim, final_hidden_dim, num_groups, dropout_p = final_dropout)
    opt <- torch::optim_adam(net$parameters, lr = final_lr)

    for (ep in seq_len(as.integer(epochs))) {
      net$train(); opt$zero_grad()
      logits <- net(xtr, temperature_val = final_temperature)
      ytr_use <- if (length(logits$size()) == 1L) ytr else ytr$unsqueeze(2)
      loss <- fairness_constrained_loss(logits, ytr_use, gtr, lambda = final_lambda)
      if (!is.finite(as.numeric(loss$item()))) break
      loss$backward(); opt$step()
    }

    net$eval()
    logits_te <- torch::with_no_grad({ net(xte, temperature_val = final_temperature) })
    probs <- as.numeric(torch::torch_sigmoid(logits_te))
    probs[!is.finite(probs)] <- NA_real_
    probs <- pmin(pmax(probs, 1e-6), 1 - 1e-6)

    # Gate weights (via attribute set in forward)
    gw <- attr(logits_te, "gate_weights")
    gw_mat <- torch::as_array(gw)
    gw_clamped <- pmin(pmax(gw_mat, 1e-10), 1 - 1e-10)
    gate_entropy <- as.numeric(-rowSums(gw_clamped * log(gw_clamped)))

    # Expert first-layer weights (input_dim x hidden), per expert
    expert_weights_list[[r]] <- lapply(
      net$subgroup_layers,
      function(layer) as.matrix(torch::as_array(layer[[1]]$weight))
    )

    gate_probs_df <- as.data.frame(gw_mat)
    names(gate_probs_df) <- paste0("gate_prob_expert_", seq_len(ncol(gate_probs_df)) - 1L)

    results_list[[r]] <- tibble::tibble(
      iteration = r,
      subjectid = subj_te,
      true = yte,
      prob = probs,
      group = gte
    )
    gate_weights_list[[r]] <- dplyr::bind_cols(
      tibble::tibble(iteration = r, subjectid = subj_te, group = gte, gate_entropy = gate_entropy),
      gate_probs_df
    )
    if (isTRUE(verbose)) message(sprintf("[final] repeat %d/%d complete", r, num_repeats))
    if (interactive()) utils::flush.console()
  }

  final_results <- dplyr::bind_rows(results_list)
  gate_weights  <- dplyr::bind_rows(gate_weights_list)

  auc_final <- if (length(unique(final_results$true)) >= 2L) {
    safe_auc(final_results$true, final_results$prob)
  } else NA_real_
  brier_final <- mean((final_results$true - final_results$prob)^2, na.rm = TRUE)

  performance_summary <- tibble::tibble(
    Metric = c("AUC", "Brier Score"),
    Value  = c(auc_final, brier_final)
  )

  aif360_data <- final_results %>%
    dplyr::select(subjectid, true_label = true, predicted_prob = prob, sensitive_attr_numeric = group)

  if (isTRUE(save_outputs)) {
    readr::write_csv(final_results, file.path(output_dir, "gnn_final_predictions.csv"))
    readr::write_csv(gate_weights,  file.path(output_dir, "gnn_gate_weights.csv"))
    readr::write_csv(aif360_data,   file.path(output_dir, "gnn_aif360_data.csv"))
    if (isTRUE(run_tuning)) readr::write_csv(tuning_results, file.path(output_dir, "gnn_tuning_results.csv"))
    saveRDS(expert_weights_list, file.path(output_dir, "gnn_expert_weights.rds"))
  }

  list(
    final_results = final_results,
    gate_weights = gate_weights,
    expert_weights = expert_weights_list,
    performance_summary = performance_summary,
    aif360_data = aif360_data,
    tuning_results = if (isTRUE(run_tuning)) tuning_results else "Tuning was skipped"
  )
}
