# Simulated toy data shared by the tests and by fixtures/make_toy_reference.R.
toy_data <- function(n = 400L, family = "binomial", seed = 11L) {
  old <- if (exists(".Random.seed", envir = globalenv())) get(".Random.seed", envir = globalenv()) else NULL
  on.exit(if (!is.null(old)) assign(".Random.seed", old, envir = globalenv()))
  set.seed(seed)
  g <- sample(0:2, n, replace = TRUE, prob = c(0.5, 0.3, 0.2))
  X <- cbind(x1 = rnorm(n) + 0.5 * g, x2 = rnorm(n), x3 = rnorm(n), x4 = rbinom(n, 1, 0.4))
  eta <- 0.8 * X[, 1] - 0.6 * X[, 2] + 0.4 * X[, 3] * X[, 4]
  y <- if (family == "binomial") rbinom(n, 1, stats::plogis(eta)) else eta + rnorm(n)
  tr <- seq_len(round(0.8 * n)); te <- setdiff(seq_len(n), tr)
  # a deterministic stand-in for cross-fitted base scores and gate signal
  lin <- 0.7 * X[, 1] - 0.5 * X[, 2]
  base <- if (family == "binomial") stats::plogis(lin) else lin
  sig <- 0.1 * X[, 3] - 0.05 * X[, 1]
  list(pd = list(X = X, y = y, group = g, feature_names = colnames(X), subject_ids = seq_len(n)),
       splits = list(list(train = tr, test = te)),
       base = list(list(train = base[tr], test = base[te])),
       signal = list(list(train = sig[tr], test = sig[te])),
       family = family)
}

toy_cases <- function() {
  list(
    bin_residual_thr = list(family = "binomial", coupling = "residual", penalty = "thresholded",
                            lambda = 4, lambda_thr = 0, cost = 0.1, reps = 2L),
    bin_expert_re = list(family = "binomial", coupling = "expert", penalty = "random_effects",
                         lambda = 16, lambda_thr = 4, cost = 0.02, reps = 0L),
    cont_residual_var = list(family = "gaussian", coupling = "residual", penalty = "variance",
                             lambda = 4, lambda_thr = 16, cost = 0, reps = 2L)
  )
}

fit_toy_case <- function(cs) {
  td <- toy_data(family = cs$family)
  fairGATE::train_gnn(td$pd, run_tuning = FALSE,
    best_params = list(lr = 0.002, hidden_dim = 16L, dropout_rate = 0.1, lambda = cs$lambda, temperature = 1),
    epochs = 15L, seed = 4242L, splits = td$splits, family = cs$family,
    n_strata = if (cs$family == "binomial") 2L else 5L, standardise = TRUE, impute = "median",
    penalty = cs$penalty, alpha = 0, early_stopping = TRUE, es_prop = 0.15, es_patience = 30L,
    es_min_epochs = 10L, es_smooth = 5L, base_score = td$base, coupling = cs$coupling,
    gate = "softmax", intervention_cost = cs$cost, lambda_thr = cs$lambda_thr, thr_temp = 0.05,
    thr_target = "eo", thr_agg = "max", reference_group = 0L, balance = 0, adversary = 0,
    gate_signal = td$signal, interval_reps = cs$reps, interval_epochs = 2L)
}

torch_ready <- function() {
  requireNamespace("torch", quietly = TRUE) &&
    isTRUE(tryCatch(torch::torch_is_installed(), error = function(e) FALSE))
}
skip_if_no_torch <- function() {
  testthat::skip_if_not(torch_ready(), "torch with LibTorch is not available")
  torch::torch_set_num_threads(1L)
}
skip_if_no_xgboost <- function() {
  testthat::skip_if_not_installed("xgboost", minimum_version = "3.1.0")
}
