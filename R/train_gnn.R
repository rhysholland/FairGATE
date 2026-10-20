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
#' @param splits Optional list of pre-computed train/test partitions, each a list
#'   with integer elements `train` and `test` indexing rows of `prepared_data$X`.
#'   When supplied, `num_repeats` is set to `length(splits)` and the internal
#'   stratified sampler is bypassed, so several models can be evaluated on
#'   byte-identical partitions. Tuning reuses the same partitions, truncated to
#'   `tune_repeats`.
#' @param family Either `"binomial"` (default; binary outcome, sigmoid output,
#'   binary cross-entropy, equalised-odds penalty on soft true- and
#'   false-positive rates) or `"gaussian"` (continuous outcome, identity output,
#'   squared-error loss, outcome-stratified conditional-mean penalty). See
#'   `n_strata`.
#' @param n_strata Integer (>=2). Number of outcome strata used by the
#'   `"gaussian"` fairness penalty; strata are equal-frequency bins of `y`
#'   computed within each training split. Ignored when `family = "binomial"`.
#'   With `n_strata = 2` and a binary outcome the penalty is algebraically
#'   identical to the binomial penalty.
#' @param standardise Logical. If `TRUE` (default), columns of `X` are centred
#'   and scaled using the training rows of each split only, and the same
#'   constants are applied to the held-out rows. Columns with zero training
#'   variance are centred but not scaled. Set `FALSE` to use `X` as supplied.
#' @param impute Either `"none"` (default) or `"median"`. With `"median"`,
#'   missing values in `X` are filled with the column median of the training
#'   rows of each split, leaving the held-out rows untouched by their own
#'   medians. With `"none"`, rows containing missing values are dropped as
#'   before.
#' @param penalty Which fairness penalty to minimise: `"random_effects"`
#'   (default), `"bias_corrected"` or `"variance"`. `"variance"` is the penalty
#'   of version 0.2.0 and earlier and reproduces it exactly, including its
#'   requirement that a group be present in every stratum to contribute. The
#'   other two rungs correct that penalty for the sampling noise in the cell
#'   means and treat each stratum independently. See Details and
#'   [fairgate_penalty_from_cells()].
#' @param lambda_max Fairness weight of the most heavily constrained expert in
#'   the frontier-anchored ladder. Defaults to `NULL`, which uses the mixture
#'   fairness weight `lambda` of the selected configuration. Only the ladder
#'   shape depends on it; it has no effect when `alpha = 0`, beyond setting the
#'   scale of the exported correction intensity.
#' @param alpha Weight on the auxiliary per-expert objective. `0` (the default)
#'   reproduces the unanchored model of version 0.2.0 exactly: no auxiliary
#'   term is formed and the experts are trained only through the mixture.
#'   Positive values anchor each expert at its own point on the ladder. See
#'   Details.
#' @param early_stopping Logical. If `TRUE`, a stratified slice of the training
#'   rows of each partition is held out, the validation objective is evaluated
#'   at every epoch, and the parameters of the best epoch are restored before
#'   prediction. This replaces tuning one global epoch budget: `epochs` becomes
#'   a ceiling rather than the budget actually used. Default `FALSE`, which
#'   trains for exactly `epochs` epochs on all training rows as before.
#' @param es_prop Proportion of each partition's training rows held out for
#'   early stopping, stratified jointly on outcome stratum and protected group.
#'   Default `0.2`. Ignored when `early_stopping = FALSE`.
#' @param es_min_epochs Minimum number of epochs before the best epoch may be
#'   selected. Default `1`. Ignored when `early_stopping = FALSE`.
#' @param es_patience Optional integer. When early stopping is enabled, stop
#'   once this many consecutive epochs have passed without the validation
#'   objective improving on its best value, and restore the best epoch.
#'   `NULL` (the default) never stops early, so the loop runs to `epochs` and
#'   the best epoch is restored, exactly as in 0.3.0. The fairGATE protocol of
#'   the accompanying study uses `es_patience = 30` with `epochs = 300`. Ignored when
#'   `early_stopping = FALSE`.
#' @param base_score Optional list aligned with `splits`, one element per
#'   partition, each `list(train = numeric, test = numeric)`: the scores of an
#'   independently fitted base learner for that partition's training rows and
#'   test rows, in the same row order as `splits[[r]]$train` and
#'   `splits[[r]]$test`, on the RESPONSE scale (probabilities for
#'   `family = "binomial"`, the outcome scale for `"gaussian"`). Training-row
#'   scores should be out-of-fold (cross-fitted), so that no training row is
#'   scored by a model that saw it. Requires `splits` and a `coupling` other
#'   than `"none"`. See Details.
#' @param coupling How the base score enters the model: `"none"` (the
#'   default, and the only value allowed when `base_score` is `NULL`) is the
#'   gated mixture of earlier versions; `"offset"` adds the mixture to the base
#'   score on the link scale; `"expert"` makes the base score one component of
#'   the mixture, weighted per individual by the gate. When `base_score` is
#'   supplied, `coupling` must be given explicitly. See Details.
#' @param gate_bias_init Optional numeric vector added to the gate's output
#'   bias at initialisation, one entry per gate output (`K` for `"none"` and
#'   `"offset"`, `K + 1` for `"expert"`, entry 1 being the base component). A
#'   diagnostic device: a very large first entry with `coupling = "expert"`
#'   forces all routing onto the base score, which must then be reproduced.
#'   `NULL` (the default) leaves the initialisation untouched.
#' @param gate `"softmax"` (default, 0.4.0), `"sparsemax"` or
#'   `"entmax15"`; the sparse gates need `coupling = "expert"` or
#'   `"residual"`. See [fairgate_gate_map()].
#' @param intervention_cost Weight `mu` of `mean |eta - b|`, the
#'   link-scale distance from the base learner. Default 0 (off).
#' @param lambda_thr,thr_temp,thr_target,thr_agg Weight,
#'   temperature, per-group target (`"eo"` or `"aod"`) and aggregation
#'   (`"max"` or `"mean"`) of the thresholded error-rate penalty added to the
#'   variance-component penalty. `lambda_thr = 0` (default) is off.
#'   `penalty = "thresholded"` uses that penalty alone, with weight `lambda`.
#' @param reference_group Reference group code for the
#'   thresholded penalty; `NULL` compares each group with the pooled rate.
#' @param gate_signal Optional list aligned with `splits`, each
#'   `list(train =, test =)` of numeric vectors or matrices: cross-fitted
#'   per-person signals given to the gate only. See
#'   [fairgate_disparity_target()].
#' @param balance Weight of the routing-diversity regulariser
#'   (negative mutual information between individual and expert). Default 0.
#' @param crossed_groups Optional two-column matrix of protected
#'   attribute codes; the variance-component penalty becomes the crossed
#'   decomposition of [fairgate_crossed_penalty()]. Objective only.
#' @param adversary,adversary_hidden Weight of the gradient
#'   reversal on an adversary predicting the protected group from the routing
#'   weights, and its hidden width. `adversary = 0` (default) is off.
#' @param es_smooth Window of the trailing moving average used
#'   for early stopping; `NULL` (default) or 1 is the unsmoothed rule.
#' @param interval_reps,interval_epochs,interval_level
#'   Warm-started bootstrap replicates, epochs per replicate and level of the
#'   per-person adjustment intervals. `interval_reps = 0` (default) is off; the
#'   point predictions are unchanged either way.
#'
#' @details
#' # Coupling to a base learner
#'
#' Write \eqn{b_i} for the base score of individual \eqn{i} on the link scale:
#' \eqn{b_i = \mathrm{logit}(\mathrm{clip}(p_i, 10^{-6}, 1 - 10^{-6}))} for a
#' binomial outcome, and \eqn{b_i = (\hat y_i - \bar y_{tr}) / s_{tr}} for a
#' gaussian one, where \eqn{\bar y_{tr}} and \eqn{s_{tr}} are the mean and
#' standard deviation of the outcome over the rows the model is fitted on,
#' the same constants the package uses to standardise the outcome, so that the
#' base score and the outcome share one scale. \eqn{b_i} is appended as an
#' extra input column, after the features have been standardised, to the gate
#' and to every expert, so the gate is
#' \eqn{\pi_i = \mathrm{softmax}(z(x_i, b_i)/\tau)} and the experts are
#' \eqn{f_k(x_i, b_i)}, \eqn{k = 1, \dots, K}. The linear predictor is
#'
#' \deqn{\eta_i = b_i + \sum_{k=1}^{K} \pi_{ik} f_k(x_i, b_i)}{eta_i = b_i + sum_k pi_ik f_k(x_i, b_i)}
#'
#' for `coupling = "offset"`, with the last layer of every expert initialised
#' at zero so that the untrained model is exactly the base learner, and
#'
#' \deqn{\eta_i = \pi_{i0} b_i + \sum_{k=1}^{K} \pi_{ik} f_k(x_i, b_i)}{eta_i = pi_i0 b_i + sum_{k>=1} pi_ik f_k(x_i, b_i)}
#'
#' for `coupling = "expert"`, where the gate has \eqn{K + 1} outputs and
#' component 0 weights the base score, so \eqn{\pi_{i0}} is the share of the
#' decision delegated to the unconstrained base learner. The task loss, the
#' fairness penalty (on the response scale, \eqn{\mathrm{sigmoid}(\eta_i)} or
#' the standardised \eqn{\eta_i}), early stopping and the protected attribute's
#' role (objective only) are exactly as for the uncoupled model. When early
#' stopping is enabled the held-out slice keeps its own out-of-fold training
#' base scores.
#'
#' Held-out predictions are assembled in double precision as
#' \eqn{\hat y_i = \tilde b_i + [h(\eta_i) - h(b_i)]}, with \eqn{h} the inverse
#' link and \eqn{\tilde b_i} the supplied base score (clipped as above for a
#' binomial outcome). This equals \eqn{h(\eta_i)} algebraically, and makes
#' \eqn{\hat y_i = \tilde b_i} hold exactly, not merely to rounding, whenever
#' the network leaves an individual's link-scale prediction unchanged. The
#' attribution export then reports \eqn{\tilde b_i}, \eqn{\hat y_i}, the
#' adjustment \eqn{\hat y_i - \tilde b_i}, \eqn{\pi_{i0}} and the full routing
#' vector.
#'
#' # The fairness penalty
#'
#' Individual `i` has prediction \eqn{\hat y_i} (a probability when
#' `family = "binomial"`, the response scale when `"gaussian"`), protected
#' group \eqn{g_i} in `1:K` and outcome stratum \eqn{s_i} in `1:Q`. For a
#' binary outcome the strata are the two outcome classes; for a continuous one
#' they are `n_strata` equal-frequency bins of `y` computed on the training
#' rows. In cell \eqn{(k, q)} of size \eqn{n_{kq}} the mean prediction is
#' \eqn{\hat m_{kq}} and its sampling variance is
#' \eqn{\hat v_{kq} = s^2_{kq} / n_{kq}}, with \eqn{s^2_{kq}} the within-cell
#' sample variance. The total penalty is \eqn{\sum_q L_q} with
#'
#' \itemize{
#'   \item `"variance"`: \eqn{L_q = \mathrm{Var}_k(\hat m_{kq})}
#'   \item `"bias_corrected"`: \eqn{L_q = \max(0, \mathrm{Var}_k(\hat m_{kq})
#'     - K_q^{-1}\sum_k \hat v_{kq})}
#'   \item `"random_effects"`: the DerSimonian and Laird between-group variance
#'     component, \eqn{L_q = \max(0, (Q_q - (K_q - 1)) / C_q)}
#' }
#'
#' The correction matters because \eqn{E[S^2] = S^2(\mu) + K^{-1}\sum_k v_k}:
#' the sample variance of the cell means carries a floor set by how precisely
#' those means are estimated, which no genuine improvement in fairness can
#' remove. The rungs nest exactly, `"random_effects"` reducing to
#' `"bias_corrected"` when the cell sampling variances are equal and both
#' reducing to `"variance"` as they go to zero, so the earlier method is a
#' special case rather than a different one. The sampling variances are always
#' detached from the autograd graph, so the optimiser cannot lower the penalty
#' by inflating within-cell spread.
#'
#' # Frontier-anchored experts
#'
#' With \eqn{\hat y_i = \sum_k \pi_{ik} f_{ik}} and
#' \eqn{\pi_i = \mathrm{softmax}(z_i/\tau)}, the gradient at gate logit
#' \eqn{l} is \eqn{\tau^{-1}\pi_{il}(f_{il} - \hat y_i)}, which vanishes when
#' the experts agree. Identical experts are therefore a fixed point: the gate
#' never specialises and the experts never differentiate. Each expert is
#' instead given its own fairness weight on the ladder of
#' [fairgate_lambda_ladder()], \eqn{\lambda_1 = 0} and
#' \eqn{\lambda_k = \lambda_{\max}10^{-(K-k)/(K-1)}}, and the objective becomes
#'
#' \deqn{L = L_{task}(\hat y) + \lambda L_{fair}(\hat y) + \alpha K^{-1}
#'   \sum_k [L_{task}(f_k) + \lambda_k L_{fair}(f_k)].}
#'
#' The auxiliary terms reuse the expert outputs already formed in the forward
#' pass, so they cost almost nothing. With \eqn{\alpha = 0} the objective is
#' exactly the earlier one.
#'
#' # Per-individual attribution
#'
#' For each held-out individual the fitted object reports the routing weights
#' \eqn{\pi_i}, the correction intensity
#' \eqn{c_i = \sum_k \pi_{ik}\lambda_k / \lambda_{\max}} in \eqn{[0, 1]}, the
#' prediction of the accuracy-anchored expert \eqn{f_{i1}}, and the fairness
#' adjustment \eqn{\delta_i = f_{i1} - \hat y_i}, the amount by which the
#' fairness objective moved that individual. The intensity is interpretable
#' only when the ladder is active, that is when `alpha > 0`.
#'
#' @return A list with:
#'   \itemize{
#'     \item `final_results` (tibble: subjectid, true, prob, group, iteration)
#'     \item `gate_weights` (tibble with gate probabilities & entropy per subject/iteration)
#'     \item `attribution` (tibble: routing weights, correction intensity
#'       `correction_intensity`, anchored-expert prediction `f_anchor`, and
#'       fairness adjustment `delta_fair`, per held-out individual and
#'       partition; with a base score also `base`, `base_link`, `adjustment`
#'       (`pred - base`) and `base_weight` (\eqn{\pi_{i0}}, `NA` unless
#'       `coupling = "expert"`). For the coupled models the ladder quantities
#'       `f_anchor`, `delta_fair` and `correction_intensity` are `NA`, because
#'       the ladder is not defined there.)
#'     \item `expert_weights` (list of expert input-layer weight matrices per repeat)
#'     \item `performance_summary` (tibble with one row per metric, AUC for a
#'       binary outcome or R-squared and RMSE for a continuous one, and columns
#'       `Metric`, `Value` (mean across partitions), `SD`, `Lower`, `Upper`
#'       (2.5th and 97.5th percentiles) and `Splits`)
#'     \item `performance_by_split` (tibble of each metric per partition)
#'     \item `early_stopping` (tibble of the epoch selected per partition, or a
#'       message when early stopping is disabled)
#'     \item `aif360_data` (tibble for fairness metric tooling)
#'     \item `tuning_results` (tibble or message when tuning skipped)
#'   }
#'
#' @examples
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
#'   data("adult_ready_small", package = "fairGATE")
#'   d <- adult_ready_small
#'   # drop columns with missing values or no variation before scaling
#'   keep <- vapply(d, function(v) !anyNA(v) && (!is.numeric(v) || stats::sd(v) > 0), TRUE)
#'   pd <- prepare_data(d[, keep], outcome_var = "income", group_var = "sex")
#'   fit <- train_gnn(pd, run_tuning = FALSE,
#'                    best_params = list(lr = 0.01, hidden_dim = 8, dropout_rate = 0.1,
#'                                       lambda = 1, temperature = 1),
#'                    num_repeats = 1, epochs = 5, seed = 1)
#'   fit$performance_summary
#'
#'   # Correcting an independently fitted base learner (here a logistic
#'   # regression, cross-fitted by hand) with the residual coupling
#'   X <- pd$X; y <- pd$y
#'   sp <- list(list(train = 1:240, test = 241:300))
#'   folds <- rep_len(1:5, 240)
#'   oof <- numeric(240)
#'   for (k in 1:5) {
#'     m <- stats::glm.fit(cbind(1, X[sp[[1]]$train[folds != k], 1:5]),
#'                         y[sp[[1]]$train[folds != k]], family = stats::binomial())
#'     oof[folds == k] <- stats::plogis(cbind(1, X[sp[[1]]$train[folds == k], 1:5]) %*% m$coefficients)
#'   }
#'   m <- stats::glm.fit(cbind(1, X[1:240, 1:5]), y[1:240], family = stats::binomial())
#'   te <- as.numeric(stats::plogis(cbind(1, X[241:300, 1:5]) %*% m$coefficients))
#'   fit2 <- train_gnn(pd, run_tuning = FALSE,
#'                     best_params = list(lr = 0.01, hidden_dim = 8, dropout_rate = 0.1,
#'                                        lambda = 1, temperature = 1),
#'                     splits = sp, base_score = list(list(train = oof, test = te)),
#'                     coupling = "residual", epochs = 5, seed = 1)
#'   head(fit2$attribution[, c("base", "pred", "adjustment", "base_weight")])
#' }
#' }
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
                      tune_epochs  = NULL,
                      splits = NULL,
                      family = c("binomial", "gaussian"),
                      n_strata = 2L,
                      standardise = TRUE,
                      impute = c("none", "median"),
                      penalty = c("random_effects", "bias_corrected", "variance", "thresholded"),
                      lambda_max = NULL,
                      alpha = 0,
                      early_stopping = FALSE,
                      es_prop = 0.2,
                      es_min_epochs = 1L,
                      es_patience = NULL,
                      base_score = NULL,
                      coupling = c("none", "offset", "expert", "residual"),
                      gate_bias_init = NULL,
                      gate = c("softmax", "sparsemax", "entmax15"),
                      intervention_cost = 0,
                      lambda_thr = 0,
                      thr_temp = 0.05,
                      thr_target = c("eo", "aod"),
                      thr_agg = c("max", "mean"),
                      reference_group = NULL,
                      gate_signal = NULL,
                      balance = 0,
                      crossed_groups = NULL,
                      adversary = 0,
                      adversary_hidden = 16L,
                      es_smooth = NULL,
                      interval_reps = 0L,
                      interval_epochs = 10L,
                      interval_level = 0.9) {

  coupling_missing <- missing(coupling)
  family <- match.arg(family)
  impute <- match.arg(impute)
  penalty <- match.arg(penalty)
  coupling <- match.arg(coupling)
  gate <- match.arg(gate)
  thr_target <- match.arg(thr_target)
  thr_agg <- match.arg(thr_agg)
  if (!is.null(base_score) && coupling_missing) {
    stop("`base_score` was supplied without `coupling`; choose coupling = 'offset' or 'expert'.")
  }
  if (!is.null(base_score) && identical(coupling, "none")) {
    stop("coupling = 'none' does not use `base_score`; choose 'offset' or 'expert', or drop `base_score`.")
  }
  if (is.null(base_score) && !identical(coupling, "none")) {
    stop(sprintf("coupling = '%s' requires `base_score`.", coupling))
  }
  coupled <- !identical(coupling, "none")
  if (!is.null(es_patience)) {
    es_patience <- as.integer(es_patience)
    if (length(es_patience) != 1L || is.na(es_patience) || es_patience < 1L) {
      stop("`es_patience` must be a single positive integer or NULL.")
    }
  }
  n_strata <- as.integer(n_strata)
  if (n_strata < 2L) stop("`n_strata` must be at least 2.")

  alpha <- as.numeric(alpha)
  if (length(alpha) != 1L || !is.finite(alpha) || alpha < 0) {
    stop("`alpha` must be a single finite non-negative number.")
  }
  if (coupled && alpha > 0) {
    stop("The frontier-anchored ladder (alpha > 0) is not defined for the coupled models; use alpha = 0 with a base score.")
  }
  if (!is.null(lambda_max)) {
    lambda_max <- as.numeric(lambda_max)
    if (length(lambda_max) != 1L || !is.finite(lambda_max) || lambda_max < 0) {
      stop("`lambda_max` must be a single finite non-negative number or NULL.")
    }
  }
  early_stopping <- isTRUE(early_stopping)
  es_prop <- as.numeric(es_prop)
  if (early_stopping && (!is.finite(es_prop) || es_prop <= 0 || es_prop >= 0.9)) {
    stop("`es_prop` must be in (0, 0.9).")
  }
  es_min_epochs <- max(1L, as.integer(es_min_epochs))

  # -------- fairGATE 3.0 components (all off by default) --------
  num1 <- function(v, nm) {
    v <- as.numeric(v)
    if (length(v) != 1L || !is.finite(v) || v < 0) stop(sprintf("`%s` must be a single finite non-negative number.", nm))
    v
  }
  intervention_cost <- num1(intervention_cost, "intervention_cost")
  lambda_thr <- num1(lambda_thr, "lambda_thr")
  balance <- num1(balance, "balance")
  adversary <- num1(adversary, "adversary")
  thr_temp <- num1(thr_temp, "thr_temp")
  if (thr_temp <= 0) stop("`thr_temp` must be positive.")
  interval_reps <- as.integer(interval_reps)
  interval_epochs <- max(1L, as.integer(interval_epochs))
  if (!is.null(es_smooth)) {
    es_smooth <- as.integer(es_smooth)
    if (length(es_smooth) != 1L || is.na(es_smooth) || es_smooth < 1L) stop("`es_smooth` must be a positive integer or NULL.")
    if (es_smooth == 1L) es_smooth <- NULL   # a window of one is the unsmoothed rule
  }
  sparse_gate <- !identical(gate, "softmax")
  base_routable <- coupling %in% c("expert", "residual")
  if (sparse_gate && !base_routable) {
    stop("A sparse gate needs coupling = 'expert' or 'residual', so that the base learner is a routable component.")
  }
  if ((intervention_cost > 0 || !is.null(gate_signal) || adversary > 0 || interval_reps > 0L) && !coupled) {
    stop("intervention_cost, gate_signal, adversary and interval_reps need a base score and coupling 'offset' or 'expert'.")
  }
  if (!is.null(crossed_groups)) {
    crossed_groups <- as.matrix(crossed_groups)
    if (ncol(crossed_groups) != 2L || nrow(crossed_groups) != nrow(as.matrix(prepared_data$X))) {
      stop("`crossed_groups` must have two columns and one row per row of X.")
    }
    if (identical(penalty, "thresholded")) stop("`crossed_groups` applies to the variance-component penalties, not 'thresholded'.")
  }
  if (!is.null(gate_signal)) {
    if (is.null(splits) || !is.list(gate_signal) || length(gate_signal) != length(splits)) {
      stop("`gate_signal` must be a list aligned with `splits`, one list(train =, test =) per partition.")
    }
  }
  # The v3 objective is formed only when a v3 term is active; otherwise the
  # 0.4.0 objective runs unchanged, call for call.
  v3_obj <- identical(penalty, "thresholded") || lambda_thr > 0 || intervention_cost > 0 ||
    balance > 0 || adversary > 0 || !is.null(crossed_groups)

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
  X[is.infinite(X)] <- NA_real_
  if (identical(family, "binomial") && any(!y[!is.na(y)] %in% c(0, 1))) {
    stop("prepared_data$y must be binary 0/1 when family = 'binomial'.")
  }
  if (identical(family, "gaussian") && !any(is.finite(y))) {
    stop("prepared_data$y contains no finite values.")
  }
  if (length(y) != nrow(X)) stop("Length of y must equal nrow(X).")
  if (length(group) != nrow(X)) stop("Length of group must equal nrow(X).")

  # Row filtering. Missing values in X are tolerated when impute = "median",
  # because per-split imputation happens later; otherwise incomplete rows are
  # dropped as in earlier versions. Rows missing y or g are always dropped.
  drop_x <- identical(impute, "none") && anyNA(X)
  if (drop_x || anyNA(y) || anyNA(group)) {
    keep <- (!drop_x | complete.cases(X)) & !is.na(y) & !is.na(group)
    if (isTRUE(verbose)) message(sprintf("[prepare] Dropped %d rows with NA in X/y/group.", sum(!keep)))
    if (!is.null(splits)) {
      stop("Rows must be dropped for missing values, which would invalidate the supplied `splits` indices. Either pass complete data, or use impute = 'median'.")
    }
    X <- X[keep, , drop = FALSE]
    y <- y[keep]
    group <- group[keep]
    if (!is.null(subject_ids)) subject_ids <- subject_ids[keep]
  }

  if (identical(family, "binomial") && length(unique(y)) < 2L) {
    stop("Only one class present in y; cannot train a classifier.")
  }
  n <- nrow(X); input_dim <- ncol(X)

  # -------- Per-split preprocessing: median imputation then standardisation --------
  # Both use training-row constants only, so held-out rows never contribute to
  # the imputation values or the centring and scaling constants.
  prep_split <- function(tr_idx, te_idx) {
    xtr <- X[tr_idx, , drop = FALSE]
    xte <- X[te_idx, , drop = FALSE]
    if (identical(impute, "median") && (anyNA(xtr) || anyNA(xte))) {
      med <- apply(xtr, 2, stats::median, na.rm = TRUE)
      med[!is.finite(med)] <- 0
      for (j in seq_len(ncol(xtr))) {
        if (anyNA(xtr[, j])) xtr[is.na(xtr[, j]), j] <- med[j]
        if (anyNA(xte[, j])) xte[is.na(xte[, j]), j] <- med[j]
      }
    }
    if (isTRUE(standardise)) {
      ctr <- colMeans(xtr)
      scl <- apply(xtr, 2, stats::sd)
      scl[!is.finite(scl) | scl < 1e-12] <- 1
      xtr <- sweep(sweep(xtr, 2, ctr, "-"), 2, scl, "/")
      xte <- sweep(sweep(xte, 2, ctr, "-"), 2, scl, "/")
    }
    if (!all(is.finite(xtr)) || !all(is.finite(xte))) {
      stop("Non-finite values remain in X after preprocessing. Check for all-missing columns.")
    }
    list(xtr = xtr, xte = xte)
  }

  # -------- Outcome scaling for the gaussian family --------
  # y is z-scored on the training rows so that the fairness weight lambda has a
  # comparable meaning across outcomes measured on different scales. All
  # reported predictions are returned on the original scale.
  y_scale_split <- function(tr_idx) {
    if (identical(family, "binomial")) return(list(m = 0, s = 1))
    m <- mean(y[tr_idx]); s <- stats::sd(y[tr_idx])
    if (!is.finite(s) || s < 1e-12) s <- 1
    list(m = m, s = s)
  }

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
      # The per-expert outputs are already formed here, so carrying them out of
      # the forward pass makes the auxiliary per-expert objective and the
      # per-individual attribution export free.
      attr(output_logits, "expert_logits") <- subgroup_logits
      output_logits
    }
  )

  # -------- fairGATE 2.0: gated correction of a base learner --------
  # A separate module, so that the uncoupled model above, and therefore the
  # order in which its parameters draw from the RNG, is untouched. The experts
  # have the same architecture as the uncoupled model. `x` already carries the
  # link-scale base score as its last column; `base` is the same score as a
  # vector, used in the combination.
  #   offset  eta = b + sum_k pi_k f_k(x, b), expert last layers zeroed
  #   expert  eta = pi_0 b + sum_{k>=1} pi_k f_k(x, b), gate with K + 1 outputs
  coupled_model <- torch::nn_module(
    "CoupledGatedModel",
    initialize = function(input_dim, hidden_dim, num_experts, dropout_p = 0.5,
                          coupling_type = "offset", gate_bias = NULL,
                          gate_type = "softmax", n_signal = 0L,
                          adv_groups = 0L, adv_hidden = 16L) {
      self$is_expert <- identical(coupling_type, "expert")
      # v3 residual coupling: component 0 is the base learner and expert k
      # proposes the corrected score b + f_k, so eta = b + sum_{k>=1} pi_k f_k.
      self$is_residual <- identical(coupling_type, "residual")
      self$n_exp <- as.integer(num_experts)
      self$gate_type <- gate_type
      n_gate <- num_experts + if (self$is_expert || self$is_residual) 1L else 0L
      # v3: the disparity-contribution signal enters the gate only, never the
      # experts; with n_signal = 0 the gate layer is the 0.4.0 layer.
      self$gate_layer <- torch::nn_linear(input_dim + as.integer(n_signal), n_gate)
      self$subgroup_layers <- torch::nn_module_list(lapply(seq_len(num_experts), function(i) {
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
      if (coupling_type %in% c("offset", "residual")) {
        for (i in seq_len(num_experts)) {
          last <- self$subgroup_layers[[i]][[7]]
          torch::nn_init_zeros_(last$weight)
          torch::nn_init_zeros_(last$bias)
        }
      }
      if (!is.null(gate_bias)) {
        if (length(gate_bias) != n_gate) {
          stop(sprintf("`gate_bias_init` must have length %d for this coupling.", n_gate))
        }
        torch::with_no_grad({
          self$gate_layer$bias$add_(torch::torch_tensor(as.numeric(gate_bias),
                                                        dtype = torch::torch_float()))
        })
      }
      # v3 adversary on the routing weights, created last and only when used,
      # so the parameter draws of every other module are unchanged.
      if (as.integer(adv_groups) > 1L) {
        self$adv <- torch::nn_sequential(
          torch::nn_linear(n_gate, as.integer(adv_hidden)),
          torch::nn_relu(),
          torch::nn_linear(as.integer(adv_hidden), as.integer(adv_groups)))
      }
    },
    forward = function(x, base, temperature_val = 0.5, gs = NULL) {
      gin <- if (is.null(gs)) x else torch::torch_cat(list(x, gs), dim = 2)
      gate_weights <- if (identical(self$gate_type, "softmax")) {
        torch::nnf_softmax(self$gate_layer(gin) / temperature_val, dim = 2)
      } else {
        fairgate_gate_map(self$gate_layer(gin) / temperature_val, self$gate_type)
      }
      f <- torch::torch_stack(lapply(self$subgroup_layers, function(layer) layer(x)$squeeze(2)),
                              dim = 2)
      if (self$is_expert) {
        eta <- gate_weights[, 1] * base +
          torch::torch_sum(gate_weights[, 2:(self$n_exp + 1L), drop = FALSE] * f, 2)
      } else if (self$is_residual) {
        eta <- base + torch::torch_sum(gate_weights[, 2:(self$n_exp + 1L), drop = FALSE] * f, 2)
      } else {
        eta <- base + torch::torch_sum(gate_weights * f, 2)
      }
      attr(eta, "gate_weights") <- gate_weights
      attr(eta, "expert_logits") <- f
      eta
    }
  )

  # -------- Outcome stratification for the fairness penalty --------
  # For a binary outcome the two strata are the negative and positive classes,
  # so the group-conditional mean prediction within a stratum is the soft
  # false-positive rate and the soft true-positive rate respectively. For a
  # continuous outcome the strata are equal-frequency bins of y computed on the
  # training rows. The penalty below is therefore one expression that reduces
  # exactly to the published binary form when family = "binomial".
  # `breaks` lets a second set of rows, in practice the early-stopping
  # validation slice, be assigned to the strata of the rows the model was fitted
  # on, rather than to bins recomputed from itself.
  make_strata <- function(y_vec, breaks = NULL) {
    if (identical(family, "binomial")) {
      return(list(ids = as.integer(y_vec == 1) + 1L, Q = 2L, breaks = NULL))
    }
    if (is.null(breaks)) {
      qs <- stats::quantile(y_vec, probs = seq(0, 1, length.out = n_strata + 1L),
                            na.rm = TRUE, type = 7)
      qs <- unique(qs)
      if (length(qs) < 3L) return(list(ids = rep(1L, length(y_vec)), Q = 1L, breaks = NULL))
      ids <- as.integer(cut(y_vec, breaks = qs, include.lowest = TRUE, labels = FALSE))
      return(list(ids = ids, Q = max(ids, na.rm = TRUE), breaks = qs))
    }
    qs <- breaks
    Qb <- length(qs) - 1L
    yc <- pmin(pmax(y_vec, qs[1]), qs[length(qs)])
    ids <- as.integer(cut(yc, breaks = qs, include.lowest = TRUE, labels = FALSE))
    ids[is.na(ids)] <- 1L
    list(ids = ids, Q = Qb, breaks = qs)
  }

  # -------- Fairness-constrained loss (robust to missing groups) --------
  fairness_constrained_loss <- function(y_out, y_true, groups, strata_ids, Q,
                                        lambda = 1.0) {
    base <- if (identical(family, "binomial")) {
      torch::nnf_binary_cross_entropy_with_logits(y_out, y_true)
    } else {
      torch::nnf_mse_loss(y_out, y_true)
    }
    if (lambda == 0 || Q < 2L) return(base)

    pred <- if (identical(family, "binomial")) {
      torch::torch_sigmoid(y_out)$view(c(-1))
    } else {
      y_out$view(c(-1))
    }

    g_int <- as.integer(as.array(groups))
    s_int <- as.integer(as.array(strata_ids))
    ug <- sort(unique(g_int))
    # A group contributes only if it is represented in every stratum. With two
    # strata and a binary outcome this is exactly the original requirement that
    # the group hold at least one positive and one negative instance.
    valid <- ug[vapply(ug, function(k) all(tabulate(s_int[g_int == k], nbins = Q) > 0),
                       logical(1))]
    if (length(valid) < 2L) return(base)

    total <- NULL
    for (q in seq_len(Q)) {
      means <- list()
      for (k in valid) {
        m <- torch::torch_tensor(as.numeric(g_int == k & s_int == q),
                                 dtype = torch::torch_float())
        cnt <- torch::torch_sum(m)
        means[[as.character(k)]] <- torch::torch_sum(pred * m) / (cnt + 1e-7)
      }
      v <- torch::torch_var(torch::torch_stack(means))
      total <- if (is.null(total)) v else total + v
    }
    base + lambda * total
  }

  # -------- Version 2 objective --------
  # The task loss and the prediction scale are separated out so that the same
  # two pieces can be applied to the mixture and to each expert in turn.
  task_loss <- function(y_out, y_true) {
    if (identical(family, "binomial")) {
      torch::nnf_binary_cross_entropy_with_logits(y_out, y_true)
    } else {
      torch::nnf_mse_loss(y_out, y_true)
    }
  }
  pred_scale <- function(y_out) {
    if (identical(family, "binomial")) {
      torch::torch_sigmoid(y_out)$view(c(-1))
    } else {
      y_out$view(c(-1))
    }
  }

  # The penalty of 0.2.0 needed no cell structure because it rebuilt its masks
  # on every call; the corrected rungs need the cells, and they do not change
  # within a partition, so they are built once.
  build_plan <- function(groups_int, strata_int, Q) {
    if (identical(penalty, "variance") || identical(penalty, "thresholded") || Q < 2L) return(NULL)
    fairgate_penalty_plan(groups_int, strata_int, Q, min_cell = 2L,
                          dtype = torch::torch_float())
  }

  # Fairness penalty on one prediction vector. Dispatches to the 0.2.0
  # arithmetic for penalty = "variance", so that path is bit-for-bit unchanged.
  # The 0.2.0 penalty, extracted from fairness_constrained_loss() with its
  # arithmetic unchanged, so that penalty = "variance" can also be applied to an
  # expert's predictions when the anchored ladder is in use.
  legacy_variance_term <- function(pred, groups, strata_ids, Q) {
    if (Q < 2L) return(NULL)
    g_int <- as.integer(as.array(groups))
    s_int <- as.integer(as.array(strata_ids))
    ug <- sort(unique(g_int))
    valid <- ug[vapply(ug, function(k) all(tabulate(s_int[g_int == k], nbins = Q) > 0),
                       logical(1))]
    if (length(valid) < 2L) return(NULL)
    total <- NULL
    for (q in seq_len(Q)) {
      means <- list()
      for (k in valid) {
        m <- torch::torch_tensor(as.numeric(g_int == k & s_int == q),
                                 dtype = torch::torch_float())
        cnt <- torch::torch_sum(m)
        means[[as.character(k)]] <- torch::torch_sum(pred * m) / (cnt + 1e-7)
      }
      v <- torch::torch_var(torch::torch_stack(means))
      total <- if (is.null(total)) v else total + v
    }
    total
  }

  fair_term <- function(y_out, groups, strata_ids, Q, plan) {
    pred <- pred_scale(y_out)
    if (identical(penalty, "variance")) {
      return(legacy_variance_term(pred, groups, strata_ids, Q))
    }
    if (is.null(plan) || length(plan) == 0L) return(NULL)
    fairgate_fair_penalty(pred, penalty = penalty, plan = plan)
  }

  # Full objective. With penalty = "variance" and alpha = 0 this is exactly the
  # 0.2.0 loss, computed by the 0.2.0 code path.
  objective <- function(y_out, y_true, groups, strata_ids, Q, plan,
                        lambda, lam_ladder, alpha_use) {
    if (identical(penalty, "variance") && alpha_use == 0) {
      return(fairness_constrained_loss(y_out, y_true, groups, strata_ids, Q,
                                       lambda = lambda))
    }
    total <- task_loss(y_out, y_true)
    if (lambda != 0) {
      ft <- fair_term(y_out, groups, strata_ids, Q, plan)
      if (!is.null(ft)) total <- total + lambda * ft
    }
    if (alpha_use > 0) {
      ex <- attr(y_out, "expert_logits")
      if (is.null(ex)) stop("internal: expert outputs unavailable for alpha > 0.")
      K_ex <- as.integer(unlist(ex$size())[2])
      aux <- NULL
      for (k in seq_len(K_ex)) {
        fk <- ex[, k]
        fk_use <- if (length(y_true$size()) == 1L) fk else fk$unsqueeze(2)
        term <- task_loss(fk_use, y_true)
        lam_k <- lam_ladder[k]
        if (lam_k != 0) {
          ftk <- fair_term(fk, groups, strata_ids, Q, plan)
          if (!is.null(ftk)) term <- term + lam_k * ftk
        }
        aux <- if (is.null(aux)) term else aux + term
      }
      total <- total + alpha_use * aux / K_ex
    }
    total
  }

  # -------- fairGATE 3.0 objective --------
  # Used only when a v3 term is active (v3_obj). `ctx` carries the per-row
  # quantities of the rows being scored: the link-scale base score `b`, the
  # thresholded-penalty plan `tplan` (binomial) or `gplan` (gaussian), the
  # crossed plan `cplan`, the training prevalence `prev`, and, for training
  # rows only, the network and group indices used by the adversary.
  # Returns the training loss and the monitored objective: task loss plus
  # every fairness term plus the intervention cost. The routing regularisers
  # (balance, adversary) shape training only and are not monitored.
  thr_term <- function(y_out, ctx) {
    p <- pred_scale(y_out)
    if (identical(family, "binomial")) {
      fairgate_threshold_penalty(p, ctx$tplan, ctx$prev, temp = thr_temp,
                                 target = thr_target, agg = thr_agg)
    } else {
      fairgate_group_gap_penalty(p, ctx$gplan, reference = reference_group, agg = thr_agg)
    }
  }
  objective_v3 <- function(y_out, y_true, groups, strata_ids, Q, plan, lambda, ctx,
                           train = TRUE) {
    mon <- task_loss(y_out, y_true)
    if (lambda != 0) {
      ft <- if (identical(penalty, "thresholded")) {
        thr_term(y_out, ctx)
      } else if (!is.null(ctx$cplan)) {
        fairgate_crossed_penalty(pred_scale(y_out), ctx$cplan, penalty = penalty)
      } else {
        fair_term(y_out, groups, strata_ids, Q, plan)
      }
      if (!is.null(ft)) mon <- mon + lambda * ft
    }
    if (lambda_thr > 0) mon <- mon + lambda_thr * thr_term(y_out, ctx)
    if (intervention_cost > 0 && !is.null(ctx$b)) {
      mon <- mon + intervention_cost * torch::torch_mean(torch::torch_abs(y_out$view(c(-1)) - ctx$b))
    }
    total <- mon
    if (train) {
      gw <- attr(y_out, "gate_weights")
      if (balance > 0 && !is.null(gw)) {
        total <- total + balance * .fg_balance(gw, base_routable)
      }
      if (adversary > 0 && !is.null(gw) && !is.null(ctx$net$adv)) {
        al <- torch::nnf_cross_entropy(ctx$net$adv(.fg_grl(gw, adversary)), ctx$gidx)
        total <- total + al
      }
    }
    list(total = total, monitor = mon)
  }
  # Plans for the v3 terms on one set of rows. `prev` is always the
  # prevalence of the rows the model is fitted on.
  v3_ctx <- function(rows, strata_ids, Q, prev, b_t = NULL) {
    if (!v3_obj) return(NULL)
    ctx <- list(prev = prev, b = b_t)
    if (identical(penalty, "thresholded") || lambda_thr > 0) {
      if (identical(family, "binomial")) {
        ctx$tplan <- fairgate_threshold_plan(group[rows], y[rows], reference = reference_group,
                                             min_cell = 2L)
      } else {
        ctx$gplan <- fairgate_penalty_plan(group[rows], strata_ids, Q, min_cell = 1L,
                                           dtype = torch::torch_float())
      }
    }
    if (!is.null(crossed_groups) && Q >= 2L) {
      ctx$cplan <- fairgate_crossed_plan(crossed_groups[rows, 1], crossed_groups[rows, 2],
                                         strata_ids, Q, min_cell = 2L)
    }
    ctx$gidx <- torch::torch_tensor(match(group[rows], group_levels), dtype = torch::torch_long())
    ctx
  }

  # -------- Early stopping: stratified validation slice of the training rows --
  # Stratification is joint on outcome stratum and protected group, so every
  # cell that carries the fairness penalty is represented on both sides of the
  # split whenever it is large enough to be.
  es_slice <- function(tr_idx, strata_ids, seed_local = NULL) {
    if (!is.null(seed_local)) set.seed(as.integer(seed_local))
    key <- paste(strata_ids, group[tr_idx], sep = "_")
    val_pos <- integer(0)
    for (lv in unique(key)) {
      pos <- which(key == lv)
      n_lv <- length(pos)
      if (n_lv < 4L) next          # too small to give up a row
      take <- max(1L, floor(n_lv * es_prop))
      take <- min(take, n_lv - 2L) # leave at least two rows in training
      if (take > 0L) val_pos <- c(val_pos, sample(pos, take))
    }
    val_pos <- sort(unique(val_pos))
    if (length(val_pos) < 2L || length(val_pos) > length(tr_idx) - 2L) {
      return(NULL)
    }
    list(fit = setdiff(seq_along(tr_idx), val_pos), val = val_pos)
  }

  # Realised outcome stratification of the most recent partition fitted. Q is
  # data-determined: equal-frequency bin edges that coincide, which happens
  # whenever the outcome is heavily tied, are collapsed, so a request for
  # n_strata bins can yield fewer. Reported so that the quantity trained
  # against can be checked against the quantity reported elsewhere.
  strata_seen <- NULL

  # -------- One partition: preprocess, fit, optionally early stop -----------
  # Shared by tuning and by the final run so that the two cannot drift apart.
  # With early_stopping = FALSE, penalty = "variance" and alpha = 0 the epoch
  # loop is the 0.2.0 loop, calling the 0.2.0 loss.
  fit_partition <- function(tr, te, p_hid, p_drop, p_lr, p_lam, p_temp,
                            epochs_use, torch_seed = NULL, es_seed = NULL,
                            bs = NULL, gsig = NULL) {
    if (!is.null(torch_seed)) try(torch::torch_manual_seed(torch_seed), silent = TRUE)

    sl <- NULL
    if (early_stopping) {
      st_all <- make_strata(y[tr])
      sl <- es_slice(tr, st_all$ids, seed_local = es_seed)
    }
    tr_fit <- if (is.null(sl)) tr else tr[sl$fit]
    tr_val <- if (is.null(sl)) integer(0) else tr[sl$val]

    pp <- prep_split(tr_fit, te)
    ys <- y_scale_split(tr_fit)
    st <- make_strata(y[tr_fit])
    plan <- build_plan(group[tr_fit], st$ids, st$Q)

    # Base scores, aligned with the partition's own row order, converted to the
    # link scale with this partition's outcome constants and appended to the
    # standardised features. The early-stopping slice keeps the out-of-fold
    # training scores of its own rows.
    if (coupled) {
      b_fit_resp <- if (is.null(sl)) bs$train else bs$train[sl$fit]
      b_val_resp <- if (is.null(sl)) numeric(0) else bs$train[sl$val]
      b_te_resp  <- bs$test
      b_fit <- base_to_link(b_fit_resp, ys)
      b_val <- base_to_link(b_val_resp, ys)
      b_te  <- base_to_link(b_te_resp, ys)
      pp$xtr <- cbind(pp$xtr, base_score = b_fit)
      pp$xte <- cbind(pp$xte, base_score = b_te)
      bfit_t <- torch::torch_tensor(b_fit, dtype = torch::torch_float())
      bte_t  <- torch::torch_tensor(b_te, dtype = torch::torch_float())
    }

    # v3: cross-fitted gate signal, standardised with the fitted rows'
    # constants, for the gate only.
    n_signal <- 0L
    gs_fit_t <- NULL; gs_val_t <- NULL; gs_te_t <- NULL
    if (!is.null(gsig)) {
      gtrm <- as.matrix(gsig$train); gtem <- as.matrix(gsig$test)
      if (nrow(gtrm) != length(tr) || nrow(gtem) != length(te)) {
        stop("`gate_signal` rows do not match the partition.")
      }
      g_fit <- if (is.null(sl)) gtrm else gtrm[sl$fit, , drop = FALSE]
      g_val <- if (is.null(sl)) gtrm[0, , drop = FALSE] else gtrm[sl$val, , drop = FALSE]
      gc <- colMeans(g_fit); gsd <- apply(g_fit, 2, stats::sd)
      gsd[!is.finite(gsd) | gsd < 1e-12] <- 1
      zs <- function(m) sweep(sweep(m, 2, gc, "-"), 2, gsd, "/")
      n_signal <- ncol(gtrm)
      gs_fit_t <- torch::torch_tensor(zs(g_fit), dtype = torch::torch_float())
      gs_te_t <- torch::torch_tensor(zs(gtem), dtype = torch::torch_float())
      if (nrow(g_val)) gs_val_t <- torch::torch_tensor(zs(g_val), dtype = torch::torch_float())
    }

    xtr <- torch::torch_tensor(pp$xtr, dtype = torch::torch_float())
    xte <- torch::torch_tensor(pp$xte, dtype = torch::torch_float())
    ytr <- torch::torch_tensor((y[tr_fit] - ys$m) / ys$s, dtype = torch::torch_float())
    gtr <- g_all_t[tr_fit]
    str_t <- torch::torch_tensor(st$ids, dtype = torch::torch_long())

    have_val <- length(tr_val) > 0L
    if (have_val) {
      ppv <- prep_split(tr_fit, tr_val)
      if (coupled) ppv$xte <- cbind(ppv$xte, base_score = b_val)
      stv <- make_strata(y[tr_val], breaks = st$breaks)
      xval <- torch::torch_tensor(ppv$xte, dtype = torch::torch_float())
      yval <- torch::torch_tensor((y[tr_val] - ys$m) / ys$s, dtype = torch::torch_float())
      gval <- g_all_t[tr_val]
      sval_t <- torch::torch_tensor(stv$ids, dtype = torch::torch_long())
      plan_val <- build_plan(group[tr_val], stv$ids, st$Q)
      if (coupled) bval_t <- torch::torch_tensor(b_val, dtype = torch::torch_float())
    }

    lam_max_use <- if (is.null(lambda_max)) p_lam else lambda_max
    lam_ladder <- fairgate_lambda_ladder(num_groups, lam_max_use)

    if (coupled) {
      net <- coupled_model(input_dim + 1L, p_hid, num_groups, dropout_p = p_drop,
                           coupling_type = coupling, gate_bias = gate_bias_init,
                           gate_type = gate, n_signal = n_signal,
                           adv_groups = if (adversary > 0) num_groups else 0L,
                           adv_hidden = adversary_hidden)
      fwd_tr  <- function() net(xtr, bfit_t, temperature_val = p_temp, gs = gs_fit_t)
      fwd_val <- function() net(xval, bval_t, temperature_val = p_temp, gs = gs_val_t)
    } else {
      net <- gated_model(input_dim, p_hid, num_groups, dropout_p = p_drop)
      if (!is.null(gate_bias_init)) {
        if (length(gate_bias_init) != num_groups) {
          stop(sprintf("`gate_bias_init` must have length %d for this coupling.", num_groups))
        }
        torch::with_no_grad({
          net$gate_layer$bias$add_(torch::torch_tensor(as.numeric(gate_bias_init),
                                                       dtype = torch::torch_float()))
        })
      }
      fwd_tr  <- function() net(xtr, temperature_val = p_temp)
      fwd_val <- function() net(xval, temperature_val = p_temp)
    }
    opt <- torch::optim_adam(net$parameters, lr = p_lr)

    # v3 contexts (NULL unless a v3 objective term is active).
    prev_fit <- if (identical(family, "binomial")) mean(y[tr_fit]) else NA_real_
    ctx_tr <- v3_ctx(tr_fit, st$ids, st$Q, prev_fit, if (coupled) bfit_t else NULL)
    if (!is.null(ctx_tr)) ctx_tr$net <- net
    ctx_val <- if (have_val) v3_ctx(tr_val, stv$ids, st$Q, prev_fit, if (coupled) bval_t else NULL) else NULL

    best_obj <- Inf; best_sd <- NULL; best_ep <- NA_integer_
    since_best <- 0L
    epochs_run <- 0L
    vo_hist <- numeric(0)
    for (ep in seq_len(as.integer(epochs_use))) {
      net$train(); opt$zero_grad()
      logits <- fwd_tr()
      ytr_use <- if (length(logits$size()) == 1L) ytr else ytr$unsqueeze(2)
      loss <- if (v3_obj) {
        objective_v3(logits, ytr_use, gtr, str_t, st$Q, plan, p_lam, ctx_tr, train = TRUE)$total
      } else {
        objective(logits, ytr_use, gtr, str_t, st$Q, plan, p_lam, lam_ladder, alpha)
      }
      if (!is.finite(as.numeric(loss$item()))) {
        if (isTRUE(verbose)) message("  Loss became non-finite; aborting this repeat.")
        break
      }
      loss$backward(); opt$step()
      epochs_run <- ep

      if (have_val) {
        net$eval()
        vo <- torch::with_no_grad({
          lo <- fwd_val()
          yv_use <- if (length(lo$size()) == 1L) yval else yval$unsqueeze(2)
          # The selection criterion is the objective the model is asked to
          # minimise, task loss plus the weighted fairness penalty, on rows it
          # was not fitted to. The auxiliary expert terms are excluded: they are
          # a device for shaping the mixture, not part of what is being judged.
          if (v3_obj) {
            as.numeric(objective_v3(lo, yv_use, gval, sval_t, st$Q, plan_val, p_lam,
                                    ctx_val, train = FALSE)$monitor$item())
          } else {
            as.numeric(objective(lo, yv_use, gval, sval_t, st$Q, plan_val,
                                 p_lam, lam_ladder, 0)$item())
          }
        })
        if (is.null(es_smooth)) {
          if (is.finite(vo) && ep >= es_min_epochs && vo < best_obj) {
            best_obj <- vo; best_ep <- ep
            best_sd <- lapply(net$state_dict(), function(z) z$clone())
            since_best <- 0L
          } else if (ep >= es_min_epochs) {
            since_best <- since_best + 1L
            if (!is.null(es_patience) && since_best >= es_patience) break
          }
        } else {
          # v3 smoothed early stopping: the criterion is the trailing moving
          # average of the monitored objective over es_smooth epochs; the
          # parameters restored are those at the end of the best window.
          vo_hist <- c(vo_hist, vo)
          if (length(vo_hist) >= es_smooth) {
            crit <- mean(utils::tail(vo_hist, es_smooth))
            if (is.finite(crit) && ep >= es_min_epochs && crit < best_obj) {
              best_obj <- crit; best_ep <- ep
              best_sd <- lapply(net$state_dict(), function(z) z$clone())
              since_best <- 0L
            } else if (ep >= es_min_epochs) {
              since_best <- since_best + 1L
              if (!is.null(es_patience) && since_best >= es_patience) break
            }
          }
        }
      }
    }
    if (!is.null(best_sd)) net$load_state_dict(best_sd)

    # v3 per-person adjustment intervals: a warm-started bootstrap. From the
    # selected parameters, each replicate refits for interval_epochs on a
    # bootstrap resample of the fitted rows and records every held-out
    # individual's adjustment; the selected parameters are then restored, so
    # the point predictions are those of the main fit.
    adj_reps <- NULL
    if (coupled && interval_reps > 0L) {
      sd0 <- lapply(net$state_dict(), function(z) z$clone())
      nfit <- length(tr_fit)
      adj_reps <- matrix(NA_real_, length(te), interval_reps)
      old_seed <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) get(".Random.seed", envir = .GlobalEnv) else NULL
      for (rr in seq_len(interval_reps)) {
        net$load_state_dict(sd0)
        if (!is.null(torch_seed)) try(torch::torch_manual_seed(torch_seed + 7919L * rr), silent = TRUE)
        set.seed(if (is.null(torch_seed)) rr else torch_seed + 7919L * rr)
        idx <- sample.int(nfit, nfit, replace = TRUE)
        st_b <- st$ids[idx]
        plan_b <- build_plan(group[tr_fit][idx], st_b, st$Q)
        str_b <- torch::torch_tensor(st_b, dtype = torch::torch_long())
        ctx_b <- v3_ctx(tr_fit[idx], st_b, st$Q, prev_fit, bfit_t[idx])
        if (!is.null(ctx_b)) ctx_b$net <- net
        xb <- xtr[idx, ]; bb <- bfit_t[idx]; yb <- ytr[idx]; gb <- gtr[idx]
        gsb <- if (is.null(gs_fit_t)) NULL else gs_fit_t[idx, , drop = FALSE]
        opt_b <- torch::optim_adam(net$parameters, lr = p_lr)
        for (e2 in seq_len(interval_epochs)) {
          net$train(); opt_b$zero_grad()
          lb <- net(xb, bb, temperature_val = p_temp, gs = gsb)
          yb_use <- if (length(lb$size()) == 1L) yb else yb$unsqueeze(2)
          lossb <- if (v3_obj) {
            objective_v3(lb, yb_use, gb, str_b, st$Q, plan_b, p_lam, ctx_b, train = TRUE)$total
          } else {
            objective(lb, yb_use, gb, str_b, st$Q, plan_b, p_lam, lam_ladder, 0)
          }
          if (!is.finite(as.numeric(lossb$item()))) break
          lossb$backward(); opt_b$step()
        }
        net$eval()
        ob <- torch::with_no_grad({ net(xte, bte_t, temperature_val = p_temp, gs = gs_te_t) })
        asm <- assemble_coupled(ob, b_te, base_resp_clip(b_te_resp), ys)
        adj_reps[, rr] <- asm$pred - asm$base
      }
      net$load_state_dict(sd0)
      if (!is.null(old_seed)) assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }

    strata_seen <<- list(Q = st$Q, breaks = st$breaks, n_fit = length(tr_fit),
                         cells = if (is.null(plan)) NULL else
                           lapply(plan, function(cl) list(q = cl$q, groups = cl$groups,
                                                          n = as.numeric(cl$n_num))))
    list(net = net, ys = ys, st = st, xte = xte, plan = plan,
         lam_ladder = lam_ladder, lam_max = lam_max_use,
         n_fit = length(tr_fit), n_val = length(tr_val),
         epochs_run = epochs_run,
         epoch_selected = if (have_val) best_ep else epochs_run,
         val_objective = if (have_val) best_obj else NA_real_,
         bte_t = if (coupled) bte_t else NULL,
         b_te_link = if (coupled) b_te else NULL,
         b_te_resp = if (coupled) base_resp_clip(b_te_resp) else NULL,
         gs_te = gs_te_t, adj_reps = adj_reps)
  }

  # -------- Held-out predictions, routing and attribution -------------------
  predict_partition <- function(ft, p_temp) {
    ft$net$eval()
    if (coupled) return(predict_coupled(ft, p_temp))
    logits_te <- torch::with_no_grad({ ft$net(ft$xte, temperature_val = p_temp) })
    to_response <- function(z) {
      if (identical(family, "binomial")) {
        as.numeric(torch::torch_sigmoid(z))
      } else {
        as.numeric(z) * ft$ys$s + ft$ys$m
      }
    }
    probs <- to_response(logits_te)
    probs[!is.finite(probs)] <- NA_real_
    if (identical(family, "binomial")) probs <- pmin(pmax(probs, 1e-6), 1 - 1e-6)

    gw_mat <- torch::as_array(attr(logits_te, "gate_weights"))
    ex <- attr(logits_te, "expert_logits")
    f_anchor <- if (is.null(ex)) rep(NA_real_, length(probs)) else {
      fa <- to_response(ex[, 1])
      fa[!is.finite(fa)] <- NA_real_
      if (identical(family, "binomial")) pmin(pmax(fa, 1e-6), 1 - 1e-6) else fa
    }
    # Correction intensity: the ladder weight the routing places on this
    # individual, as a share of the heaviest weight on the ladder. Interpretable
    # only when the ladder is active, that is when alpha > 0.
    c_i <- if (is.finite(ft$lam_max) && ft$lam_max > 0) {
      as.numeric(gw_mat %*% ft$lam_ladder) / ft$lam_max
    } else rep(NA_real_, length(probs))

    na <- rep(NA_real_, length(probs))
    list(pred = probs, gate = gw_mat, f_anchor = f_anchor,
         delta_fair = f_anchor - probs, correction_intensity = c_i,
         base = na, base_link = na, adjustment = na, base_weight = na)
  }

  # Coupled models. The network runs in single precision; the prediction is
  # assembled in double precision as base + [h(eta) - h(b)], algebraically
  # h(eta), so that an individual the network leaves unchanged on the link
  # scale gets back the supplied base score exactly.
  predict_coupled <- function(ft, p_temp) {
    out <- torch::with_no_grad({ ft$net(ft$xte, ft$bte_t, temperature_val = p_temp, gs = ft$gs_te) })
    asm <- assemble_coupled(out, ft$b_te_link, ft$b_te_resp, ft$ys)
    na <- rep(NA_real_, length(asm$pred))
    res <- list(pred = asm$pred, gate = asm$gate, f_anchor = na, delta_fair = na,
                correction_intensity = na,
                base = asm$base, base_link = asm$base_link, adjustment = asm$pred - asm$base,
                base_weight = asm$bw)
    if (sparse_gate) { res$n_active <- asm$n_active; res$base_only <- asm$base_only }
    if (!is.null(ft$adj_reps)) {
      lo <- (1 - interval_level) / 2
      res$adj_lo <- apply(ft$adj_reps, 1, stats::quantile, probs = lo, na.rm = TRUE, names = FALSE)
      res$adj_hi <- apply(ft$adj_reps, 1, stats::quantile, probs = 1 - lo, na.rm = TRUE, names = FALSE)
      res$adj_sd <- apply(ft$adj_reps, 1, stats::sd, na.rm = TRUE)
    }
    res
  }

  # Assembly of a coupled model's held-out prediction in double precision.
  # With a sparse gate the routing weights are renormalised in double
  # precision, so an individual whose only non-zero weight is on the base
  # component gets weight exactly one there and the base score back exactly.
  assemble_coupled <- function(out, b_d, base_resp, ys) {
    gw_mat <- torch::as_array(attr(out, "gate_weights"))
    f_mat <- torch::as_array(attr(out, "expert_logits"))
    if (!is.matrix(gw_mat)) gw_mat <- matrix(gw_mat, nrow = 1L)
    if (!is.matrix(f_mat)) f_mat <- matrix(f_mat, nrow = 1L)
    n_active <- NULL; base_only <- NULL
    gw_use <- gw_mat
    if (sparse_gate) {
      gw_use <- gw_mat / rowSums(gw_mat)
      n_active <- rowSums(gw_mat > 0)
      base_only <- gw_mat[, 1] > 0 & n_active == 1L
    }
    if (identical(coupling, "expert")) {
      eta_d <- gw_use[, 1] * b_d + rowSums(gw_use[, -1, drop = FALSE] * f_mat)
      bw <- gw_use[, 1]
    } else if (identical(coupling, "residual")) {
      eta_d <- b_d + rowSums(gw_use[, -1, drop = FALSE] * f_mat)
      bw <- gw_use[, 1]
    } else {
      eta_d <- b_d + rowSums(gw_use * f_mat)
      bw <- rep(NA_real_, length(b_d))
    }
    h <- function(z) if (identical(family, "binomial")) stats::plogis(z) else z * ys$s + ys$m
    pred <- base_resp + (h(eta_d) - h(b_d))
    pred[!is.finite(pred)] <- NA_real_
    if (identical(family, "binomial")) pred <- pmin(pmax(pred, 1e-6), 1 - 1e-6)
    list(pred = pred, gate = gw_mat, bw = bw, base = base_resp, base_link = b_d,
         n_active = n_active, base_only = base_only)
  }

  # Response-scale base score as the model uses it: clipped for a binomial
  # outcome, as supplied for a gaussian one.
  base_resp_clip <- function(b) {
    if (identical(family, "binomial")) pmin(pmax(as.numeric(b), 1e-6), 1 - 1e-6) else as.numeric(b)
  }
  # Link-scale base score: logit of the clipped probability, or the outcome
  # standardised with the partition's own outcome constants.
  base_to_link <- function(b, ys) {
    if (!length(b)) return(numeric(0))
    if (identical(family, "binomial")) stats::qlogis(base_resp_clip(b)) else (as.numeric(b) - ys$m) / ys$s
  }

  # -------- Prepare group tensor; feature tensors are built per split --------
  num_groups <- length(unique(group))
  group_levels <- sort(unique(group))
  g_all_t <- torch::torch_tensor(as.integer(group), dtype = torch::torch_long())

  # -------- Split supply: either pre-computed or drawn internally --------
  if (!is.null(splits)) {
    if (!is.list(splits) || length(splits) == 0L) stop("`splits` must be a non-empty list.")
    for (i in seq_along(splits)) {
      sp <- splits[[i]]
      if (!is.list(sp) || !all(c("train", "test") %in% names(sp))) {
        stop(sprintf("splits[[%d]] must be a list with elements `train` and `test`.", i))
      }
      idx <- c(sp$train, sp$test)
      if (any(idx < 1L) || any(idx > n)) {
        stop(sprintf("splits[[%d]] contains indices outside 1..%d.", i, n))
      }
      if (length(intersect(sp$train, sp$test)) > 0L) {
        stop(sprintf("splits[[%d]] has overlapping train and test indices.", i))
      }
    }
    num_repeats <- length(splits)
    if (isTRUE(verbose)) message(sprintf("[splits] Using %d supplied partitions.", num_repeats))
  }

  # -------- Base scores: one list(train, test) per supplied partition --------
  if (coupled) {
    if (is.null(splits)) stop("`base_score` requires `splits`, so that each score can be aligned with its row.")
    if (!is.list(base_score) || length(base_score) != length(splits)) {
      stop(sprintf("`base_score` must be a list with one element per partition (%d).", length(splits)))
    }
    for (i in seq_along(base_score)) {
      b <- base_score[[i]]
      if (!is.list(b) || !all(c("train", "test") %in% names(b))) {
        stop(sprintf("base_score[[%d]] must be a list with elements `train` and `test`.", i))
      }
      if (length(b$train) != length(splits[[i]]$train) || length(b$test) != length(splits[[i]]$test)) {
        stop(sprintf("base_score[[%d]] lengths (%d, %d) do not match splits[[%d]] (%d, %d).",
                     i, length(b$train), length(b$test), i,
                     length(splits[[i]]$train), length(splits[[i]]$test)))
      }
      vals <- c(b$train, b$test)
      if (!is.numeric(vals) || any(!is.finite(vals))) {
        stop(sprintf("base_score[[%d]] contains non-numeric or non-finite values.", i))
      }
      if (identical(family, "binomial") && any(vals < 0 | vals > 1)) {
        stop(sprintf("base_score[[%d]] must hold probabilities in [0, 1] for family = 'binomial'.", i))
      }
    }
  }

  random_split <- function(n_obs, test_prop = 0.20, seed_local = NULL) {
    if (!is.null(seed_local)) set.seed(as.integer(seed_local))
    te <- sort(sample(seq_len(n_obs), max(1L, floor(n_obs * test_prop))))
    list(train = setdiff(seq_len(n_obs), te), test = te)
  }

  get_split <- function(r) {
    if (!is.null(splits)) return(splits[[r]])
    s <- next_split_seed()
    if (identical(family, "binomial")) {
      stratified_split_both_in_test(y, test_prop = 0.20, seed_local = s)
    } else {
      random_split(n, test_prop = 0.20, seed_local = s)
    }
  }

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
      es_epochs_tune <- integer(0)
      n_tune <- if (!is.null(splits)) min(as.integer(tune_repeats), length(splits)) else as.integer(tune_repeats)
      for (r in seq_len(n_tune)) {
        if (interactive()) utils::flush.console()
        sp <- get_split(r)
        tr <- sp$train; te <- sp$test
        yte <- y[te]

        ft <- fit_partition(tr, te, p_hid, p_drop, p_lr, p_lam, p_temp,
                            epochs_use = as.integer(tune_epochs),
                            torch_seed = if (is.null(base_seed)) NULL else base_seed + r,
                            es_seed = if (is.null(base_seed)) NULL else base_seed + 500000L + r,
                            bs = if (coupled) base_score[[r]] else NULL,
                            gsig = if (is.null(gate_signal)) NULL else gate_signal[[r]])
        preds <- predict_partition(ft, p_temp)$pred

        if (identical(family, "binomial")) {
          aucs <- c(aucs, safe_auc(yte, preds))
        } else {
          ss_res <- sum((yte - preds)^2, na.rm = TRUE)
          ss_tot <- sum((yte - mean(yte))^2, na.rm = TRUE)
          aucs <- c(aucs, if (ss_tot > 0) 1 - ss_res / ss_tot else NA_real_)
        }
        es_epochs_tune <- c(es_epochs_tune, ft$epoch_selected)
      }

      mean_auc <- mean(aucs, na.rm = TRUE)
      tuning_results <- dplyr::bind_rows(
        tuning_results,
        tibble::as_tibble(list(
          lr = p_lr, hidden_dim = p_hid, dropout_rate = p_drop, lambda = p_lam, temperature = p_temp,
          mean_auc = mean_auc,
          mean_epoch_selected = if (length(es_epochs_tune)) {
            mean(as.numeric(es_epochs_tune), na.rm = TRUE)
          } else NA_real_
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
  attribution_list <- list()
  es_list <- list()

  for (r in seq_len(as.integer(num_repeats))) {
    sp <- get_split(r)
    tr <- sp$train; te <- sp$test

    yte <- y[te]
    gte <- group[te]
    # Row indices of the supplied data are used when no identifier column is
    # available, so that predictions remain linkable to the individual they
    # came from. Earlier versions used the within-fold position, which is not
    # an identifier and silently collides across repeats.
    subj_te <- if (!is.null(subject_ids)) subject_ids[te] else te

    ft <- fit_partition(tr, te, final_hidden_dim, final_dropout, final_lr,
                        final_lambda, final_temperature,
                        epochs_use = as.integer(epochs),
                        torch_seed = if (is.null(base_seed)) NULL else base_seed + 10000L + r,
                        es_seed = if (is.null(base_seed)) NULL else base_seed + 600000L + r,
                        bs = if (coupled) base_score[[r]] else NULL,
                            gsig = if (is.null(gate_signal)) NULL else gate_signal[[r]])
    pr <- predict_partition(ft, final_temperature)
    probs <- pr$pred
    gw_mat <- pr$gate

    gw_clamped <- pmin(pmax(gw_mat, 1e-10), 1 - 1e-10)
    gate_entropy <- as.numeric(-rowSums(gw_clamped * log(gw_clamped)))

    # Expert first-layer weights (input_dim x hidden), per expert
    expert_weights_list[[r]] <- lapply(
      ft$net$subgroup_layers,
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
    attribution_list[[r]] <- dplyr::bind_cols(
      tibble::tibble(iteration = r, subjectid = subj_te, group = gte,
                     pred = probs, f_anchor = pr$f_anchor,
                     delta_fair = pr$delta_fair,
                     correction_intensity = pr$correction_intensity,
                     base = pr$base, base_link = pr$base_link,
                     adjustment = pr$adjustment, base_weight = pr$base_weight),
      gate_probs_df
    )
    # v3 columns, present only when the component producing them is on.
    if (!is.null(pr$n_active)) {
      attribution_list[[r]]$n_active <- as.integer(pr$n_active)
      attribution_list[[r]]$base_only <- pr$base_only
    }
    if (!is.null(pr$adj_lo)) {
      attribution_list[[r]]$adj_lo <- pr$adj_lo
      attribution_list[[r]]$adj_hi <- pr$adj_hi
      attribution_list[[r]]$adj_sd <- pr$adj_sd
    }
    es_list[[r]] <- tibble::tibble(
      iteration = r, epoch_selected = as.integer(ft$epoch_selected),
      epochs_run = as.integer(ft$epochs_run),
      val_objective = as.numeric(ft$val_objective),
      n_fit = ft$n_fit, n_val = ft$n_val
    )
    if (isTRUE(verbose)) message(sprintf("[final] repeat %d/%d complete (epoch %s)",
                                         r, num_repeats, as.character(ft$epoch_selected)))
    if (interactive()) utils::flush.console()
  }

  final_results <- dplyr::bind_rows(results_list)
  gate_weights  <- dplyr::bind_rows(gate_weights_list)
  attribution   <- dplyr::bind_rows(attribution_list)
  early_stopping_record <- dplyr::bind_rows(es_list)

  # Performance is summarised per split and then averaged, rather than computed
  # once on predictions pooled across splits. Pooling treats repeated
  # appearances of the same individual as independent observations, which
  # understates uncertainty; the per-split spread below is the honest summary.
  per_split <- do.call(rbind, lapply(split(final_results, final_results$iteration), function(d) {
    if (identical(family, "binomial")) {
      data.frame(iteration = d$iteration[1],
                 metric = "AUC",
                 value = safe_auc(d$true, d$prob))
    } else {
      ss_res <- sum((d$true - d$prob)^2, na.rm = TRUE)
      ss_tot <- sum((d$true - mean(d$true, na.rm = TRUE))^2, na.rm = TRUE)
      rbind(
        data.frame(iteration = d$iteration[1], metric = "R2",
                   value = if (ss_tot > 0) 1 - ss_res / ss_tot else NA_real_),
        data.frame(iteration = d$iteration[1], metric = "RMSE",
                   value = sqrt(mean((d$true - d$prob)^2, na.rm = TRUE)))
      )
    }
  }))
  per_split <- tibble::as_tibble(per_split)

  performance_summary <- per_split %>%
    dplyr::group_by(.data$metric) %>%
    dplyr::summarise(
      Value   = mean(.data$value, na.rm = TRUE),
      SD      = stats::sd(.data$value, na.rm = TRUE),
      Lower   = stats::quantile(.data$value, 0.025, na.rm = TRUE, names = FALSE),
      Upper   = stats::quantile(.data$value, 0.975, na.rm = TRUE, names = FALSE),
      Splits  = sum(is.finite(.data$value)),
      .groups = "drop"
    ) %>%
    dplyr::rename(Metric = "metric")

  aif360_data <- final_results %>%
    dplyr::select(subjectid, true_label = true, predicted_prob = prob, sensitive_attr_numeric = group)

  if (isTRUE(save_outputs)) {
    readr::write_csv(final_results, file.path(output_dir, "gnn_final_predictions.csv"))
    readr::write_csv(gate_weights,  file.path(output_dir, "gnn_gate_weights.csv"))
    readr::write_csv(attribution,   file.path(output_dir, "gnn_attribution.csv"))
    readr::write_csv(aif360_data,   file.path(output_dir, "gnn_aif360_data.csv"))
    if (isTRUE(early_stopping)) {
      readr::write_csv(early_stopping_record, file.path(output_dir, "gnn_early_stopping.csv"))
    }
    if (isTRUE(run_tuning)) readr::write_csv(tuning_results, file.path(output_dir, "gnn_tuning_results.csv"))
    saveRDS(expert_weights_list, file.path(output_dir, "gnn_expert_weights.rds"))
  }

  lam_max_final <- if (is.null(lambda_max)) final_lambda else lambda_max
  list(
    final_results = final_results,
    gate_weights = gate_weights,
    attribution = attribution,
    expert_weights = expert_weights_list,
    performance_summary = performance_summary,
    performance_by_split = per_split,
    early_stopping = if (isTRUE(early_stopping)) early_stopping_record else "Early stopping was disabled",
    aif360_data = aif360_data,
    tuning_results = if (isTRUE(run_tuning)) tuning_results else "Tuning was skipped",
    settings = list(family = family, n_strata = if (identical(family, "gaussian")) n_strata else 2L,
                    standardise = standardise, impute = impute,
                    num_repeats = num_repeats, epochs = epochs,
                    supplied_splits = !is.null(splits),
                    penalty = penalty,
                    alpha = alpha,
                    lambda_max = lam_max_final,
                    lambda_ladder = fairgate_lambda_ladder(num_groups, lam_max_final),
                    ladder_active = alpha > 0,
                    early_stopping = early_stopping,
                    es_prop = if (isTRUE(early_stopping)) es_prop else NA_real_,
                    es_min_epochs = if (isTRUE(early_stopping)) es_min_epochs else NA_integer_,
                    es_patience = if (isTRUE(early_stopping) && !is.null(es_patience)) es_patience else NA_integer_,
                    coupling = coupling,
                    base_supplied = coupled,
                    gate_outputs = num_groups + if (base_routable) 1L else 0L,
                    num_groups = num_groups,
                    strata_realised = strata_seen,
                    v3 = list(gate = gate, intervention_cost = intervention_cost,
                              lambda_thr = lambda_thr, thr_temp = thr_temp,
                              thr_target = thr_target, thr_agg = thr_agg,
                              reference_group = reference_group,
                              gate_signal = !is.null(gate_signal), balance = balance,
                              crossed = !is.null(crossed_groups), adversary = adversary,
                              es_smooth = if (is.null(es_smooth)) NA_integer_ else es_smooth,
                              interval_reps = interval_reps, interval_epochs = interval_epochs,
                              interval_level = interval_level, v3_objective = v3_obj),
                    best_params = best_params)
  )
}
