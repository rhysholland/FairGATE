# ------------------------------------------------------------------
# One-call wrapper: cross-fitted base learner, disparity signal, the
# configuration search under the selection rule of the accompanying study,
# and the final fairGATE correction with per-person adjustment intervals.
# Every fit goes through train_gnn() with the same arguments as the study's
# runner, so supplying the same base scores reproduces its fits.
# ------------------------------------------------------------------

#' fairGATE Search Space and Fixed Settings
#'
#' The configuration space searched by [fairgate()] and the settings held
#' fixed, as used in the accompanying study.
#'
#' @return A list with `space` (candidate values: learning rate `lr`, hidden
#'   width `hidden_dim`, `dropout_rate`, gate `temperature`, fairness weight
#'   `lambda`, `penalty` type, thresholded-penalty weight `lambda_thr`,
#'   `intervention_cost` and `coupling`), `fixed` (gate, routing balance,
#'   adversary, smoothing window and earliest selectable epoch of early
#'   stopping, threshold temperature, target and aggregation, whether the
#'   disparity signal and crossed groups are used), `n_draws` (48 distinct
#'   random configurations) and `seed` (4242; draws use the Mersenne-Twister
#'   generator with Inversion and Rejection sampling, `lambda_thr` is set to 0
#'   when `penalty = "thresholded"`, and duplicate draws are rejected).
#' @examples
#' str(fairgate_space()$space)
#' @export
fairgate_space <- function() {
  list(space = list(lr = c(0.001, 0.002), hidden_dim = c(32L, 64L, 128L),
                    dropout_rate = c(0.1, 0.2), temperature = c(0.5, 1),
                    lambda = c(1L, 4L, 16L, 64L),
                    penalty = c("random_effects", "variance", "thresholded"),
                    lambda_thr = c(0L, 4L, 16L), intervention_cost = c(0, 0.02, 0.1),
                    coupling = c("residual", "expert")),
       fixed = list(gate = "softmax", balance = 0L, adversary = 0L, es_smooth = 5L,
                    es_min_epochs = 10L, thr_temp = 0.05, thr_target = "eo", thr_agg = "max",
                    signal = TRUE, crossed = FALSE),
       n_draws = 48L, seed = 4242L)
}

#' Draw fairGATE Configurations
#'
#' Draws distinct random configurations from a search space with the
#' generator settings of the accompanying study.
#'
#' @param space Named list of candidate values, as `fairgate_space()$space`.
#' @param n Number of distinct configurations.
#' @param seed Integer seed.
#' @return A list of configurations (named lists). With the defaults of
#'   [fairgate_space()] this is the configuration list searched in the study.
#' @examples
#' g <- fairgate_draw_configs(fairgate_space()$space, 3)
#' str(g[[1]])
#' @export
fairgate_draw_configs <- function(space, n = 48L, seed = 4242L) {
  .fg_with_seed(seed, {
    seen <- character(0); out <- list()
    while (length(out) < n && length(seen) < 5000L) {
      cfg <- lapply(space, function(v) v[sample.int(length(v), 1L)])
      if (identical(cfg$penalty, "thresholded")) cfg$lambda_thr <- 0
      sig <- paste(unlist(cfg), collapse = "|")
      if (sig %in% seen) { seen <- c(seen, sig); next }
      seen <- c(seen, sig); out[[length(out) + 1L]] <- cfg
    }
    out
  })
}

# Groups eligible for the worst-group maximum (minimum-cell rule): at least
# 10 expected positives and 10 expected negatives per held-out set (binary)
# or 50 expected people (continuous), from full-cohort counts times the
# held-out share.
.fg_eligible <- function(y, g, ref, is_bin, share) {
  ug <- sort(unique(g))
  ok <- vapply(ug, function(k) {
    if (is_bin) sum(g == k & y == 1) * share >= 10 && sum(g == k & y == 0) * share >= 10
    else sum(g == k) * share >= 50
  }, logical(1))
  setdiff(ug[ok], ref)
}

.fg_score <- function(yv, pv, gv, ref, is_bin, prev, sd_y, elig) {
  if (is_bin) {
    thr <- as.numeric(stats::quantile(pv, 1 - prev, type = 7, names = FALSE))
    m <- fairgate_group_disparity(yv, pv, gv, ref, "binomial", threshold = thr, eligible = elig)
    perf <- .fg_auc(yv, pv)
  } else {
    ed <- stats::quantile(yv, probs = seq(0, 1, length.out = 6L), names = FALSE)
    m <- fairgate_group_disparity(yv, pv, gv, ref, "gaussian", edges = ed, scale = sd_y, eligible = elig)
    perf <- .fg_r2(yv, pv)
  }
  list(perf = perf, worst = m$worst, worst_group = m$worst_group, pooled = m$pooled)
}

.fg_make_splits <- function(y, family, n_partitions, test_prop, seed) {
  lapply(seq_len(n_partitions), function(r) {
    sp <- .fg_inner_split(seq_along(y), y, family, test_prop, seed + r)
    list(train = sp$train, test = sp$valid)
  })
}

#' Fit fairGATE in One Call
#'
#' Fits the cross-fitted base learner (XGBoost or a neural network), the
#' cross-fitted disparity signal, searches the fairGATE configuration space on
#' an inner split under the selection rule of the accompanying study, and fits
#' the selected configuration on every partition with per-person adjustment
#' intervals. The defaults reproduce the fairGATE protocol of the study.
#'
#' @param X Numeric feature matrix or data frame, without the protected
#'   attribute; or the list returned by [prepare_data()], in which case `y`
#'   and `group` are taken from it.
#' @param y Outcome: 0/1 (`family = "binomial"`) or numeric (`"gaussian"`).
#' @param group Integer codes of the protected attribute. Used only in the
#'   fairness objective, in the target of the disparity signal and in
#'   evaluation; never as an input to the base learner, the gate or the
#'   experts.
#' @param family `"binomial"` or `"gaussian"`.
#' @param reference_group Code of the reference (privileged) group. Defaults
#'   to the largest group.
#' @param splits Optional list of partitions, each `list(train =, test =)` of
#'   row indices. When `NULL`, `n_partitions` stratified partitions with
#'   held-out share `test_prop` are drawn.
#' @param n_partitions,test_prop Number of partitions and held-out share used
#'   when `splits` is `NULL`. Defaults 1 and 0.2. `test_prop` is also the
#'   nominal held-out share of the minimum-cell rule (see Details), so set it
#'   to the held-out share of `splits` when supplying your own partitions.
#' @param inner Optional `list(train =, valid =)` of row indices for the
#'   configuration search. Defaults to a stratified 75:25 split of the
#'   training rows of the first partition.
#' @param base_learner `"xgboost"` (default, as in the study) or `"nn"`.
#' @param base_params Optional fixed base-learner hyperparameters (see
#'   [fairgate_base_learner()]); when `NULL` they are tuned on `inner`.
#' @param base_tune_args Optional list of further arguments to
#'   [fairgate_tune_base_learner()].
#' @param base_score Optional precomputed base scores: a list with element
#'   `inner` (`list(train =, test =)` for `inner$train` and `inner$valid`)
#'   and element `partitions` (a list aligned with `splits`), on the response
#'   scale, training rows out of fold. When supplied, no base learner is
#'   fitted.
#' @param signal Logical; use the cross-fitted disparity signal in the gate.
#'   Defaults to the `fixed` setting of `space` (`TRUE`).
#' @param signal_learner Learner for the disparity signal, `"xgboost"`
#'   (default, as in the study) or `"nn"`.
#' @param signal_params Hyperparameters of the signal learner; default 200
#'   rounds, learning rate 0.05, depth 4, minimum child weight 5, subsample
#'   and column subsample 0.8, L2 penalty 1.
#' @param gate_signal Optional precomputed signal in the layout of
#'   `base_score`; overrides `signal_learner`.
#' @param params Optional fixed fairGATE configuration (a named list with the
#'   elements of `fairgate_space()$space`); when supplied, no search is run.
#' @param space Search space and fixed settings, default [fairgate_space()].
#' @param n_draws Number of configurations searched. Default 48.
#' @param perf_tolerance Selection rule: among configurations whose
#'   inner-validation AUC or R-squared is within this distance of the best,
#'   the one with the smallest worst-group absolute disparity (average odds
#'   difference at the prevalence-matched threshold, or conditional mean
#'   prediction difference in outcome standard deviations). Default 0.01.
#' @param crossed_groups Optional two-column matrix of two protected
#'   attributes for the crossed penalty (used only when `space$fixed$crossed`
#'   is `TRUE` and the penalty is a variance component).
#' @param epochs,es_prop,es_patience Epoch cap, early-stopping slice and
#'   patience of [train_gnn()]. Defaults 300, 0.15 and 30.
#' @param interval_reps,interval_epochs Bootstrap replicates and epochs per
#'   replicate of the per-person adjustment intervals. Defaults 10 and 10;
#'   `interval_reps = 0` switches the intervals off.
#' @param seed Integer seed. Configuration `j` of the search is fitted with
#'   `seed + j`, final fits with `seed`, the base learner and signal of
#'   partition `r` with `seed + r` (the inner split with `seed`).
#' @param nthread Threads used by XGBoost. Default 1.
#' @param verbose Logical; print progress.
#' @details
#' **Base learner.** Fitted without the protected attribute by
#' [fairgate_base_learner()]: 5-fold out-of-fold scores for training rows, a
#' fit on all training rows for held-out rows.
#'
#' **Disparity signal.** For training rows a target is computed by
#' [fairgate_disparity_target()] from the outcome, the protected group and
#' the out-of-fold base score; a regression on the features and the
#' link-scale base score predicts it, cross-fitted in the same way, and the
#' prediction is given to the gate only.
#'
#' **Search and selection.** `n_draws` configurations drawn by
#' [fairgate_draw_configs()] are fitted on `inner` and scored on its
#' validation rows; the selection rule is described under `perf_tolerance`,
#' with the worst group taken over groups passing a minimum-cell rule (at
#' least 10 expected positives and negatives, or 50 expected people, per
#' held-out set, computed as full-sample counts times `test_prop`). When no
#' group passes the rule (small samples), the worst group is taken over all
#' non-reference groups, with a warning.
#'
#' **Final fit.** The selected configuration is fitted on every partition
#' by [train_gnn()] with median imputation, standardisation, early stopping
#' on a stratified 15 per cent slice (smoothed over 5 epochs, earliest
#' selectable epoch 10, patience 30, cap 300) and per-person adjustment
#' intervals from warm-started bootstrap replicates.
#'
#' The full protocol is computationally heavy (48 searched configurations of
#' up to 300 epochs each). For a quick look reduce `n_draws`, `epochs` and
#' `interval_reps`.
#' @return An object of class `fairgate_fit`: a list with `predictions` (a
#'   data frame with `partition`, `row`, `y`, `group`, `base`, `pred` and
#'   `adjustment` for every held-out row), `export` (the per-person export:
#'   the routing weights, base weight, adjustment and, when intervals are on,
#'   `adj_lo`, `adj_hi`, `adj_sd` and `interval_excludes_zero`), `selected`
#'   (the selected configuration with its fixed settings), `search` (one row
#'   per searched configuration with inner-validation performance and
#'   worst-group disparity), `base` (base-learner scores and settings),
#'   `signal`, `fit` (the [train_gnn()] result) and `settings`.
#' @examples
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed() &&
#'     requireNamespace("xgboost", quietly = TRUE) &&
#'     utils::packageVersion("xgboost") >= "3.1.0") {
#'   set.seed(1)
#'   n <- 800
#'   g <- rbinom(n, 1, 0.3)
#'   X <- cbind(x1 = rnorm(n) + 0.5 * g, x2 = rnorm(n), x3 = rnorm(n))
#'   y <- rbinom(n, 1, stats::plogis(X[, 1] - 0.5 * X[, 2]))
#'   fg <- fairgate(X, y, g, "binomial", reference_group = 0,
#'                  base_params = list(nrounds = 50, max_depth = 3),
#'                  n_draws = 2, epochs = 20, interval_reps = 2,
#'                  interval_epochs = 2)
#'   fg
#'   head(fg$export)
#' }
#' }
#' @export
fairgate <- function(X, y = NULL, group = NULL, family = c("binomial", "gaussian"),
                     reference_group = NULL, splits = NULL, n_partitions = 1L, test_prop = 0.2,
                     inner = NULL, base_learner = c("xgboost", "nn"), base_params = NULL,
                     base_tune_args = NULL, base_score = NULL,
                     signal = NULL, signal_learner = c("xgboost", "nn"), signal_params = NULL,
                     gate_signal = NULL, params = NULL, space = fairgate_space(),
                     n_draws = space$n_draws %||% 48L, perf_tolerance = 0.01,
                     crossed_groups = NULL, epochs = 300L, es_prop = 0.15, es_patience = 30L,
                     interval_reps = 10L, interval_epochs = 10L, seed = 4242L,
                     nthread = 1L, verbose = FALSE) {
  family <- match.arg(family); base_learner <- match.arg(base_learner)
  signal_learner <- match.arg(signal_learner)
  .require_torch()
  if (is.list(X) && !is.data.frame(X) && all(c("X", "y", "group") %in% names(X))) {
    y <- X$y; group <- X$group; X <- X$X
  }
  if (is.null(y) || is.null(group)) stop("Supply `y` and `group` (or the list from prepare_data()).", call. = FALSE)
  xy <- .fg_check_xy(X, y, family); X <- xy$X; y <- xy$y
  if (is.null(colnames(X))) colnames(X) <- paste0("x", seq_len(ncol(X)))
  g <- as.integer(group)
  if (length(g) != nrow(X) || anyNA(g)) stop("`group` must be complete, one code per row of `X`.", call. = FALSE)
  if (anyNA(y)) stop("`y` must not contain missing values.", call. = FALSE)
  is_bin <- identical(family, "binomial")
  if (is.null(reference_group)) reference_group <- as.integer(names(which.max(table(g))))
  ref <- as.integer(reference_group)
  if (!ref %in% g) stop("`reference_group` is not one of the codes in `group`.", call. = FALSE)
  if (is.null(signal)) signal <- isTRUE(space$fixed$signal)

  if (is.null(splits)) splits <- .fg_make_splits(y, family, as.integer(n_partitions), test_prop, seed)
  splits <- lapply(splits, function(s) list(train = as.integer(s$train), test = as.integer(s$test)))
  if (is.null(inner)) {
    inner <- .fg_inner_split(splits[[1]]$train, y, family, 0.25, seed)
  }
  inner <- list(train = as.integer(inner$train), valid = as.integer(inner$valid))
  R <- length(splits)
  say <- function(...) if (isTRUE(verbose)) message(sprintf(...))

  # ---- base learner -------------------------------------------------
  base_info <- NULL
  if (is.null(base_score)) {
    if (is.null(base_params)) {
      say("Tuning the %s base learner on the inner split", base_learner)
      ta <- c(list(X = X, y = y, family = family, train = inner$train, valid = inner$valid,
                   base_learner = base_learner, seed = seed, nthread = nthread, verbose = verbose),
              base_tune_args %||% list())
      base_tuning <- do.call(fairgate_tune_base_learner, ta)
      base_params <- base_tuning$best
    } else base_tuning <- NULL
    fitb <- function(tr, te, s) {
      b <- fairgate_base_learner(X, y, family, train = tr, test = te, base_learner = base_learner,
                                 params = base_params, seed = s, nthread = nthread)
      list(train = b$train, test = b$test, perf = b$perf)
    }
    say("Cross-fitting the base learner")
    b_inner <- fitb(inner$train, inner$valid, seed)
    b_parts <- lapply(seq_len(R), function(r) fitb(splits[[r]]$train, splits[[r]]$test, seed + r))
    base_info <- list(base_learner = base_learner, params = base_params, tuning = base_tuning,
                      perf = do.call(rbind, lapply(b_parts, function(b) b$perf)))
  } else {
    b_inner <- base_score$inner; b_parts <- base_score$partitions
    if (length(b_parts) != R) stop("`base_score$partitions` must be aligned with `splits`.", call. = FALSE)
  }
  bs_inner <- list(list(train = b_inner$train, test = b_inner$test))
  bs_parts <- lapply(b_parts, function(b) list(train = b$train, test = b$test))

  # ---- disparity signal ---------------------------------------------
  sg_inner <- NULL; sg_parts <- NULL
  if (isTRUE(signal)) {
    if (is.null(gate_signal)) {
      if (is.null(signal_params)) {
        signal_params <- if (identical(signal_learner, "xgboost")) {
          list(nrounds = 200L, eta = 0.05, max_depth = 4L, min_child_weight = 5,
               subsample = 0.8, colsample_bytree = 0.8, lambda = 1, alpha = 0)
        } else list(n_layers = 2L, width = 32L, dropout = 0.1, epochs = 30L)
      }
      fits <- function(tr, te, b, s) {
        target <- fairgate_disparity_target(y[tr], g[tr], b$train, ref, family)
        lk <- numeric(nrow(X))
        lk[tr] <- .fg_link(b$train, family, y[tr])
        lk[te] <- .fg_link(b$test, family, y[tr])
        yy <- numeric(nrow(X)); yy[tr] <- target
        sf <- fairgate_base_learner(cbind(X, base_link = lk), yy, "gaussian", train = tr, test = te,
                                    base_learner = signal_learner, params = signal_params, seed = s,
                                    nthread = nthread)
        list(train = sf$train, test = sf$test)
      }
      say("Cross-fitting the disparity signal")
      sg_inner <- list(fits(inner$train, inner$valid, b_inner, seed))
      sg_parts <- lapply(seq_len(R), function(r) fits(splits[[r]]$train, splits[[r]]$test, b_parts[[r]], seed + r))
    } else {
      sg_inner <- list(gate_signal$inner); sg_parts <- gate_signal$partitions
    }
  }

  n_strata <- if (is_bin) 2L else 5L
  pd <- list(X = X, y = y, group = g, feature_names = colnames(X), subject_ids = seq_len(nrow(X)))
  share <- test_prop
  elig <- .fg_eligible(y, g, ref, is_bin, share)
  if (!length(elig) && is.null(params)) {
    warning("No non-reference group passes the minimum-cell rule; the worst group is taken over all groups.",
            call. = FALSE)
    elig <- NULL
  }
  sd_y <- stats::sd(y)
  crossed_use <- isTRUE(space$fixed$crossed) && !is.null(crossed_groups)

  fit_cfg <- function(s, sp, bs, sg, sd, reps) {
    args <- list(pd, run_tuning = FALSE,
                 best_params = list(lr = s$lr, hidden_dim = as.integer(s$hidden_dim),
                                    dropout_rate = s$dropout_rate, lambda = s$lambda,
                                    temperature = s$temperature),
                 epochs = as.integer(epochs), seed = sd, splits = sp, family = family, n_strata = n_strata,
                 standardise = TRUE, impute = "median", penalty = s$penalty, alpha = 0,
                 early_stopping = TRUE, es_prop = es_prop, es_patience = as.integer(es_patience),
                 es_min_epochs = as.integer(s$es_min_epochs),
                 base_score = bs, coupling = s$coupling,
                 gate = s$gate, intervention_cost = s$intervention_cost,
                 lambda_thr = s$lambda_thr, thr_temp = s$thr_temp,
                 thr_target = s$thr_target, thr_agg = s$thr_agg, reference_group = ref,
                 balance = s$balance, adversary = s$adversary,
                 interval_reps = as.integer(reps), interval_epochs = as.integer(interval_epochs))
    if (!is.null(s$es_smooth) && !is.na(s$es_smooth)) args$es_smooth <- as.integer(s$es_smooth)
    if (isTRUE(s$signal) && !is.null(sg)) args$gate_signal <- sg
    if (isTRUE(s$crossed) && crossed_use && !identical(s$penalty, "thresholded")) {
      args$crossed_groups <- crossed_groups
    }
    do.call(train_gnn, args)
  }

  # ---- configuration search -------------------------------------------
  search <- NULL
  if (is.null(params)) {
    grid <- fairgate_draw_configs(space$space, as.integer(n_draws), space$seed %||% seed)
    iv <- inner$valid
    prev_inner <- if (is_bin) mean(y[inner$train]) else NA_real_
    rows <- vector("list", length(grid))
    for (j in seq_along(grid)) {
      s <- c(grid[[j]], space$fixed); s$signal <- isTRUE(signal)
      t0 <- proc.time()[["elapsed"]]
      fit <- try(fit_cfg(s, list(list(train = inner$train, test = inner$valid)), bs_inner, sg_inner,
                         seed + j, 0L), silent = TRUE)
      el <- proc.time()[["elapsed"]] - t0
      if (inherits(fit, "try-error")) {
        rows[[j]] <- data.frame(config_id = j, perf = NA_real_, worst = NA_real_, worst_group = NA_integer_,
                                pooled = NA_real_, secs = el, error = as.character(fit))
        say("config %d/%d failed: %s", j, length(grid), as.character(fit))
        next
      }
      at <- fit$attribution
      sc <- .fg_score(y[iv], fit$final_results$prob, g[iv], ref, is_bin, prev_inner, sd_y, elig)
      rows[[j]] <- data.frame(config_id = j, perf = sc$perf, worst = sc$worst, worst_group = sc$worst_group,
                              pooled = sc$pooled, secs = el, error = NA_character_)
      say("config %d/%d: perf %.4f, worst-group disparity %.4f (%.1f s)", j, length(grid), sc$perf, sc$worst, el)
      rm(fit)
    }
    gdf <- do.call(rbind, lapply(seq_along(grid), function(j) data.frame(config_id = j, as.data.frame(grid[[j]], stringsAsFactors = FALSE))))
    search <- merge(gdf, do.call(rbind, rows), by = "config_id", sort = TRUE)
    ok <- which(is.finite(search$perf) & is.finite(search$worst))
    if (!length(ok)) {
      err <- search$error[!is.na(search$error)]
      stop(sprintf("Every searched configuration failed. First error: %s",
                   if (length(err)) trimws(err[1]) else "non-finite validation scores"), call. = FALSE)
    }
    band <- ok[search$perf[ok] >= max(search$perf[ok]) - perf_tolerance]
    best_i <- band[which.min(search$worst[band])]
    search$selected <- seq_len(nrow(search)) == best_i
    params <- grid[[search$config_id[best_i]]]
  }
  sel <- c(params, space$fixed[setdiff(names(space$fixed), names(params))])
  sel$signal <- isTRUE(signal)

  # ---- final fit --------------------------------------------------------
  say("Fitting the selected configuration on %d partition(s)", R)
  fit <- fit_cfg(sel, splits, bs_parts, sg_parts, seed, interval_reps)
  fr <- fit$final_results
  at <- as.data.frame(fit$attribution)
  names(at)[names(at) == "iteration"] <- "partition"
  names(at)[names(at) == "subjectid"] <- "row"
  if (all(c("adj_lo", "adj_hi") %in% names(at))) {
    at$interval_excludes_zero <- at$adj_lo > 0 | at$adj_hi < 0
  }
  preds <- data.frame(partition = fr$iteration, row = as.integer(fr$subjectid), y = y[as.integer(fr$subjectid)],
                      group = g[as.integer(fr$subjectid)], base = at$base, pred = fr$prob,
                      adjustment = at$adjustment)
  out <- list(predictions = preds, export = at, selected = sel, search = search,
              base = list(info = base_info, inner = b_inner, partitions = b_parts),
              signal = list(inner = sg_inner, partitions = sg_parts), fit = fit,
              settings = list(family = family, reference_group = ref, eligible_groups = elig,
                              splits = splits, inner = inner, n_draws = if (is.null(search)) 0L else nrow(search),
                              perf_tolerance = perf_tolerance, epochs = epochs, es_prop = es_prop,
                              es_patience = es_patience, interval_reps = interval_reps,
                              interval_epochs = interval_epochs, seed = seed))
  class(out) <- "fairgate_fit"
  out
}

#' @export
print.fairgate_fit <- function(x, ...) {
  s <- x$settings
  cat(sprintf("fairGATE fit (%s outcome), %d partition(s), reference group %s\n",
              if (identical(s$family, "binomial")) "binary" else "continuous", length(s$splits), s$reference_group))
  if (!is.null(x$base$info)) {
    cat(sprintf("  base learner: %s\n", if (identical(x$base$info$base_learner, "nn")) "neural network" else "XGBoost"))
  }
  if (!is.null(x$search)) cat(sprintf("  configurations searched: %d\n", nrow(x$search)))
  p <- x$selected
  cat(sprintf("  selected: coupling %s, penalty %s, lambda %s, lambda_thr %s, intervention cost %s\n",
              p$coupling, p$penalty, p$lambda, p$lambda_thr, p$intervention_cost))
  pr <- x$predictions
  pf <- tapply(seq_len(nrow(pr)), pr$partition, function(i) {
    if (identical(s$family, "binomial")) c(.fg_auc(pr$y[i], pr$base[i]), .fg_auc(pr$y[i], pr$pred[i]))
    else c(.fg_r2(pr$y[i], pr$base[i]), .fg_r2(pr$y[i], pr$pred[i]))
  })
  pf <- do.call(rbind, pf)
  cat(sprintf("  held-out %s: base %.4f, fairGATE %.4f (mean over partitions)\n",
              if (identical(s$family, "binomial")) "AUC" else "R-squared", mean(pf[, 1]), mean(pf[, 2])))
  cat(sprintf("  mean |adjustment|: %.4f\n", mean(abs(pr$adjustment))))
  invisible(x)
}

#' Per-Person Export of a fairGATE Fit
#'
#' Returns, and optionally writes, the per-person table of a [fairgate()] fit:
#' partition, row, group, outcome, base score, fairGATE prediction,
#' adjustment, base weight, routing weights and, when intervals were fitted,
#' the adjustment interval (`adj_lo`, `adj_hi`) and whether it excludes zero.
#'
#' @param x A `fairgate_fit` object.
#' @param file Optional path of a CSV file to write. Nothing is written when
#'   `NULL` (default).
#' @return A data frame, invisibly when `file` is given.
#' @examples
#' \donttest{
#' # fg <- fairgate(...)
#' # tab <- fairgate_export(fg)
#' # fairgate_export(fg, file.path(tempdir(), "fairgate_export.csv"))
#' }
#' @export
fairgate_export <- function(x, file = NULL) {
  if (!inherits(x, "fairgate_fit")) stop("`x` must be a fairgate_fit object.", call. = FALSE)
  at <- x$export
  key <- paste(x$predictions$partition, x$predictions$row)
  m <- match(paste(at$partition, at$row), key)
  at$y <- x$predictions$y[m]
  at$group <- x$predictions$group[m]
  front <- intersect(c("partition", "row", "group", "y", "base", "pred", "adjustment", "base_weight",
                       "adj_lo", "adj_hi", "adj_sd", "interval_excludes_zero"), names(at))
  at <- at[, c(front, setdiff(names(at), front)), drop = FALSE]
  # the ladder quantities are not defined for the coupled models; drop them when empty
  for (nc in c("f_anchor", "delta_fair", "correction_intensity")) {
    if (nc %in% names(at) && all(is.na(at[[nc]]))) at[[nc]] <- NULL
  }
  if (!is.null(file)) {
    utils::write.csv(at, file, row.names = FALSE)
    return(invisible(at))
  }
  at
}
