# ------------------------------------------------------------------
# Cross-fitted base learners: XGBoost or a torch multilayer perceptron.
#
# The base learner is fitted without the protected attribute. Training rows
# receive K-fold out-of-fold scores and held-out rows the prediction of a fit
# on all training rows, so that no training row is scored by a model that saw
# its outcome. The scores are then passed to train_gnn(base_score = ).
# ------------------------------------------------------------------

`%||%` <- function(a, b) if (is.null(a)) b else a

# Run `expr` under a fixed seed and restore the caller's random number state
# afterwards, so that the base learner never alters the user's RNG stream.
.fg_with_seed <- function(seed, expr) {
  if (is.null(seed)) return(expr)
  genv <- globalenv()
  had <- exists(".Random.seed", envir = genv, inherits = FALSE)
  old <- if (had) get(".Random.seed", envir = genv, inherits = FALSE) else NULL
  on.exit({
    if (had) assign(".Random.seed", old, envir = genv)
    else if (exists(".Random.seed", envir = genv, inherits = FALSE)) rm(".Random.seed", envir = genv)
  }, add = TRUE)
  set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  expr
}

# Median imputation and standardisation fitted on the training rows only:
# missing values take the training median (0 if a column is entirely
# missing), columns are centred by the training mean and scaled by the
# training standard deviation with divisor n; columns with standard deviation
# at most 1e-12 are centred only.
.fg_prep <- function(Xtr, Xte = NULL, standardise = TRUE) {
  Xtr <- as.matrix(Xtr); storage.mode(Xtr) <- "double"
  if (!is.null(Xte)) { Xte <- as.matrix(Xte); storage.mode(Xte) <- "double" }
  if (anyNA(Xtr) || (!is.null(Xte) && anyNA(Xte))) {
    med <- apply(Xtr, 2, stats::median, na.rm = TRUE)
    med[!is.finite(med)] <- 0
    for (j in seq_len(ncol(Xtr))) {
      Xtr[is.na(Xtr[, j]), j] <- med[j]
      if (!is.null(Xte)) Xte[is.na(Xte[, j]), j] <- med[j]
    }
  }
  if (isTRUE(standardise)) {
    mu <- colMeans(Xtr)
    sdv <- sqrt(colMeans(sweep(Xtr, 2, mu)^2))
    sdv[!is.finite(sdv) | sdv <= 1e-12] <- 1
    Xtr <- sweep(sweep(Xtr, 2, mu), 2, sdv, "/")
    if (!is.null(Xte)) Xte <- sweep(sweep(Xte, 2, mu), 2, sdv, "/")
  }
  list(train = Xtr, test = Xte)
}

# Fold labels 1..n_folds for the training rows: stratified on the outcome
# class for a binary outcome, a random permutation otherwise.
.fg_folds <- function(y, family, n_folds, seed) {
  n <- length(y)
  .fg_with_seed(seed, {
    f <- integer(n)
    if (identical(family, "binomial")) {
      for (cl in sort(unique(y))) {
        i <- which(y == cl)
        i <- i[sample.int(length(i))]
        f[i] <- rep_len(seq_len(n_folds), length(i))
      }
    } else {
      i <- sample.int(n)
      f[i] <- rep_len(seq_len(n_folds), n)
    }
    f
  })
}

# A stratified two-way split of `rows` (validation share `prop`).
.fg_inner_split <- function(rows, y, family, prop, seed) {
  yr <- y[rows]
  v <- .fg_with_seed(seed, {
    if (identical(family, "binomial")) {
      unlist(lapply(sort(unique(yr)), function(cl) {
        i <- which(yr == cl)
        i[sample.int(length(i), max(1L, round(prop * length(i))))]
      }))
    } else {
      sample.int(length(rows), max(1L, round(prop * length(rows))))
    }
  })
  v <- sort(v)
  list(train = rows[-v], valid = rows[v])
}

.fg_auc <- function(y, s) {
  pos <- s[y == 1]; neg <- s[y == 0]
  if (!length(pos) || !length(neg)) return(NA_real_)
  r <- rank(c(pos, neg))
  (sum(r[seq_along(pos)]) - length(pos) * (length(pos) + 1) / 2) / (length(pos) * length(neg))
}
.fg_r2 <- function(y, p) {
  ss <- sum((y - mean(y))^2)
  if (ss > 0) 1 - sum((y - p)^2) / ss else NA_real_
}
.fg_logloss <- function(y, p) {
  p <- pmin(pmax(p, 1e-15), 1 - 1e-15)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}
.fg_perf <- function(y, p, family) if (identical(family, "binomial")) .fg_auc(y, p) else .fg_r2(y, p)
.fg_loss <- function(y, p, family) if (identical(family, "binomial")) .fg_logloss(y, p) else mean((y - p)^2)

.fg_check_xgboost <- function() {
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    stop("base_learner = 'xgboost' requires the 'xgboost' package (>= 3.1.0). Install it with install.packages('xgboost').",
         call. = FALSE)
  }
  if (utils::packageVersion("xgboost") < "3.1.0") {
    stop("base_learner = 'xgboost' requires 'xgboost' >= 3.1.0.", call. = FALSE)
  }
  invisible(TRUE)
}

# ---- hyperparameters ------------------------------------------------

.fg_base_defaults <- function(base_learner) {
  if (identical(base_learner, "xgboost")) {
    list(nrounds = 500L, eta = 0.05, max_depth = 4L, min_child_weight = 1,
         subsample = 0.8, colsample_bytree = 0.8, lambda = 1, alpha = 0,
         early_stopping_rounds = 50L)
  } else {
    list(n_layers = 2L, width = 64L, dropout = 0.1, lr = 1e-3, weight_decay = 1e-4,
         batch_size = 512L, epochs = 100L, patience = 20L)
  }
}

#' Default Hyperparameter Grids for the Base Learner
#'
#' Returns the candidate values searched by [fairgate_tune_base_learner()] for
#' each base learner. Every element is a vector of candidate values; a single
#' value is held fixed.
#'
#' @param base_learner `"xgboost"` or `"nn"`.
#' @return A named list of candidate values.
#' @details
#' **XGBoost** (`"xgboost"`, gradient-boosted trees with the histogram method):
#' \describe{
#'   \item{`max_depth`}{tree depth, 3, 4, 6 or 8}
#'   \item{`eta`}{learning rate, 0.02, 0.05 or 0.1}
#'   \item{`nrounds`}{ceiling on boosting rounds, 1000; the number actually
#'     used is chosen by early stopping on the inner validation slice}
#'   \item{`early_stopping_rounds`}{50 rounds without improvement in
#'     validation loss}
#'   \item{`subsample`}{row subsampling per tree, 0.7, 0.9 or 1}
#'   \item{`colsample_bytree`}{column subsampling per tree, 0.7, 0.9 or 1}
#'   \item{`min_child_weight`}{minimum hessian weight per leaf, 1, 5 or 10}
#'   \item{`lambda`, `alpha`}{L2 penalty 0.5, 1 or 5 and L1 penalty 0 or 0.5
#'     on leaf weights}
#' }
#' These are the values of the gradient-boosting comparator in the
#' accompanying study, with the number of rounds chosen by early stopping
#' rather than drawn from a list.
#'
#' **Neural network** (`"nn"`, a torch multilayer perceptron with rectified
#' linear units and dropout, trained by Adam on mini-batches):
#' \describe{
#'   \item{`n_layers`}{hidden layers, 1, 2 or 3}
#'   \item{`width`}{units per hidden layer, 32, 64, 128 or 256}
#'   \item{`dropout`}{dropout rate, 0, 0.1, 0.2 or 0.3}
#'   \item{`lr`}{Adam learning rate, 3e-4, 1e-3 or 3e-3}
#'   \item{`weight_decay`}{Adam weight decay, 0, 1e-5, 1e-4 or 1e-3}
#'   \item{`batch_size`}{mini-batch size, 256, 512 or 1024}
#'   \item{`epochs`}{ceiling on epochs, 200; the number actually used is
#'     chosen by early stopping on the inner validation slice}
#'   \item{`patience`}{20 epochs without improvement in validation loss}
#' }
#' The loss is binary cross-entropy on the logit (binary outcome) or squared
#' error on the standardised outcome (continuous outcome).
#' @examples
#' str(fairgate_base_grid("xgboost"))
#' str(fairgate_base_grid("nn"))
#' @export
fairgate_base_grid <- function(base_learner = c("xgboost", "nn")) {
  base_learner <- match.arg(base_learner)
  if (identical(base_learner, "xgboost")) {
    list(max_depth = c(3L, 4L, 6L, 8L), eta = c(0.02, 0.05, 0.1), nrounds = 1000L,
         early_stopping_rounds = 50L, subsample = c(0.7, 0.9, 1), colsample_bytree = c(0.7, 0.9, 1),
         min_child_weight = c(1, 5, 10), lambda = c(0.5, 1, 5), alpha = c(0, 0.5))
  } else {
    list(n_layers = 1:3, width = c(32L, 64L, 128L, 256L), dropout = c(0, 0.1, 0.2, 0.3),
         lr = c(3e-4, 1e-3, 3e-3), weight_decay = c(0, 1e-5, 1e-4, 1e-3),
         batch_size = c(256L, 512L, 1024L), epochs = 200L, patience = 20L)
  }
}

# Configurations to evaluate: the full Cartesian grid when `search = "grid"`
# or when it has no more than `n_search` points, otherwise `n_search`
# distinct seeded random draws.
.fg_draw_configs <- function(grid, n_search, search, seed) {
  sizes <- vapply(grid, length, 1L)
  full <- prod(sizes)
  if (identical(search, "grid") || full <= n_search) {
    eg <- expand.grid(lapply(grid, seq_along), KEEP.OUT.ATTRS = FALSE)
    return(lapply(seq_len(nrow(eg)), function(r) {
      stats::setNames(lapply(names(grid), function(k) grid[[k]][eg[r, k]]), names(grid))
    }))
  }
  .fg_with_seed(seed, {
    seen <- character(0); out <- list()
    while (length(out) < n_search && length(seen) < 5000L) {
      cfg <- lapply(grid, function(v) v[sample.int(length(v), 1L)])
      sig <- paste(unlist(cfg), collapse = "|")
      if (sig %in% seen) { seen <- c(seen, sig); next }
      seen <- c(seen, sig); out[[length(out) + 1L]] <- cfg
    }
    out
  })
}

# ---- XGBoost ----------------------------------------------------------

.fg_xgb_fit <- function(Xtr, ytr, family, p, seed, Xval = NULL, yval = NULL, nthread = 1L) {
  .fg_check_xgboost()
  bin <- identical(family, "binomial")
  params <- list(objective = if (bin) "binary:logistic" else "reg:squarederror",
                 eval_metric = if (bin) "logloss" else "rmse",
                 tree_method = "hist", nthread = as.integer(nthread), seed = as.integer(seed),
                 eta = p$eta, max_depth = as.integer(p$max_depth),
                 min_child_weight = p$min_child_weight, subsample = p$subsample,
                 colsample_bytree = p$colsample_bytree, lambda = p$lambda, alpha = p$alpha)
  if (!is.null(p$max_bin)) params$max_bin <- as.integer(p$max_bin)
  dtr <- xgboost::xgb.DMatrix(Xtr, label = ytr, nthread = as.integer(nthread))
  if (!is.null(Xval)) {
    dva <- xgboost::xgb.DMatrix(Xval, label = yval, nthread = as.integer(nthread))
    m <- xgboost::xgb.train(params = params, data = dtr, nrounds = as.integer(p$nrounds),
                            evals = list(valid = dva),
                            early_stopping_rounds = as.integer(p$early_stopping_rounds %||% 50L),
                            verbose = 0)
    bi <- xgboost::xgb.attr(m, "best_iteration")
    best <- if (is.null(bi)) as.integer(p$nrounds) else as.integer(bi) + 1L
    pv <- stats::predict(m, Xval, iterationrange = c(1L, best))
    return(list(model = m, best = best, val_pred = as.numeric(pv)))
  }
  m <- xgboost::xgb.train(params = params, data = dtr, nrounds = as.integer(p$nrounds), verbose = 0)
  list(model = m, best = as.integer(p$nrounds))
}

.fg_xgb_predict <- function(fit, X) as.numeric(stats::predict(fit$model, X))

# ---- neural network ---------------------------------------------------

.fg_mlp <- function() {
  torch::nn_module(
    "fairgate_mlp",
    initialize = function(n_in, n_layers, width, dropout) {
      layers <- list(); d <- n_in
      for (l in seq_len(n_layers)) {
        layers <- c(layers, list(torch::nn_linear(d, width), torch::nn_relu(), torch::nn_dropout(dropout)))
        d <- width
      }
      layers <- c(layers, list(torch::nn_linear(d, 1L)))
      self$net <- do.call(torch::nn_sequential, layers)
    },
    forward = function(x) self$net(x)$squeeze(2L)
  )
}

.fg_nn_raw <- function(net, X) {
  net$eval()
  out <- torch::with_no_grad(net(torch::torch_tensor(X, dtype = torch::torch_float())))
  as.numeric(out$cpu())
}

.fg_nn_fit <- function(Xtr, ytr, family, p, seed, Xval = NULL, yval = NULL) {
  .require_torch()
  bin <- identical(family, "binomial")
  mu <- if (bin) 0 else mean(ytr)
  sdv <- if (bin) 1 else stats::sd(ytr)
  if (!is.finite(sdv) || sdv <= 0) sdv <- 1
  to_resp <- function(raw) if (bin) stats::plogis(raw) else raw * sdv + mu
  .fg_with_seed(seed, {
    torch::torch_manual_seed(as.integer(seed))
    net <- .fg_mlp()(ncol(Xtr), as.integer(p$n_layers), as.integer(p$width), p$dropout)
    opt <- torch::optim_adam(net$parameters, lr = p$lr, weight_decay = p$weight_decay)
    lossf <- if (bin) torch::nn_bce_with_logits_loss() else torch::nn_mse_loss()
    xt <- torch::torch_tensor(Xtr, dtype = torch::torch_float())
    yt <- torch::torch_tensor((ytr - mu) / sdv, dtype = torch::torch_float())
    n <- nrow(Xtr); bs <- min(as.integer(p$batch_size), n)
    epochs <- as.integer(p$epochs)
    patience <- p$patience %||% 20L
    best_loss <- Inf; best_epoch <- epochs; best_state <- NULL; since <- 0L; run <- 0L
    for (ep in seq_len(epochs)) {
      net$train()
      perm <- sample.int(n)
      starts <- seq.int(1L, n, by = bs)
      for (s in starts) {
        idx <- perm[s:min(n, s + bs - 1L)]
        if (length(idx) < 2L) next
        opt$zero_grad()
        loss <- lossf(net(xt[idx, ]), yt[idx])
        loss$backward()
        opt$step()
      }
      run <- ep
      if (!is.null(Xval)) {
        vl <- .fg_loss(yval, to_resp(.fg_nn_raw(net, Xval)), family)
        if (vl < best_loss - 1e-10) {
          best_loss <- vl; best_epoch <- ep; since <- 0L
          best_state <- lapply(net$state_dict(), function(t) t$clone())
        } else {
          since <- since + 1L
          if (since >= patience) break
        }
      }
    }
    if (!is.null(best_state)) net$load_state_dict(best_state)
    out <- list(model = net, best = if (is.null(Xval)) epochs else best_epoch, epochs_run = run,
                to_resp = to_resp)
    if (!is.null(Xval)) out$val_pred <- to_resp(.fg_nn_raw(net, Xval))
    out
  })
}

.fg_nn_predict <- function(fit, X) fit$to_resp(.fg_nn_raw(fit$model, X))

# ---- dispatch ---------------------------------------------------------

.fg_fit_one <- function(base_learner, Xtr, ytr, family, p, seed, Xval = NULL, yval = NULL, nthread = 1L) {
  if (identical(base_learner, "xgboost")) {
    .fg_xgb_fit(Xtr, ytr, family, p, seed, Xval, yval, nthread)
  } else {
    .fg_nn_fit(Xtr, ytr, family, p, seed, Xval, yval)
  }
}
.fg_predict_one <- function(base_learner, fit, X) {
  if (identical(base_learner, "xgboost")) .fg_xgb_predict(fit, X) else .fg_nn_predict(fit, X)
}

.fg_check_xy <- function(X, y, family) {
  X <- as.matrix(X)
  if (!is.numeric(X)) stop("`X` must be numeric (a numeric matrix or data frame).", call. = FALSE)
  storage.mode(X) <- "double"
  y <- as.numeric(y)
  if (length(y) != nrow(X)) stop("`y` must have one value per row of `X`.", call. = FALSE)
  if (identical(family, "binomial") && !all(y[!is.na(y)] %in% c(0, 1))) {
    stop("For family = 'binomial', `y` must be coded 0/1.", call. = FALSE)
  }
  list(X = X, y = y)
}

.fg_link <- function(b, family, ytr) {
  if (identical(family, "binomial")) stats::qlogis(pmin(pmax(b, 1e-6), 1 - 1e-6))
  else (b - mean(ytr)) / stats::sd(ytr)
}

#' Tune the Base Learner on an Inner Split
#'
#' Evaluates candidate hyperparameters of the XGBoost or neural-network base
#' learner on an inner training/validation split of the training rows, with
#' early stopping on the validation slice, and selects the configuration with
#' the smallest validation loss (log loss for a binary outcome, mean squared
#' error for a continuous one). The AUC or R-squared is reported beside it.
#'
#' @param X Numeric feature matrix or data frame. It must not contain the
#'   protected attribute.
#' @param y Outcome: 0/1 for `family = "binomial"`, numeric for
#'   `"gaussian"`.
#' @param family `"binomial"` or `"gaussian"`.
#' @param train Integer row indices to tune on (default all rows).
#' @param valid Optional integer row indices of the validation slice. When
#'   `NULL`, `train` is split at random into an inner training part and a
#'   validation share `inner_prop` (stratified on the outcome when binary).
#' @param base_learner `"xgboost"` or `"nn"`.
#' @param grid Named list of candidate values; defaults to
#'   [fairgate_base_grid()]. Elements not supplied take the defaults of the
#'   chosen learner.
#' @param n_search Number of configurations to evaluate by seeded random
#'   search; when the grid has no more points than this, every point is
#'   evaluated.
#' @param search `"random"` (default) or `"grid"` (every point of `grid`).
#' @param inner_prop Validation share when `valid` is `NULL`. Default 0.25.
#' @param seed Integer seed for the split, the random draws and the model
#'   fits. The caller's random number state is restored on exit.
#' @param standardise Logical; centre and scale features with the inner
#'   training rows' constants (median imputation of missing values first).
#' @param nthread Threads used by XGBoost. Default 1 (reproducible).
#' @param verbose Logical; print one line per configuration.
#' @return A list with `best` (the selected hyperparameters, with `nrounds` or
#'   `epochs` set to the early-stopped value, ready to pass as `params` to
#'   [fairgate_base_learner()]), `results` (a data frame with one row per
#'   configuration: its hyperparameters, `val_loss`, `val_perf` (AUC or
#'   R-squared), `best_iteration`, `secs` and `selected`), and `inner` (the
#'   training and validation row indices used).
#' @examples
#' if (requireNamespace("xgboost", quietly = TRUE) &&
#'     utils::packageVersion("xgboost") >= "3.1.0") {
#'   set.seed(1)
#'   X <- matrix(rnorm(600 * 5), 600, 5)
#'   y <- rbinom(600, 1, stats::plogis(X[, 1] - X[, 2]))
#'   tu <- fairgate_tune_base_learner(X, y, "binomial", base_learner = "xgboost",
#'                                    n_search = 3, seed = 1)
#'   tu$results[, c("max_depth", "eta", "val_loss", "val_perf", "best_iteration")]
#' }
#' @export
fairgate_tune_base_learner <- function(X, y, family = c("binomial", "gaussian"),
                                       train = NULL, valid = NULL,
                                       base_learner = c("xgboost", "nn"),
                                       grid = NULL, n_search = 24L,
                                       search = c("random", "grid"),
                                       inner_prop = 0.25, seed = 4242L,
                                       standardise = TRUE, nthread = 1L, verbose = FALSE) {
  family <- match.arg(family); base_learner <- match.arg(base_learner); search <- match.arg(search)
  if (identical(base_learner, "xgboost")) .fg_check_xgboost() else .require_torch()
  xy <- .fg_check_xy(X, y, family); X <- xy$X; y <- xy$y
  if (is.null(train)) train <- seq_len(nrow(X))
  train <- as.integer(train)
  if (is.null(valid)) {
    sp <- .fg_inner_split(train, y, family, inner_prop, seed)
  } else {
    sp <- list(train = setdiff(train, valid), valid = as.integer(valid))
  }
  defaults <- fairgate_base_grid(base_learner)
  if (!is.null(grid)) {
    bad <- setdiff(names(grid), names(defaults))
    if (length(bad)) stop(sprintf("Unknown hyperparameter(s) for '%s': %s", base_learner, paste(bad, collapse = ", ")),
                          call. = FALSE)
    defaults[names(grid)] <- grid
  }
  grid <- defaults
  cfgs <- .fg_draw_configs(grid, as.integer(n_search), search, seed)
  pr <- .fg_prep(X[sp$train, , drop = FALSE], X[sp$valid, , drop = FALSE], standardise)
  ytr <- y[sp$train]; yva <- y[sp$valid]
  rows <- vector("list", length(cfgs))
  for (j in seq_along(cfgs)) {
    p <- utils::modifyList(.fg_base_defaults(base_learner), cfgs[[j]])
    t0 <- proc.time()[["elapsed"]]
    fit <- .fg_fit_one(base_learner, pr$train, ytr, family, p, seed, pr$test, yva, nthread)
    el <- proc.time()[["elapsed"]] - t0
    rows[[j]] <- data.frame(config_id = j, as.data.frame(cfgs[[j]], stringsAsFactors = FALSE),
                            val_loss = .fg_loss(yva, fit$val_pred, family),
                            val_perf = .fg_perf(yva, fit$val_pred, family),
                            best_iteration = fit$best, secs = el, stringsAsFactors = FALSE)
    if (isTRUE(verbose)) {
      message(sprintf("[%s] config %d/%d: val loss %.5f, %s %.4f, best iteration %d (%.1f s)",
                      base_learner, j, length(cfgs), rows[[j]]$val_loss,
                      if (identical(family, "binomial")) "AUC" else "R2", rows[[j]]$val_perf, fit$best, el))
    }
    rm(fit)
  }
  res <- do.call(rbind, rows)
  ok <- which(is.finite(res$val_loss))
  if (!length(ok)) stop("No base-learner configuration produced a finite validation loss.", call. = FALSE)
  bi <- ok[which.min(res$val_loss[ok])]
  res$selected <- seq_len(nrow(res)) == bi
  best <- utils::modifyList(.fg_base_defaults(base_learner), cfgs[[bi]])
  if (identical(base_learner, "xgboost")) best$nrounds <- as.integer(res$best_iteration[bi])
  else best$epochs <- as.integer(res$best_iteration[bi])
  attr(res, "criterion") <- sprintf("min validation %s", if (identical(family, "binomial")) "log loss" else "mean squared error")
  list(best = best, results = res, inner = sp, base_learner = base_learner, family = family)
}

#' Cross-Fitted Base Learner (XGBoost or Neural Network)
#'
#' Fits the base learner whose scores fairGATE corrects. Training rows receive
#' out-of-fold scores from `n_folds`-fold cross-fitting, so that no training
#' row is scored by a model that saw its outcome; held-out rows receive the
#' prediction of a fit on all training rows. Features are median-imputed and
#' standardised with the training rows' constants. The protected attribute is
#' never an input: pass only the features in `X`.
#'
#' @inheritParams fairgate_tune_base_learner
#' @param train Integer row indices of the training rows (default all rows).
#' @param test Optional integer row indices of held-out rows to score.
#' @param params Named list of fixed hyperparameters. When `NULL` (default)
#'   they are tuned first by [fairgate_tune_base_learner()] on an inner split
#'   of `train`. Entries not supplied take the learner's defaults
#'   (XGBoost: `nrounds = 500`, `eta = 0.05`, `max_depth = 4`,
#'   `min_child_weight = 1`, `subsample = 0.8`, `colsample_bytree = 0.8`,
#'   `lambda = 1`, `alpha = 0`; neural network: `n_layers = 2`,
#'   `width = 64`, `dropout = 0.1`, `lr = 0.001`, `weight_decay = 1e-4`,
#'   `batch_size = 512`, `epochs = 100`). With fixed parameters the number of
#'   rounds or epochs is used as given (no early stopping).
#' @param n_folds Number of cross-fitting folds. Default 5.
#' @param folds Optional integer vector of fold labels `1..n_folds`, one per
#'   element of `train`, to reuse a fold assignment made elsewhere.
#' @param tune_args Optional list of further arguments to
#'   [fairgate_tune_base_learner()] (for example `grid`, `n_search`,
#'   `valid`).
#' @return An object of class `fairgate_base`: a list with `train` and `test`
#'   (scores on the response scale, probabilities for a binary outcome, in the
#'   order of `train` and `test`), `train_link` and `test_link` (the same on
#'   the link scale: logit, or standardised by the training outcome mean and
#'   standard deviation), `train_rows`, `test_rows`, `folds`, `params`,
#'   `tuning` (the tuning result or `NULL`), `perf` (out-of-fold and, when
#'   `test` is given, held-out AUC or R-squared), `family` and
#'   `base_learner`. Pass `list(list(train = b$train, test = b$test))` as
#'   `base_score` to [train_gnn()].
#' @examples
#' if (requireNamespace("xgboost", quietly = TRUE) &&
#'     utils::packageVersion("xgboost") >= "3.1.0") {
#'   set.seed(2)
#'   X <- matrix(rnorm(500 * 4), 500, 4)
#'   y <- X[, 1] + 0.5 * X[, 2] + rnorm(500)
#'   b <- fairgate_base_learner(X, y, "gaussian", train = 1:400, test = 401:500,
#'                              base_learner = "xgboost",
#'                              params = list(nrounds = 50, max_depth = 3))
#'   b$perf
#' }
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
#'   set.seed(3)
#'   X <- matrix(rnorm(400 * 4), 400, 4)
#'   y <- rbinom(400, 1, stats::plogis(X[, 1]))
#'   b <- fairgate_base_learner(X, y, "binomial", train = 1:300, test = 301:400,
#'                              base_learner = "nn",
#'                              params = list(epochs = 10, width = 16))
#'   b$perf
#' }
#' }
#' @export
fairgate_base_learner <- function(X, y, family = c("binomial", "gaussian"),
                                  train = NULL, test = NULL,
                                  base_learner = c("xgboost", "nn"),
                                  params = NULL, n_folds = 5L, folds = NULL,
                                  seed = 4242L, standardise = TRUE, nthread = 1L,
                                  tune_args = NULL, verbose = FALSE) {
  family <- match.arg(family); base_learner <- match.arg(base_learner)
  if (identical(base_learner, "xgboost")) .fg_check_xgboost() else .require_torch()
  xy <- .fg_check_xy(X, y, family); X <- xy$X; y <- xy$y
  if (is.null(train)) train <- seq_len(nrow(X))
  train <- as.integer(train); if (!is.null(test)) test <- as.integer(test)
  if (anyNA(y[train])) stop("`y` has missing values in the training rows.", call. = FALSE)
  n_folds <- as.integer(n_folds)
  if (n_folds < 2L) stop("`n_folds` must be at least 2.", call. = FALSE)
  tuning <- NULL
  if (is.null(params)) {
    ta <- c(list(X = X, y = y, family = family, train = train, base_learner = base_learner,
                 seed = seed, standardise = standardise, nthread = nthread, verbose = verbose),
            tune_args %||% list())
    tuning <- do.call(fairgate_tune_base_learner, ta)
    p <- tuning$best
  } else {
    allowed <- names(.fg_base_defaults(base_learner))
    if (identical(base_learner, "xgboost")) allowed <- c(allowed, "max_bin")
    bad <- setdiff(names(params), allowed)
    if (length(bad)) stop(sprintf("Unknown hyperparameter(s) for '%s': %s", base_learner, paste(bad, collapse = ", ")),
                          call. = FALSE)
    p <- utils::modifyList(.fg_base_defaults(base_learner), params)
  }
  pr <- .fg_prep(X[train, , drop = FALSE], if (is.null(test)) NULL else X[test, , drop = FALSE], standardise)
  ytr <- y[train]
  if (is.null(folds)) {
    folds <- .fg_folds(ytr, family, n_folds, seed)
  } else {
    folds <- as.integer(folds)
    if (length(folds) != length(train)) stop("`folds` must have one label per training row.", call. = FALSE)
    n_folds <- length(unique(folds))
  }
  oof <- rep(NA_real_, length(train))
  for (k in sort(unique(folds))) {
    a <- folds != k
    fit <- .fg_fit_one(base_learner, pr$train[a, , drop = FALSE], ytr[a], family, p, seed, nthread = nthread)
    oof[!a] <- .fg_predict_one(base_learner, fit, pr$train[!a, , drop = FALSE])
    rm(fit)
    if (isTRUE(verbose)) message(sprintf("[%s] fold %d of %d done", base_learner, k, n_folds))
  }
  te <- NULL
  if (!is.null(test)) {
    fit <- .fg_fit_one(base_learner, pr$train, ytr, family, p, seed, nthread = nthread)
    te <- .fg_predict_one(base_learner, fit, pr$test)
    rm(fit)
  }
  perf <- c(oof = .fg_perf(ytr, oof, family))
  if (!is.null(test) && !anyNA(y[test])) perf <- c(perf, test = .fg_perf(y[test], te, family))
  out <- list(train = oof, test = te,
              train_link = .fg_link(oof, family, ytr),
              test_link = if (is.null(te)) NULL else .fg_link(te, family, ytr),
              train_rows = train, test_rows = test, folds = folds, params = p, tuning = tuning,
              perf = perf, family = family, base_learner = base_learner, seed = seed)
  class(out) <- "fairgate_base"
  out
}

#' @export
print.fairgate_base <- function(x, ...) {
  pn <- if (identical(x$family, "binomial")) "AUC" else "R-squared"
  cat(sprintf("Cross-fitted %s base learner (%s outcome), %d folds\n",
              if (identical(x$base_learner, "nn")) "neural-network" else "XGBoost",
              if (identical(x$family, "binomial")) "binary" else "continuous",
              length(unique(x$folds))))
  cat(sprintf("  training rows: %d (out-of-fold %s %.4f)\n", length(x$train_rows), pn, x$perf[["oof"]]))
  if (!is.null(x$test_rows)) {
    cat(sprintf("  held-out rows: %d%s\n", length(x$test_rows),
                if ("test" %in% names(x$perf)) sprintf(" (%s %.4f)", pn, x$perf[["test"]]) else ""))
  }
  cat(sprintf("  hyperparameters: %s\n", paste(names(x$params), unlist(x$params), sep = " = ", collapse = ", ")))
  invisible(x)
}
