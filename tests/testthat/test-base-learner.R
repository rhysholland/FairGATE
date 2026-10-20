sim <- function(n = 400, family = "binomial", seed = 1) {
  set.seed(seed)
  X <- matrix(rnorm(n * 4), n, 4)
  y <- if (family == "binomial") rbinom(n, 1, stats::plogis(X[, 1] - X[, 2])) else X[, 1] + rnorm(n)
  list(X = X, y = y)
}

test_that("the base learner never takes the protected attribute", {
  expect_false(any(c("group", "g", "protected") %in% names(formals(fairgate_base_learner))))
  expect_false(any(c("group", "g", "protected") %in% names(formals(fairgate_tune_base_learner))))
})

test_that("fold labels are stratified and seeded", {
  y <- rep(0:1, c(70, 30))
  f <- fairGATE:::.fg_folds(y, "binomial", 5L, 1L)
  expect_equal(as.vector(table(f, y)[, "1"]), rep(6L, 5))
  expect_identical(f, fairGATE:::.fg_folds(y, "binomial", 5L, 1L))
})

test_that("XGBoost cross-fitting is reproducible and keeps the RNG state", {
  skip_if_no_xgboost()
  d <- sim()
  set.seed(7); before <- .Random.seed
  b1 <- fairgate_base_learner(d$X, d$y, "binomial", train = 1:300, test = 301:400,
                              base_learner = "xgboost", params = list(nrounds = 30, max_depth = 2))
  expect_identical(before, .Random.seed)
  b2 <- fairgate_base_learner(d$X, d$y, "binomial", train = 1:300, test = 301:400,
                              base_learner = "xgboost", params = list(nrounds = 30, max_depth = 2))
  expect_s3_class(b1, "fairgate_base")
  expect_length(b1$train, 300); expect_length(b1$test, 100)
  expect_true(all(b1$train > 0 & b1$train < 1))
  expect_identical(b1$train, b2$train); expect_identical(b1$test, b2$test)
  expect_gt(b1$perf[["test"]], 0.6)
  # supplied folds are used as given
  b3 <- fairgate_base_learner(d$X, d$y, "binomial", train = 1:300, base_learner = "xgboost",
                              params = list(nrounds = 30), folds = rep_len(1:3, 300))
  expect_equal(length(unique(b3$folds)), 3L)
  expect_error(fairgate_base_learner(d$X, d$y, "binomial", base_learner = "xgboost",
                                     params = list(depth = 3)), "Unknown hyperparameter")
})

test_that("XGBoost tuning selects the smallest validation loss", {
  skip_if_no_xgboost()
  d <- sim(family = "gaussian")
  tu <- fairgate_tune_base_learner(d$X, d$y, "gaussian", base_learner = "xgboost", n_search = 3,
                                   grid = list(nrounds = 200L, early_stopping_rounds = 10L))
  expect_equal(nrow(tu$results), 3L)
  expect_equal(which(tu$results$selected), which.min(tu$results$val_loss))
  expect_equal(tu$best$nrounds, tu$results$best_iteration[tu$results$selected])
  expect_lte(tu$best$nrounds, 200L)
})

test_that("the neural-network base learner fits, tunes and is reproducible", {
  skip_if_no_torch()
  d <- sim(family = "gaussian")
  p <- list(epochs = 20L, width = 16L, n_layers = 1L, batch_size = 64L, lr = 0.01)
  b1 <- fairgate_base_learner(d$X, d$y, "gaussian", train = 1:300, test = 301:400,
                              base_learner = "nn", params = p)
  b2 <- fairgate_base_learner(d$X, d$y, "gaussian", train = 1:300, test = 301:400,
                              base_learner = "nn", params = p)
  expect_identical(b1$test, b2$test)
  expect_gt(b1$perf[["test"]], 0.3)
  skip_on_cran()
  db <- sim()
  tu <- fairgate_tune_base_learner(db$X, db$y, "binomial", base_learner = "nn", n_search = 2,
                                   grid = list(epochs = 15L, patience = 3L, width = 16L))
  expect_true(tu$best$epochs <= 15L)
  expect_equal(sum(tu$results$selected), 1L)
})
