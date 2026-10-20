test_that("fairgate() runs end to end and applies the selection rule", {
  skip_on_cran()
  skip_if_no_torch()
  skip_if_no_xgboost()
  td <- toy_data(n = 500L)
  fg <- fairgate(td$pd$X, td$pd$y, td$pd$group, "binomial", reference_group = 0,
                 base_params = list(nrounds = 30, max_depth = 2), n_draws = 3, epochs = 15,
                 interval_reps = 2, interval_epochs = 2, n_partitions = 2)
  expect_s3_class(fg, "fairgate_fit")
  expect_equal(nrow(fg$search), 3L)
  s <- fg$search; ok <- is.finite(s$perf) & is.finite(s$worst)
  band <- which(ok & s$perf >= max(s$perf[ok]) - 0.01)
  expect_equal(which(s$selected), band[which.min(s$worst[band])])
  ex <- fairgate_export(fg)
  expect_true(all(c("partition", "row", "group", "base", "pred", "adjustment", "adj_lo", "adj_hi",
                    "interval_excludes_zero") %in% names(ex)))
  expect_equal(nrow(ex), sum(vapply(fg$settings$splits, function(s) length(s$test), 1L)))
  expect_equal(ex$adjustment, ex$pred - ex$base, tolerance = 1e-12)
  f <- tempfile(fileext = ".csv")
  fairgate_export(fg, f)
  expect_true(file.exists(f)); unlink(f)
  expect_output(print(fg), "fairGATE fit")
})

test_that("fairgate() accepts precomputed base scores and a neural-network base learner", {
  skip_on_cran()
  skip_if_no_torch()
  td <- toy_data(n = 400L, family = "gaussian")
  fg <- fairgate(td$pd$X, td$pd$y, td$pd$group, "gaussian", reference_group = 0,
                 base_learner = "nn", base_params = list(epochs = 5L, width = 8L),
                 signal_learner = "nn", signal_params = list(epochs = 3L, width = 8L),
                 params = list(lr = 0.002, hidden_dim = 16L, dropout_rate = 0.1, temperature = 1,
                               lambda = 4L, penalty = "random_effects", lambda_thr = 0L,
                               intervention_cost = 0, coupling = "residual"),
                 epochs = 10, interval_reps = 0)
  expect_null(fg$search)
  expect_equal(fg$selected$coupling, "residual")
  expect_true(all(is.finite(fg$predictions$pred)))
})

test_that("fairgate() validates its inputs", {
  skip_if_no_torch()
  X <- matrix(rnorm(40), 20)
  expect_error(fairgate(X, rbinom(20, 1, 0.5)), "Supply `y` and `group`")
  expect_error(fairgate(X, rbinom(20, 1, 0.5), rep(0:1, 10), reference_group = 5), "reference_group")
})
