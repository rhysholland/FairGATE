test_that("the search space and draws are those of the accompanying study", {
  sp <- fairgate_space()
  expect_equal(sp$n_draws, 48L)
  expect_equal(sp$seed, 4242L)
  expect_setequal(names(sp$space), c("lr", "hidden_dim", "dropout_rate", "temperature", "lambda",
                                     "penalty", "lambda_thr", "intervention_cost", "coupling"))
  g <- fairgate_draw_configs(sp$space, sp$n_draws, sp$seed)
  expect_length(g, 48L)
  sig <- vapply(g, function(x) paste(unlist(x), collapse = "|"), "")
  expect_false(anyDuplicated(sig) > 0)
  thr <- vapply(g, function(x) identical(x$penalty, "thresholded"), TRUE)
  expect_true(all(vapply(g[thr], function(x) x$lambda_thr == 0, TRUE)))
  # first three configurations of the study's grid
  expect_equal(unlist(g[[1]][c("lr", "hidden_dim", "dropout_rate", "temperature", "lambda")]),
               c(lr = 0.002, hidden_dim = 128, dropout_rate = 0.1, temperature = 1, lambda = 1))
  expect_equal(g[[1]]$penalty, "thresholded"); expect_equal(g[[1]]$coupling, "expert")
  expect_equal(g[[2]]$penalty, "random_effects"); expect_equal(g[[2]]$coupling, "residual")
  expect_equal(g[[2]]$intervention_cost, 0.02)
  expect_equal(g[[3]]$lambda, 64L); expect_equal(g[[3]]$penalty, "variance")
})

test_that("drawing configurations leaves the caller's random stream alone", {
  set.seed(99); a <- stats::runif(1)
  set.seed(99); invisible(fairgate_draw_configs(fairgate_space()$space, 5)); b <- stats::runif(1)
  expect_identical(a, b)
})

test_that("default base-learner grids are complete", {
  gx <- fairgate_base_grid("xgboost")
  expect_true(all(c("max_depth", "eta", "nrounds", "early_stopping_rounds", "subsample",
                    "colsample_bytree", "min_child_weight") %in% names(gx)))
  gn <- fairgate_base_grid("nn")
  expect_true(all(c("n_layers", "width", "dropout", "lr", "weight_decay", "batch_size",
                    "epochs", "patience") %in% names(gn)))
})
