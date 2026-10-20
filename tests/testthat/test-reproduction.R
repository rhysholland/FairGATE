test_that("fits with the fairGATE components on reproduce the 0.5.0 reference", {
  skip_on_cran()
  skip_if_no_torch()
  ref <- readRDS(test_path("fixtures", "toy_reference.rds"))
  skip_if_not(identical(as.character(utils::packageVersion("torch")), ref$torch),
              "reference made with a different torch version")
  for (nm in names(toy_cases())) {
    f <- fit_toy_case(toy_cases()[[nm]])
    expect_equal(f$final_results$prob, ref[[nm]]$prob, tolerance = 1e-12, info = nm)
    cols <- names(ref[[nm]]$attribution)
    expect_equal(as.data.frame(f$attribution)[, cols], ref[[nm]]$attribution, tolerance = 1e-12, info = nm)
  }
})

test_that("an untrained residual model returns the base learner exactly", {
  skip_if_no_torch()
  td <- toy_data()
  f <- train_gnn(td$pd, run_tuning = FALSE,
                 best_params = list(lr = 0, hidden_dim = 8L, dropout_rate = 0, lambda = 0, temperature = 1),
                 epochs = 1L, seed = 1L, splits = td$splits, base_score = td$base, coupling = "residual")
  expect_equal(f$attribution$pred, f$attribution$base, tolerance = 0)
  expect_true(all(f$attribution$adjustment == 0))
})
