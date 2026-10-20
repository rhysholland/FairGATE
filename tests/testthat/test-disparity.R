test_that("fairgate_group_disparity matches hand-computed average odds differences", {
  y    <- c(1, 1, 0, 0, 1, 1, 0, 0)
  g    <- c(0, 0, 0, 0, 1, 1, 1, 1)
  pred <- c(0.9, 0.8, 0.7, 0.1, 0.9, 0.2, 0.1, 0.1)
  d <- fairgate_group_disparity(y, pred, g, reference = 0, family = "binomial", threshold = 0.5)
  # reference: TPR 1, FPR 0.5; group 1: TPR 0.5, FPR 0
  expect_equal(d$per_group$gap, 0.5 * ((0 - 0.5) + (0.5 - 1)))
  expect_equal(d$worst, 0.5)
  expect_equal(d$worst_group, 1L)
})

test_that("fairgate_group_disparity respects the eligibility rule", {
  set.seed(3)
  g <- rep(0:2, c(100, 100, 20)); y <- rbinom(220, 1, 0.5)
  pred <- stats::plogis(rnorm(220) + ifelse(g == 2, 2, 0))
  d <- fairgate_group_disparity(y, pred, g, reference = 0, family = "binomial",
                                threshold = 0.5, eligible = 1L)
  expect_false(d$per_group$eligible[d$per_group$group == 2])
  expect_equal(d$worst_group, 1L)
})

test_that("continuous disparity is a stratified mean gap in outcome SDs", {
  set.seed(4)
  y <- rnorm(500); g <- rep(0:1, 250)
  d <- fairgate_group_disparity(y, y + 0.5 * g, g, reference = 0, family = "gaussian", scale = 1)
  expect_equal(d$worst, 0.5, tolerance = 0.1)
  d0 <- fairgate_group_disparity(y, y, g, reference = 0, family = "gaussian", scale = 1)
  expect_lt(abs(d0$worst - d$worst + 0.5), 0.2)
})

test_that("fairgate_disparity_target points towards closing the gap", {
  set.seed(5)
  n <- 2000; g <- rep(0:1, each = n / 2); y <- rbinom(n, 1, 0.5)
  base <- stats::plogis(rnorm(n) + 2 * y - 1 - 0.8 * g)   # group 1 under-scored
  t <- fairgate_disparity_target(y, g, base, reference = 0, family = "binomial")
  expect_length(t, n)
  expect_gt(mean(t[g == 1]), 0)   # raise group 1
  expect_lt(mean(t[g == 0]), 0)   # lower the reference group
  tc <- fairgate_disparity_target(y + rnorm(n), g, base, reference = 0, family = "gaussian")
  expect_true(all(is.finite(tc)))
})

test_that("penalty rungs nest as documented", {
  m <- c(0.2, 0.3, 0.5)
  v_eq <- rep(0.01, 3)
  expect_equal(fairgate_penalty_from_cells(m, penalty = "variance"), stats::var(m))
  expect_equal(fairgate_penalty_from_cells(m, rep(0, 3), penalty = "bias_corrected"),
               fairgate_penalty_from_cells(m, penalty = "variance"))
  expect_equal(fairgate_penalty_from_cells(m, v_eq, penalty = "random_effects"),
               fairgate_penalty_from_cells(m, v_eq, penalty = "bias_corrected"), tolerance = 1e-8)
  expect_equal(fairgate_penalty_from_cells(c(0.3, 0.3, 0.3), v_eq, penalty = "random_effects"), 0)
})

test_that("numeric and tensor penalty paths agree", {
  skip_if_no_torch()
  m <- c(0.2, 0.3, 0.5, 0.45)
  v <- c(0.010, 0.004, 0.020, 0.008)
  for (p in c("variance", "bias_corrected", "random_effects")) {
    num <- fairgate_penalty_from_cells(m, v, penalty = p)
    ten <- fairgate_penalty_from_cells(
      torch::torch_tensor(m, dtype = torch::torch_float64()), v, penalty = p)
    expect_equal(num, as.numeric(ten), tolerance = 1e-10)
  }
})

test_that("lambda ladder starts at zero and ends at lambda_max", {
  l <- fairgate_lambda_ladder(4, 16)
  expect_equal(l[1], 0)
  expect_equal(l[4], 16)
  expect_true(all(diff(l) > 0))
})
