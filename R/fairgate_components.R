# ------------------------------------------------------------------
# fairGATE components: sparse gates, the thresholded worst-group penalty,
# the crossed (intersectional) penalty, the routing-balance regulariser, the
# disparity-contribution target and the multi-group disparity summary.
#
# Every function here is reached from train_gnn() only when the corresponding
# argument is switched on; with every such argument at its default the earlier
# code path runs unchanged. The helpers are exported so that each piece can be
# evaluated and differentiated on constructed inputs, independently of a fit.
# ------------------------------------------------------------------

# ---- sparse gates --------------------------------------------------
# Forward maps on the rows of a matrix of gate logits. Both have closed forms
# after sorting (Martins and Astudillo 2016; Peters, Niculae and Martins 2019).
.fg_sparsemax_fwd <- function(z) {
  K <- as.integer(z$size(2))
  zs <- torch::torch_sort(z, dim = 2L, descending = TRUE)[[1]]
  rho <- torch::torch_arange(1, K, dtype = z$dtype)$view(c(1L, -1L))
  cs <- zs$cumsum(dim = 2L)
  supp <- ((1 + rho * zs) > cs)$sum(dim = 2L, keepdim = TRUE)
  tau <- (cs$gather(2L, supp) - 1) / supp$to(dtype = z$dtype)
  torch::torch_clamp(z - tau, min = 0)
}

.fg_entmax15_fwd <- function(z) {
  K <- as.integer(z$size(2))
  X <- z / 2
  X <- X - X$max(dim = 2L, keepdim = TRUE)[[1]]
  Xs <- torch::torch_sort(X, dim = 2L, descending = TRUE)[[1]]
  rho <- torch::torch_arange(1, K, dtype = z$dtype)$view(c(1L, -1L))
  mu <- Xs$cumsum(dim = 2L) / rho
  mu2 <- (Xs * Xs)$cumsum(dim = 2L) / rho
  ss <- rho * (mu2 - mu * mu)
  delta <- torch::torch_clamp((1 - ss) / rho, min = 0)
  tau <- mu - torch::torch_sqrt(delta)
  supp <- (tau <= Xs)$sum(dim = 2L, keepdim = TRUE)
  tau_star <- tau$gather(2L, supp)
  torch::torch_clamp(X - tau_star, min = 0)^2
}

# Custom backward passes: the Jacobians are known in closed form, and
# differentiating through the sort-based forward would pass through a square
# root at zero (entmax) and produce NaN.
.fg_sparsemax_fn <- NULL
.fg_entmax15_fn <- NULL
.fg_sparse_fns <- function() {
  if (is.null(.fg_sparsemax_fn)) {
    sm <- torch::autograd_function(
      forward = function(ctx, z) {
        y <- .fg_sparsemax_fwd(z)
        ctx$save_for_backward(y = y)
        y
      },
      backward = function(ctx, grad_output) {
        y <- ctx$saved_variables$y
        s <- (y > 0)$to(dtype = grad_output$dtype)
        g <- grad_output * s
        m <- g$sum(dim = 2L, keepdim = TRUE) / s$sum(dim = 2L, keepdim = TRUE)
        list(z = (grad_output - m) * s)
      })
    em <- torch::autograd_function(
      forward = function(ctx, z) {
        y <- .fg_entmax15_fwd(z)
        ctx$save_for_backward(y = y)
        y
      },
      backward = function(ctx, grad_output) {
        y <- ctx$saved_variables$y
        u <- torch::torch_sqrt(y)
        dx <- grad_output * u
        q <- dx$sum(dim = 2L, keepdim = TRUE) / u$sum(dim = 2L, keepdim = TRUE)
        list(z = dx - q * u)
      })
    utils::assignInMyNamespace(".fg_sparsemax_fn", sm)
    utils::assignInMyNamespace(".fg_entmax15_fn", em)
  }
  list(sparsemax = .fg_sparsemax_fn, entmax15 = .fg_entmax15_fn)
}

#' Gate Maps: softmax, sparsemax and entmax-1.5
#'
#' Maps each row of a matrix of gate logits to a probability vector. The
#' sparse maps return exact zeros, so a gate built on them can place all of an
#' individual's weight on a single component, in particular on the base
#' learner, in which case the fairGATE adjustment for that individual is
#' exactly zero.
#'
#' @param z A two-dimensional `torch_tensor` of logits (rows by components),
#'   already divided by the gate temperature.
#' @param type One of `"softmax"`, `"sparsemax"` or `"entmax15"`.
#' @return A tensor of the same shape whose rows are non-negative and sum to
#'   one. Differentiable; the sparse maps use their closed-form Jacobians.
#' @examples
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
#'   z <- torch::torch_tensor(matrix(c(2, 1, -1, 0.5, 0.4, 0.3), 2, byrow = TRUE))
#'   fairgate_gate_map(z, "softmax")
#'   fairgate_gate_map(z, "sparsemax")
#'   fairgate_gate_map(z, "entmax15")
#' }
#' }
#' @export
fairgate_gate_map <- function(z, type = c("softmax", "sparsemax", "entmax15")) {
  type <- match.arg(type)
  if (identical(type, "softmax")) return(torch::nnf_softmax(z, dim = 2))
  fns <- .fg_sparse_fns()
  if (identical(type, "sparsemax")) fns$sparsemax(z) else fns$entmax15(z)
}

# Gradient reversal: identity in the forward pass, gradient multiplied by
# -gamma in the backward pass. Written without a custom autograd function:
# x_d - gamma (x - x_d), with x_d = x detached, has value exactly x (the
# bracket is exactly zero) and derivative -gamma.
.fg_grl <- function(x, gamma) { xd <- x$detach(); xd - gamma * (x - xd) }

# ---- thresholded group error rates ---------------------------------

#' Cell Structure for the Thresholded Error-Rate Penalty
#'
#' @param group Protected group codes, one per row.
#' @param y Binary outcome (0/1) for `family = "binomial"`; ignored otherwise.
#' @param reference Code of the reference group. `NULL` compares every group
#'   with the pooled rate over all rows.
#' @param min_cell Smallest (group, outcome class) cell that may contribute.
#' @param dtype torch dtype of the indicator matrices.
#' @return A list with indicator matrices `M1`, `M0` (rows by groups) for the
#'   positive and negative classes, their sizes, the group codes and the column
#'   of the reference group (`0` for the pooled comparison).
#' @examples
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
#'   set.seed(1)
#'   group <- sample(0:2, 100, replace = TRUE)
#'   y <- rbinom(100, 1, 0.4)
#'   plan <- fairgate_threshold_plan(group, y, reference = 0)
#'   names(plan)
#' }
#' }
#' @export
fairgate_threshold_plan <- function(group, y, reference = NULL, min_cell = 2L,
                                    dtype = torch::torch_float()) {
  g <- as.integer(group); y <- as.numeric(y)
  ug <- sort(unique(g))
  ok <- ug[vapply(ug, function(k) sum(g == k & y == 1) >= min_cell &&
                                    sum(g == k & y == 0) >= min_cell, logical(1))]
  if (length(ok) < 1L) return(NULL)
  M1 <- sapply(ok, function(k) as.numeric(g == k & y == 1))
  M0 <- sapply(ok, function(k) as.numeric(g == k & y == 0))
  if (!is.matrix(M1)) { M1 <- matrix(M1, ncol = length(ok)); M0 <- matrix(M0, ncol = length(ok)) }
  ref_col <- if (!is.null(reference) && as.integer(reference) %in% ok) match(as.integer(reference), ok) else 0L
  if (ref_col == 0L && length(ok) < 1L) return(NULL)
  if (ref_col > 0L && length(ok) < 2L) return(NULL)
  list(groups = ok, ref_col = ref_col,
       M1 = torch::torch_tensor(M1, dtype = dtype), M0 = torch::torch_tensor(M0, dtype = dtype),
       n1 = torch::torch_tensor(colSums(M1), dtype = dtype),
       n0 = torch::torch_tensor(colSums(M0), dtype = dtype),
       pos = torch::torch_tensor(as.numeric(y == 1), dtype = dtype),
       n_pos = sum(y == 1), n_neg = sum(y == 0))
}

#' Differentiable Penalty on Thresholded Group Error Rates
#'
#' Replaces the hard decision `1[p_i >= t]` by `sigmoid((p_i - t) / s)`, with
#' `t` the prevalence-matched threshold of the current batch (the
#' `1 - prevalence` quantile of the predictions, treated as a constant) and `s`
#' a temperature, and penalises the worst group's gap in the resulting soft
#' true- and false-positive rates relative to the reference group.
#'
#' @param pred Predicted probabilities, a `torch_tensor` of length `n`.
#' @param plan From [fairgate_threshold_plan()].
#' @param prevalence Share of rows to flag positive, in `(0, 1)`: the training
#'   prevalence.
#' @param temp Temperature `s` of the smooth indicator, on the probability
#'   scale.
#' @param target `"eo"` penalises `0.5 (dTPR^2 + dFPR^2)` per group, an upper
#'   bound on the squared average odds difference; `"aod"` penalises
#'   `AOD^2 = (0.5 (dTPR + dFPR))^2`.
#' @param agg `"max"` (worst group) or `"mean"` over non-reference groups.
#' @return A scalar tensor. As `temp` goes to zero the soft rates converge to
#'   the hard rates at threshold `t`.
#' @examples
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
#'   set.seed(1)
#'   group <- sample(0:2, 300, replace = TRUE)
#'   y <- rbinom(300, 1, 0.4)
#'   p <- torch::torch_tensor(stats::plogis(rnorm(300) + y + 0.4 * group))
#'   plan <- fairgate_threshold_plan(group, y, reference = 0)
#'   fairgate_threshold_penalty(p, plan, prevalence = mean(y))
#' }
#' }
#' @export
fairgate_threshold_penalty <- function(pred, plan, prevalence, temp = 0.05,
                                       target = c("eo", "aod"), agg = c("max", "mean")) {
  target <- match.arg(target); agg <- match.arg(agg)
  if (is.null(plan)) return(torch::torch_zeros(1, dtype = pred$dtype)$squeeze())
  n <- as.integer(pred$size(1))
  pd <- pred$detach()
  t <- torch::torch_quantile(pd, 1 - prevalence)$detach()
  d <- torch::torch_sigmoid((pred - t) / temp)
  tpr <- torch::torch_matmul(plan$M1$t(), d) / plan$n1
  fpr <- torch::torch_matmul(plan$M0$t(), d) / plan$n0
  if (plan$ref_col > 0L) {
    tr <- tpr[plan$ref_col]; fr <- fpr[plan$ref_col]
    keep <- setdiff(seq_along(plan$groups), plan$ref_col)
    dT <- tpr[keep] - tr; dF <- fpr[keep] - fr
  } else {
    tr <- torch::torch_sum(d * plan$pos) / plan$n_pos
    fr <- torch::torch_sum(d * (1 - plan$pos)) / plan$n_neg
    dT <- tpr - tr; dF <- fpr - fr
  }
  e <- if (identical(target, "eo")) 0.5 * (dT^2 + dF^2) else (0.5 * (dT + dF))^2
  if (identical(agg, "max")) torch::torch_max(e) else torch::torch_mean(e)
}

#' Worst-Group Conditional Mean Gap Penalty (continuous outcomes)
#'
#' The continuous counterpart of [fairgate_threshold_penalty()]: for each
#' non-reference group the squared mean over outcome strata of the difference
#' between that group's and the reference group's mean prediction (the squared
#' CMPD of that group), aggregated by the maximum or the mean.
#'
#' @param pred Predictions on the scale the penalty is defined on.
#' @param plan From [fairgate_penalty_plan()] with `min_cell = 1`.
#' @param reference Reference group code, or `NULL` for the pooled stratum mean.
#' @param agg `"max"` or `"mean"`.
#' @return A scalar tensor.
#' @examples
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
#'   set.seed(1)
#'   group <- sample(0:2, 300, replace = TRUE)
#'   y <- rnorm(300)
#'   strata <- as.integer(cut(y, stats::quantile(y, 0:5 / 5), include.lowest = TRUE))
#'   pred <- torch::torch_tensor(y + 0.2 * group)
#'   plan <- fairgate_penalty_plan(group, strata, min_cell = 1L)
#'   fairgate_group_gap_penalty(pred, plan, reference = 0)
#' }
#' }
#' @export
fairgate_group_gap_penalty <- function(pred, plan, reference = NULL, agg = c("max", "mean")) {
  agg <- match.arg(agg)
  gaps <- list()
  for (cell in plan) {
    mhat <- torch::torch_matmul(cell$M$t(), pred) / cell$n
    if (!is.null(reference) && as.integer(reference) %in% cell$groups) {
      rj <- match(as.integer(reference), cell$groups)
      for (j in setdiff(seq_along(cell$groups), rj)) {
        key <- as.character(cell$groups[j])
        gaps[[key]] <- c(gaps[[key]], list(mhat[j] - mhat[rj]))
      }
    } else {
      pooled <- torch::torch_sum(mhat * cell$n) / torch::torch_sum(cell$n)
      for (j in seq_along(cell$groups)) {
        key <- as.character(cell$groups[j])
        gaps[[key]] <- c(gaps[[key]], list(mhat[j] - pooled))
      }
    }
  }
  if (!length(gaps)) return(torch::torch_zeros(1, dtype = pred$dtype)$squeeze())
  e <- torch::torch_stack(lapply(gaps, function(l) torch::torch_mean(torch::torch_stack(l))^2))
  if (identical(agg, "max")) torch::torch_max(e) else torch::torch_mean(e)
}

# ---- crossed variance components -----------------------------------

#' Cell Structure for Crossed (Intersectional) Variance Components
#'
#' @param A,B Codes of the two protected attributes, one per row.
#' @param strata Outcome stratum identifiers in `1:Q`.
#' @param Q Number of strata.
#' @param min_cell Smallest (a, b, stratum) cell that may contribute.
#' @param dtype torch dtype.
#' @return A list with one element per stratum holding the indicator matrix of
#'   the contributing (a, b) cells, their sizes and codes.
#' @examples
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
#'   set.seed(1)
#'   A <- sample(0:1, 400, replace = TRUE); B <- sample(0:2, 400, replace = TRUE)
#'   strata <- sample(1:2, 400, replace = TRUE)
#'   plan <- fairgate_crossed_plan(A, B, strata)
#'   length(plan)
#' }
#' }
#' @export
fairgate_crossed_plan <- function(A, B, strata, Q = max(as.integer(strata)), min_cell = 2L,
                                  dtype = torch::torch_float()) {
  A <- as.integer(A); B <- as.integer(B); s <- as.integer(strata)
  out <- list()
  for (q in seq_len(as.integer(Q))) {
    iq <- s == q
    cells <- unique(data.frame(a = A[iq], b = B[iq]))
    if (!nrow(cells)) next
    cells <- cells[order(cells$a, cells$b), , drop = FALSE]
    cnt <- mapply(function(a, b) sum(iq & A == a & B == b), cells$a, cells$b)
    cells <- cells[cnt >= min_cell, , drop = FALSE]
    if (nrow(cells) < 2L) next
    M <- sapply(seq_len(nrow(cells)), function(j) as.numeric(iq & A == cells$a[j] & B == cells$b[j]))
    if (!is.matrix(M)) M <- matrix(M, ncol = nrow(cells))
    out[[length(out) + 1L]] <- list(q = q, a = cells$a, b = cells$b,
                                    M = torch::torch_tensor(M, dtype = dtype),
                                    n = torch::torch_tensor(colSums(M), dtype = dtype))
  }
  out
}

#' Crossed Variance-Component Penalty for Intersectional Groups
#'
#' Decomposes, within each outcome stratum, the between-cell dispersion of the
#' mean predictions of the (a, b) intersectional cells into two main effects and
#' an interaction, each estimated with a correction for the sampling noise in
#' the cell means, and returns their sum over strata.
#'
#' @param pred Predictions (tensor), on the penalty scale.
#' @param plan From [fairgate_crossed_plan()].
#' @param penalty `"random_effects"`, `"bias_corrected"` or `"variance"`, applied
#'   to the main effects; the interaction is bias corrected unless
#'   `"variance"`.
#' @param eps Floor on sampling variances.
#' @details For stratum q write m_ab for the cell means and v_ab for their
#'   detached sampling variances. The main effect of A is the between-level
#'   variance component (as in [fairgate_penalty_from_cells()]) of the
#'   unweighted marginal means m_a. = mean_b m_ab, with sampling variances
#'   sum_b v_ab / B_a^2; likewise for B. The interaction is computed on the
#'   largest complete sub-grid A' x B': with residuals
#'   r_ab = m_ab - m_a. - m_.b + m_.., E(sum r^2) = (A'-1)(B'-1) sigma^2_int +
#'   (1 - 1/A')(1 - 1/B') sum v_ab, because the residual projection has constant
#'   diagonal, so sigma^2_int is estimated by
#'   max(0, (sum r^2 - (1 - 1/A')(1 - 1/B') sum v) / ((A'-1)(B'-1))). With a
#'   single level of B the penalty reduces to the one-way component of A.
#' @return A scalar tensor.
#' @examples
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
#'   set.seed(1)
#'   A <- sample(0:1, 400, replace = TRUE); B <- sample(0:2, 400, replace = TRUE)
#'   strata <- sample(1:2, 400, replace = TRUE)
#'   pred <- torch::torch_tensor(stats::plogis(rnorm(400) + 0.3 * A * B))
#'   plan <- fairgate_crossed_plan(A, B, strata)
#'   fairgate_crossed_penalty(pred, plan, penalty = "random_effects")
#' }
#' }
#' @export
fairgate_crossed_penalty <- function(pred, plan, penalty = c("random_effects", "bias_corrected", "variance"),
                                     eps = 1e-12) {
  penalty <- match.arg(penalty)
  total <- NULL
  add <- function(x) total <<- if (is.null(total)) x else total + x
  for (cell in plan) {
    M <- cell$M$to(dtype = pred$dtype); nn <- cell$n$to(dtype = pred$dtype)
    m <- torch::torch_matmul(M$t(), pred) / nn
    pd <- pred$detach(); md <- m$detach()
    ss <- torch::torch_sum(M * (pd$unsqueeze(2) - md$unsqueeze(1))^2, dim = 1)
    v <- torch::torch_clamp(ss / (nn - 1) / nn, min = eps)$detach()
    marg <- function(codes) {
      lv <- sort(unique(codes))
      if (length(lv) < 2L) return(NULL)
      mm <- list(); vv <- list()
      for (l in lv) {
        j <- which(codes == l)
        mm[[length(mm) + 1L]] <- torch::torch_mean(m[j])
        vv[[length(vv) + 1L]] <- torch::torch_sum(v[j]) / length(j)^2
      }
      fairgate_penalty_from_cells(torch::torch_stack(mm), torch::torch_stack(vv), penalty = penalty, eps = eps)
    }
    la <- marg(cell$a); if (!is.null(la)) add(la)
    lb <- marg(cell$b); if (!is.null(lb)) add(lb)
    # Largest complete sub-grid: keep a-levels observed with every b-level
    # that is itself observed with every kept a-level (greedy, by coverage).
    ua <- sort(unique(cell$a)); ub <- sort(unique(cell$b))
    tab <- table(factor(cell$a, ua), factor(cell$b, ub)) > 0
    keep_b <- ub[colSums(tab) == nrow(tab)]
    keep_a <- ua[rowSums(tab[, as.character(keep_b), drop = FALSE]) == length(keep_b)]
    if (length(keep_a) >= 2L && length(keep_b) >= 2L) {
      Aa <- length(keep_a); Bb <- length(keep_b)
      idx <- matrix(0L, Aa, Bb)
      for (i in seq_len(Aa)) for (j in seq_len(Bb)) idx[i, j] <- which(cell$a == keep_a[i] & cell$b == keep_b[j])
      mg <- m[as.vector(idx)]$view(c(Bb, Aa))$t()   # Aa x Bb, column-major fill
      vg <- v[as.vector(idx)]
      r <- mg - mg$mean(dim = 2L, keepdim = TRUE) - mg$mean(dim = 1L, keepdim = TRUE) + mg$mean()
      sr2 <- torch::torch_sum(r^2)
      df <- (Aa - 1) * (Bb - 1)
      li <- if (identical(penalty, "variance")) sr2 / df else
        torch::torch_clamp((sr2 - (1 - 1 / Aa) * (1 - 1 / Bb) * torch::torch_sum(vg)) / df, min = 0)
      add(li)
    }
  }
  if (is.null(total)) total <- torch::torch_zeros(1, dtype = pred$dtype)$squeeze()
  total
}

# ---- routing diversity ---------------------------------------------
# Negative mutual information between individual and expert under the
# routing distribution: mean_i H(q_i) - H(mean_i q_i). Minimising it balances
# the load across experts (high entropy of the average) while making each
# individual's routing decisive (low average entropy), so it penalises both
# collapse onto one expert and collapse to uniform routing. With the base as
# component 0 it is computed over the non-base experts, renormalised, with
# each individual weighted by the share not routed to the base.
.fg_balance <- function(gw, expert_coupling) {
  eps <- 1e-8
  if (isTRUE(expert_coupling)) {
    q <- gw[, 2:as.integer(gw$size(2)), drop = FALSE]
    w <- q$sum(dim = 2L)
    q <- q / (w$unsqueeze(2) + eps)
  } else {
    q <- gw; w <- torch::torch_ones(as.integer(gw$size(1)), dtype = gw$dtype)
  }
  ws <- torch::torch_sum(w) + eps
  H <- function(p) -torch::torch_sum(p * torch::torch_log(torch::torch_clamp(p, min = eps)), dim = -1L)
  qbar <- torch::torch_sum(q * w$unsqueeze(2), dim = 1L) / ws
  torch::torch_sum(H(q) * w) / ws - H(qbar)
}

# ---- disparity-contribution signal ---------------------------------

#' Per-Person Disparity-Contribution Target
#'
#' The quantity whose cross-fitted prediction fairGATE feeds to the gate:
#' for each training individual, the signed direction in which moving that
#' individual's score would close its group's gap relative to the reference
#' group, times its leverage on the decision.
#'
#' @param y Outcome.
#' @param group Protected group codes.
#' @param base Base-learner scores on the response scale, out of fold.
#' @param reference Reference group code.
#' @param family `"binomial"` or `"gaussian"`.
#' @param prevalence Share flagged positive (binomial); defaults to `mean(y)`.
#' @param n_strata Outcome strata for `"gaussian"`.
#' @param bandwidth Bandwidth, on the logit scale, of the Gaussian leverage
#'   kernel centred on the threshold (binomial).
#' @details Binomial: with `t` the prevalence-matched threshold of `base` and
#'   gaps `d_kc = r_kc - r_ref,c` in the rate of positive decisions of group k in
#'   outcome class c, the target is `-d_{g_i y_i} * phi((logit b_i - logit t)/h)`
#'   for a non-reference individual and `+sum_k w_k d_{k y_i} * phi(.)` for a
#'   reference one, `w_k` the non-reference group shares. Gaussian: the same with
#'   the stratum-conditional mean base prediction gap, in outcome standard
#'   deviations, and unit leverage. The target uses the outcome and the
#'   protected attribute, so it must be cross-fitted from the features alone
#'   before it is given to the gate; the protected attribute itself never is.
#' @return A numeric vector, one value per row.
#' @examples
#' set.seed(1)
#' n <- 400
#' g <- sample(0:2, n, replace = TRUE)
#' y <- rbinom(n, 1, 0.4)
#' base <- stats::plogis(rnorm(n) + y - 0.3 * g)   # out-of-fold base scores
#' tgt <- fairgate_disparity_target(y, g, base, reference = 0, family = "binomial")
#' tapply(tgt, list(g, y), mean)
#' @export
fairgate_disparity_target <- function(y, group, base, reference, family = c("binomial", "gaussian"),
                                      prevalence = NULL, n_strata = 5L, bandwidth = 1) {
  family <- match.arg(family)
  g <- as.integer(group); y <- as.numeric(y); b <- as.numeric(base)
  ref <- as.integer(reference)
  if (identical(family, "binomial")) {
    if (is.null(prevalence)) prevalence <- mean(y)
    t <- stats::quantile(b, 1 - prevalence, type = 7, names = FALSE)
    yh <- as.numeric(b >= t)
    s <- as.integer(y == 1) + 1L
    lev <- stats::dnorm((stats::qlogis(pmin(pmax(b, 1e-6), 1 - 1e-6)) -
                           stats::qlogis(min(max(t, 1e-6), 1 - 1e-6))) / bandwidth)
    rate <- function(k, c) { i <- g == k & s == c; if (any(i)) mean(yh[i]) else NA_real_ }
    sd_use <- 1
  } else {
    br <- unique(stats::quantile(y, probs = seq(0, 1, length.out = n_strata + 1L), type = 7))
    s <- if (length(br) < 3L) rep(1L, length(y)) else as.integer(cut(y, br, include.lowest = TRUE, labels = FALSE))
    lev <- rep(1, length(y))
    rate <- function(k, c) { i <- g == k & s == c; if (any(i)) mean(b[i]) else NA_real_ }
    sd_use <- stats::sd(y); if (!is.finite(sd_use) || sd_use <= 0) sd_use <- 1
  }
  ug <- sort(unique(g)); nonref <- setdiff(ug, ref)
  wk <- vapply(nonref, function(k) mean(g == k), 0); wk <- wk / sum(wk)
  out <- numeric(length(y))
  for (c in sort(unique(s))) {
    rr <- rate(ref, c)
    d <- vapply(nonref, function(k) rate(k, c) - rr, 0) / sd_use
    d[!is.finite(d)] <- 0
    for (j in seq_along(nonref)) {
      i <- g == nonref[j] & s == c
      out[i] <- -d[j] * lev[i]
    }
    i <- g == ref & s == c
    out[i] <- sum(wk * d) * lev[i]
  }
  out
}

# ---- multi-group disparity for selection and reporting -------------

#' Multi-Group Separation Disparity: Per Group, Worst Group and Joint
#'
#' Computes, for every non-reference group, the average odds difference
#' (binary, at a supplied threshold) or the conditional mean prediction
#' difference (continuous, in outcome standard deviations) relative to the
#' reference group, and summarises them.
#'
#' @param y Outcome.
#' @param pred Predictions (probabilities for a binary outcome, the outcome
#'   scale for a continuous one).
#' @param group Protected group codes.
#' @param reference Reference group code.
#' @param family `"binomial"` or `"gaussian"`.
#' @param threshold Decision threshold (binomial). Use the prevalence-matched
#'   threshold `quantile(pred, 1 - prevalence)`.
#' @param edges Stratum edges for the continuous outcome; default the
#'   equal-frequency quintiles of `y`.
#' @param scale Outcome standard deviation used to express CMPD.
#' @param eligible Optional vector of group codes over which the worst-group
#'   maximum and the joint summaries are taken (a minimum-cell rule). Groups
#'   outside it are still reported in `per_group`, flagged `eligible = FALSE`.
#'   `NULL` uses every group.
#' @return A list: `per_group` (data frame of group, n, gap), `worst` (the
#'   largest absolute gap), `worst_group`, `signed_worst` (the gap of that
#'   group), `joint_mean` (size-weighted mean absolute gap), `joint_pairwise`
#'   (largest absolute gap between any two groups, reference included) and
#'   `pooled` (every non-reference group pooled against the reference).
#' @examples
#' set.seed(1)
#' n <- 600
#' g <- sample(0:2, n, replace = TRUE)
#' y <- rbinom(n, 1, 0.4)
#' pred <- stats::plogis(rnorm(n) + y - 0.4 * g)
#' thr <- stats::quantile(pred, 1 - mean(y), names = FALSE)   # prevalence-matched
#' d <- fairgate_group_disparity(y, pred, g, reference = 0, family = "binomial",
#'                               threshold = thr)
#' d$per_group
#' d$worst
#' @export
fairgate_group_disparity <- function(y, pred, group, reference, family = c("binomial", "gaussian"),
                                     threshold = NULL, edges = NULL, scale = NULL,
                                     eligible = NULL) {
  family <- match.arg(family)
  y <- as.numeric(y); pred <- as.numeric(pred); g <- as.integer(group); ref <- as.integer(reference)
  if (identical(family, "binomial")) {
    if (is.null(threshold)) stop("`threshold` is required for family = 'binomial'.")
    yh <- as.numeric(pred >= threshold)
    rates <- function(m) c(tpr = if (any(m & y == 1)) mean(yh[m & y == 1]) else NA_real_,
                           fpr = if (any(m & y == 0)) mean(yh[m & y == 0]) else NA_real_)
    gapf <- function(ma, mb) { ra <- rates(ma); rb <- rates(mb); 0.5 * ((ra[["fpr"]] - rb[["fpr"]]) + (ra[["tpr"]] - rb[["tpr"]])) }
  } else {
    if (is.null(scale)) scale <- stats::sd(y)
    if (is.null(edges)) edges <- stats::quantile(y, probs = seq(0, 1, length.out = 6L), names = FALSE)
    br <- unique(edges)
    sid <- as.integer(cut(pmin(pmax(y, br[1]), br[length(br)]), br, include.lowest = TRUE, labels = FALSE))
    gapf <- function(ma, mb) {
      d <- c()
      for (q in sort(unique(sid))) {
        a <- ma & sid == q; b <- mb & sid == q
        if (any(a) && any(b)) d <- c(d, mean(pred[a]) - mean(pred[b]))
      }
      if (!length(d)) NA_real_ else mean(d) / scale
    }
  }
  ug <- sort(unique(g)); nonref <- setdiff(ug, ref)
  per <- data.frame(group = nonref, n = vapply(nonref, function(k) sum(g == k), 0),
                    gap = vapply(nonref, function(k) gapf(g == k, g == ref), 0))
  per$eligible <- if (is.null(eligible)) TRUE else per$group %in% as.integer(eligible)
  ok <- is.finite(per$gap) & per$eligible
  worst_i <- if (any(ok)) which(ok)[which.max(abs(per$gap[ok]))] else NA_integer_
  pw <- c()
  ugp <- if (is.null(eligible)) ug else ug[ug %in% c(as.integer(eligible), ref)]
  for (i in seq_along(ugp)) for (j in seq_along(ugp)) if (i < j) pw <- c(pw, gapf(g == ugp[j], g == ugp[i]))
  list(per_group = per,
       worst = if (is.na(worst_i)) NA_real_ else abs(per$gap[worst_i]),
       worst_group = if (is.na(worst_i)) NA_integer_ else per$group[worst_i],
       signed_worst = if (is.na(worst_i)) NA_real_ else per$gap[worst_i],
       joint_mean = if (any(ok)) sum(per$n[ok] * abs(per$gap[ok])) / sum(per$n[ok]) else NA_real_,
       joint_pairwise = if (any(is.finite(pw))) max(abs(pw[is.finite(pw)])) else NA_real_,
       pooled = gapf(g != ref, g == ref))
}
