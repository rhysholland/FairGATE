#' Between-Group Variance Components for the fairGATE Fairness Penalty
#'
#' Computes one stratum's contribution to the fairGATE fairness penalty from
#' the group-conditional mean predictions of that stratum and their sampling
#' variances. This is the lowest level of the penalty and is exported so that
#' the algebra can be exercised directly on constructed inputs, independently
#' of any model fit.
#'
#' @param mhat Group-conditional mean predictions for one outcome stratum: a
#'   `torch_tensor` of length `K_q`, or a numeric vector of the same length.
#'   When a tensor carrying a gradient is supplied, the returned value is
#'   differentiable with respect to it. A numeric vector is evaluated in base
#'   R and does not need torch.
#' @param vhat Sampling variances of the entries of `mhat`, that is
#'   `s2_kq / n_kq` where `s2_kq` is the within-cell sample variance of the
#'   individual predictions (denominator `n_kq - 1`). A numeric vector or a
#'   `torch_tensor` of the same length as `mhat`. Ignored, and may be `NULL`,
#'   when `penalty = "variance"`. Always used detached from the autograd graph;
#'   see Details.
#' @param penalty One of `"random_effects"` (default), `"bias_corrected"` or
#'   `"variance"`. See Details.
#' @param eps Lower bound applied to `vhat` before it is inverted. Default
#'   `1e-12`.
#'
#' @details
#' Write the cell \eqn{(k, q)} for the individuals in protected group \eqn{k}
#' and outcome stratum \eqn{q}, of size \eqn{n_{kq}}, with mean prediction
#' \eqn{\hat m_{kq}}, within-cell sample variance \eqn{s^2_{kq}} and sampling
#' variance \eqn{\hat v_{kq} = s^2_{kq} / n_{kq}}. Let \eqn{K_q} be the number
#' of groups contributing to stratum \eqn{q}. The three rungs are
#'
#' \describe{
#'   \item{`"variance"`}{\eqn{L_q = \mathrm{Var}_k(\hat m_{kq})}, the unbiased
#'     sample variance over the \eqn{K_q} group means. This is the penalty of
#'     fairGATE 0.2.0 and earlier.}
#'   \item{`"bias_corrected"`}{\eqn{L_q = \max(0, \mathrm{Var}_k(\hat m_{kq})
#'     - K_q^{-1} \sum_k \hat v_{kq})}.}
#'   \item{`"random_effects"`}{the DerSimonian and Laird between-group variance
#'     component. With \eqn{w_{kq} = 1 / \hat v_{kq}},
#'     \eqn{\bar m_{wq} = \sum_k w_{kq} \hat m_{kq} / \sum_k w_{kq}},
#'     \eqn{Q_q = \sum_k w_{kq} (\hat m_{kq} - \bar m_{wq})^2} and
#'     \eqn{C_q = \sum_k w_{kq} - (\sum_k w_{kq}^2) / (\sum_k w_{kq})},
#'     \eqn{L_q = \max(0, (Q_q - (K_q - 1)) / C_q)}.}
#' }
#'
#' The correction terms follow from the bias of the sample variance as an
#' estimator of the between-group dispersion. Writing
#' \eqn{\hat m_k = \mu_k + e_k} with \eqn{e_k} independent, mean zero and
#' variance \eqn{v_k}, the unbiased sample variance \eqn{S^2} over the \eqn{K}
#' estimates satisfies \eqn{E[S^2] = S^2(\mu) + K^{-1} \sum_k v_k}. The second
#' term is a floor that no improvement in fairness can remove: an optimiser can
#' only reduce it by distorting predictions.
#'
#' `vhat`, and therefore the weights and the constant \eqn{C_q}, are always
#' detached from the autograd graph. This is load-bearing rather than an
#' optimisation. If they were differentiable, the optimiser could lower the
#' penalty by inflating the within-cell prediction variance, which raises
#' \eqn{\hat v_{kq}} and shrinks the weighted departure from the pooled mean
#' without making any group's mean prediction closer to any other's. Gradients
#' therefore flow only through `mhat`.
#'
#' The rungs nest exactly. If every \eqn{\hat v_{kq}} equals a common \eqn{v},
#' then \eqn{\sum w = K/v}, \eqn{\sum w^2 = K/v^2} and \eqn{C = (K-1)/v}, while
#' \eqn{Q = (K-1) S^2 / v}, so the random-effects rung collapses to
#' \eqn{\max(0, S^2 - v)}, which is the bias-corrected rung; and as
#' \eqn{v \to 0} that collapses in turn to \eqn{S^2}, the 0.2.0 penalty.
#'
#' @return A scalar: a `torch_tensor` if `mhat` was a tensor, otherwise a
#'   numeric value. Zero when fewer than two groups are supplied.
#'
#' @seealso [fairgate_fair_penalty()] for the same quantity computed from
#'   individual predictions, and [train_gnn()] for its use in fitting.
#' @examples
#' # Three groups' mean predictions in one outcome stratum, with their
#' # sampling variances
#' mhat <- c(0.30, 0.35, 0.42)
#' vhat <- c(0.002, 0.004, 0.010)
#' fairgate_penalty_from_cells(mhat, penalty = "variance")
#' fairgate_penalty_from_cells(mhat, vhat, penalty = "bias_corrected")
#' fairgate_penalty_from_cells(mhat, vhat, penalty = "random_effects")
#' @export
fairgate_penalty_from_cells <- function(mhat, vhat = NULL,
                                        penalty = c("random_effects",
                                                    "bias_corrected",
                                                    "variance"),
                                        eps = 1e-12) {
  penalty <- match.arg(penalty)
  was_tensor <- inherits(mhat, "torch_tensor")
  if (!was_tensor) {
    # Numeric input: the same algebra in base R, so that the penalty can be
    # evaluated without torch.
    m <- as.numeric(mhat)
    K <- length(m)
    if (K < 2L) return(0)
    s2 <- stats::var(m)
    if (identical(penalty, "variance")) return(s2)
    if (is.null(vhat)) {
      stop("`vhat` is required unless penalty = \"variance\".")
    }
    v <- if (inherits(vhat, "torch_tensor")) {
      as.numeric(torch::as_array(vhat$detach()$cpu()))
    } else {
      as.numeric(vhat)
    }
    if (length(v) != K) {
      stop("`vhat` must have the same length as `mhat`.")
    }
    v <- pmax(v, eps)
    if (identical(penalty, "bias_corrected")) {
      return(max(0, s2 - mean(v)))
    }
    w <- 1 / v
    sw <- sum(w)
    mbar <- sum(w * m) / sw
    qstat <- sum(w * (m - mbar)^2)
    cconst <- sw - sum(w * w) / sw
    return(max(0, (qstat - (K - 1)) / cconst))
  }
  K <- as.integer(prod(unlist(mhat$size())))
  if (K < 2L) {
    z <- torch::torch_zeros(1, dtype = mhat$dtype)$squeeze()
    return(if (was_tensor) z else 0)
  }

  s2 <- torch::torch_var(mhat)          # unbiased, denominator K - 1

  if (identical(penalty, "variance")) {
    out <- s2
  } else {
    if (is.null(vhat)) {
      stop("`vhat` is required unless penalty = \"variance\".")
    }
    v <- if (inherits(vhat, "torch_tensor")) {
      vhat$detach()$to(dtype = mhat$dtype)
    } else {
      torch::torch_tensor(as.numeric(vhat), dtype = mhat$dtype)
    }
    if (as.integer(prod(unlist(v$size()))) != K) {
      stop("`vhat` must have the same length as `mhat`.")
    }
    v <- torch::torch_clamp(v, min = eps)
    if (identical(penalty, "bias_corrected")) {
      out <- torch::torch_clamp(s2 - torch::torch_mean(v), min = 0)
    } else {
      w   <- 1 / v
      sw  <- torch::torch_sum(w)
      sw2 <- torch::torch_sum(w * w)
      mbar <- torch::torch_sum(w * mhat) / sw
      qstat <- torch::torch_sum(w * (mhat - mbar)^2)
      cconst <- sw - sw2 / sw
      out <- torch::torch_clamp((qstat - (K - 1)) / cconst, min = 0)
    }
  }
  if (was_tensor) out else as.numeric(out)
}


#' Cell Structure for the fairGATE Fairness Penalty
#'
#' Builds the fixed part of the fairness penalty for one set of rows: which
#' group-by-stratum cells contribute, and the indicator matrices and cell sizes
#' needed to form the group-conditional means. The result depends only on the
#' protected attribute and the outcome strata, so it is computed once per
#' partition and reused at every epoch.
#'
#' @param group Integer or numeric vector of protected group codes, one per row.
#' @param strata Integer vector of outcome stratum identifiers in `1:Q`, one per
#'   row. For a binary outcome these are the two outcome classes.
#' @param Q Number of outcome strata. Defaults to `max(strata)`.
#' @param min_cell Smallest cell size that may contribute. Cells smaller than
#'   this are dropped. Use `2` whenever a within-cell variance is needed, that
#'   is for every rung except `"variance"`.
#' @param dtype A torch dtype for the indicator matrices. Defaults to
#'   `torch::torch_float()`.
#'
#' @details
#' A group contributes to stratum `q` only if its cell in that stratum has at
#' least `min_cell` rows, and a stratum contributes only if at least two groups
#' contribute to it. Strata are treated independently, so a group absent from
#' one stratum still contributes to the others. Note that the 0.2.0 penalty
#' used a stricter global rule in which a group had to be present in every
#' stratum to contribute to any of them; `train_gnn(penalty = "variance")`
#' retains that rule for exact backward compatibility.
#'
#' @return A list with one element per contributing stratum, each a list with
#'   the stratum index `q`, the contributing group codes, the indicator matrix
#'   `M` (rows by contributing groups), the cell sizes `n`, and `K` the number
#'   of contributing groups. An empty list when no stratum contributes.
#' @examples
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
#'   set.seed(1)
#'   group <- sample(0:2, 60, replace = TRUE)
#'   strata <- sample(1:2, 60, replace = TRUE)
#'   plan <- fairgate_penalty_plan(group, strata)
#'   length(plan)
#' }
#' }
#' @export
fairgate_penalty_plan <- function(group, strata, Q = max(as.integer(strata)),
                                  min_cell = 2L,
                                  dtype = torch::torch_float()) {
  g <- as.integer(group)
  s <- as.integer(strata)
  if (length(g) != length(s)) stop("`group` and `strata` must be the same length.")
  n <- length(g)
  Q <- as.integer(Q)
  min_cell <- as.integer(min_cell)
  ug <- sort(unique(g))
  out <- list()
  for (q in seq_len(Q)) {
    in_q <- s == q
    ks <- ug[vapply(ug, function(k) sum(in_q & g == k) >= min_cell, logical(1))]
    if (length(ks) < 2L) next
    M <- matrix(0, nrow = n, ncol = length(ks))
    for (j in seq_along(ks)) M[, j] <- as.numeric(in_q & g == ks[j])
    out[[length(out) + 1L]] <- list(
      q = q,
      groups = ks,
      M = torch::torch_tensor(M, dtype = dtype),
      n = torch::torch_tensor(colSums(M), dtype = dtype),
      n_num = colSums(M),
      K = length(ks)
    )
  }
  out
}


#' fairGATE Fairness Penalty from Individual Predictions
#'
#' Evaluates the total fairness penalty `sum_q L_q` on a vector of individual
#' predictions, on the scale on which the penalty is defined: probabilities for
#' a binary outcome, the response scale for a continuous one.
#'
#' @param pred Individual predictions: a `torch_tensor` of length `n`, or a
#'   numeric vector. A tensor carrying a gradient gives a differentiable result.
#' @param group Protected group codes, one per row, or `NULL` when `plan` is
#'   supplied.
#' @param strata Outcome stratum identifiers in `1:Q`, one per row, or `NULL`
#'   when `plan` is supplied.
#' @param Q Number of outcome strata. Defaults to `max(strata)`.
#' @param penalty One of `"random_effects"` (default), `"bias_corrected"` or
#'   `"variance"`; see [fairgate_penalty_from_cells()].
#' @param eps Lower bound applied to the cell sampling variances before they are
#'   inverted. Default `1e-12`.
#' @param plan Optional precomputed cell structure from
#'   [fairgate_penalty_plan()]. Supplying it avoids rebuilding the indicator
#'   matrices, which is what the training loop does.
#'
#' @details
#' Per cell, the mean prediction is differentiable and the sampling variance is
#' not: the within-cell variance is computed from detached predictions, so the
#' weights of the random-effects rung cannot be gamed by inflating within-cell
#' spread. Cells with fewer than two rows carry no estimable within-cell
#' variance and are dropped for every rung except `"variance"`, which needs
#' none and admits cells of one row.
#'
#' @return A scalar of the same kind as `pred` (tensor in, tensor out), zero
#'   when no stratum contributes.
#' @examples
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
#'   set.seed(1)
#'   group <- sample(0:2, 200, replace = TRUE)
#'   y <- rbinom(200, 1, 0.4)
#'   pred <- torch::torch_tensor(stats::plogis(rnorm(200) + 0.3 * group), requires_grad = TRUE)
#'   pen <- fairgate_fair_penalty(pred, group, y + 1L, penalty = "random_effects")
#'   pen$backward()
#'   as.numeric(pen)
#' }
#' }
#' @export
fairgate_fair_penalty <- function(pred, group = NULL, strata = NULL,
                                  Q = NULL,
                                  penalty = c("random_effects",
                                              "bias_corrected", "variance"),
                                  eps = 1e-12, plan = NULL) {
  penalty <- match.arg(penalty)
  was_tensor <- inherits(pred, "torch_tensor")
  if (!was_tensor) {
    pred <- torch::torch_tensor(as.numeric(pred), dtype = torch::torch_float64())
  }
  if (is.null(plan)) {
    if (is.null(group) || is.null(strata)) {
      stop("Supply either `plan`, or both `group` and `strata`.")
    }
    if (is.null(Q)) Q <- max(as.integer(strata))
    plan <- fairgate_penalty_plan(
      group, strata, Q,
      min_cell = if (identical(penalty, "variance")) 1L else 2L,
      dtype = pred$dtype
    )
  }
  total <- NULL
  for (cell in plan) {
    mhat <- torch::torch_matmul(cell$M$t()$to(dtype = pred$dtype), pred) /
      cell$n$to(dtype = pred$dtype)
    vhat <- NULL
    if (!identical(penalty, "variance")) {
      # Detached: the sampling variance must not be a channel the optimiser can
      # push on. See fairgate_penalty_from_cells() Details.
      pd <- pred$detach()
      md <- mhat$detach()
      dev <- pd$unsqueeze(2) - md$unsqueeze(1)
      ss <- torch::torch_sum(cell$M$to(dtype = pred$dtype) * dev^2, dim = 1)
      s2 <- ss / (cell$n$to(dtype = pred$dtype) - 1)
      vhat <- s2 / cell$n$to(dtype = pred$dtype)
    }
    l_q <- fairgate_penalty_from_cells(mhat, vhat, penalty = penalty, eps = eps)
    total <- if (is.null(total)) l_q else total + l_q
  }
  if (is.null(total)) {
    total <- torch::torch_zeros(1, dtype = pred$dtype)$squeeze()
  }
  if (was_tensor) total else as.numeric(total)
}


#' Frontier Ladder of Expert Fairness Weights
#'
#' The auxiliary fairness weights carried by the individual experts of the
#' frontier-anchored model.
#'
#' @param K Number of experts, which is the number of protected groups.
#' @param lambda_max Fairness weight of the most heavily constrained expert.
#'
#' @details
#' Expert 1 is the accuracy anchor and carries \eqn{\lambda_1 = 0}. The
#' remaining experts are spaced geometrically,
#' \eqn{\lambda_k = \lambda_{\max} 10^{-(K - k)/(K - 1)}} for \eqn{k \ge 2}, so
#' that \eqn{\lambda_K = \lambda_{\max}}. The experts therefore sit at
#' different points of the fairness and accuracy trade-off by construction,
#' rather than having to discover a division of labour from a symmetric
#' starting point that the gate gradient cannot leave.
#'
#' The gate gradient is the reason the anchoring is structural rather than
#' cosmetic. With \eqn{\hat y_i = \sum_k \pi_{ik} f_{ik}} and
#' \eqn{\pi_i = \mathrm{softmax}(z_i / \tau)},
#' \eqn{\partial \hat y_i / \partial z_{il} =
#' \tau^{-1} \pi_{il} (f_{il} - \hat y_i)}, so the gradient reaching a gate
#' logit is proportional to that expert's departure from the mixture. Identical
#' experts make it exactly zero: the gate cannot specialise and the experts
#' never differentiate.
#'
#' @return A numeric vector of length `K`, increasing, with first element zero
#'   and last element `lambda_max`. Length-one input returns `0`.
#' @examples
#' fairgate_lambda_ladder(K = 4, lambda_max = 16)
#' @export
fairgate_lambda_ladder <- function(K, lambda_max) {
  K <- as.integer(K)
  if (K < 1L) stop("`K` must be at least 1.")
  if (K == 1L) return(0)
  lambda_max <- as.numeric(lambda_max)
  if (!is.finite(lambda_max) || lambda_max < 0) {
    stop("`lambda_max` must be a finite non-negative number.")
  }
  k <- seq_len(K)
  lam <- lambda_max * 10^(-(K - k) / (K - 1))
  lam[1] <- 0
  lam
}
