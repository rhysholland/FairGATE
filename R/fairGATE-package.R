#' fairGATE: Fair Gated Algorithm for Targeted Equity
#'
#' fairGATE corrects an independently fitted prediction model for fairness
#' across protected groups. A cross-fitted base learner (XGBoost or a neural
#' network, [fairgate_base_learner()]) is fitted without the protected
#' attribute; a gated mixture of neural experts ([train_gnn()]) then learns a
#' per-person correction to its score under a fairness penalty, for binary or
#' continuous outcomes and two or more groups. [fairgate()] runs the whole
#' method, including the configuration search and selection rule of the
#' accompanying study, in one call, and [fairgate_export()] returns the
#' per-person adjustments with their stability intervals.
#'
#' The neural-network components use the 'torch' package, which needs the
#' LibTorch backend (`torch::install_torch()`). For reproducible fits call
#' `torch::torch_set_num_threads(1)` first.
#'
#' @keywords internal
"_PACKAGE"
