if (getRversion() >= "2.15.1")  utils::globalVariables(
  c("group", "feature", "importance", "avg_importance", "difference",
    "reorder", "true", "specificity", "sensitivity", "prob", "prob_bin",
    "mean_predicted_prob", "observed_proportion", "gate_prob_expert_1",
    "group_label", "gate_entropy", "Actual_Group", "Feature_Profile",
    "Assigned_Expert", "N", "stratum", "subjectid", "self", ".")
)
#' @keywords internal
#' @importFrom stats complete.cases
NULL

utils::globalVariables(c(
  "fpr", "tpr", "spread",         # analysis columns
  "prob_bin", "mean_predicted_prob", "observed_proportion", # calib table cols
  "group_label"                    # used in dplyr/ggplot
))

utils::globalVariables(c(
  "Actual_Group", "Feature_Profile", "Assigned_Expert", "N",
  "group_A_score", "group_B_score"
))

utils::globalVariables(c("id", "group_A_score", "group_B_score"))

# internal helper
.inform <- function(..., .verbose = FALSE) {
  if (isTRUE(.verbose)) message(...)
}
