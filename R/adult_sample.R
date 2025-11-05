#' Adult dataset sample (n = 800)
#'
#' Cleaned subset of the UCI Adult data used in the vignette.
#' Outcome `income` is 1 (>50K) / 0 (<=50K). Protected attribute: `sex`.
#'
#' @format A data frame with 800 rows and (subset of) columns:
#' \describe{
#'   \item{age}{numeric}
#'   \item{education_num}{numeric}
#'   \item{capital_gain}{numeric}
#'   \item{capital_loss}{numeric}
#'   \item{hours_per_week}{numeric}
#'   \item{workclass, education, marital_status, occupation, relationship, race, sex, native_country}{character}
#'   \item{income}{integer: 0/1}
#' }
#' @source UCI Adult dataset
#' @usage data(adult_sample)
"adult_sample"
