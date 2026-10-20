# fairGATE

fairGATE (Fair Gated Algorithm for Targeted Equity) corrects an existing
prediction model so that its errors are spread more evenly across protected
groups, and reports, person by person, how far and how reliably each
prediction was moved. It works for binary and continuous outcomes, for two or
more groups and for intersections of two attributes, in any domain where a
prediction is used to make decisions about people: health care, income and
credit, or the review of expert judgements.

Developed by Rhys Holland under the supervision of Dr Raquel Iniesta, King's
College London (https://www.kcl.ac.uk/people/raquel-iniesta).

## Installation

```r
install.packages("fairGATE")
# development version
# remotes::install_github("rhysholland/FairGATE")

# the neural-network components need the LibTorch backend once:
torch::install_torch()
```

The base learner uses the xgboost package (`base_learner = "xgboost"`) or
torch (`base_learner = "nn"`); both are suggested dependencies.

## How it works

1. A **base learner** is fitted without the protected attribute and
   cross-fitted: training rows get out-of-fold scores, held-out rows the
   prediction of a fit on all training rows (`fairgate_base_learner()`;
   XGBoost or a neural network, each with its own tuning grid).
2. A **gate** weights the base learner and a set of neural **experts** for each
   person; each expert proposes a correction to the base score. The untrained
   model is exactly the base learner.
3. The correction is trained under a **fairness penalty**: a random-effects or
   variance penalty on outcome-stratified group means, or a thresholded
   worst-group penalty on error rates, plus an intervention cost for moving
   people away from the base score. A cross-fitted **disparity signal** tells
   the gate where corrections would close group gaps; the protected attribute
   itself is never an input.
4. A configuration is chosen on an inner split by one rule: the smallest
   worst-group disparity among configurations within 0.01 of the best
   validation AUC or R-squared.
5. Every held-out person receives a base score, a corrected score, the
   adjustment, the routing weights and a bootstrap **stability interval** for
   the adjustment (`fairgate_export()`).

## Quick start

```r
library(fairGATE)
torch::torch_set_num_threads(1)          # reproducible fits

set.seed(1)
n <- 2000
g <- sample(0:2, n, replace = TRUE, prob = c(0.6, 0.25, 0.15))
X <- cbind(x1 = rnorm(n) + 0.5 * g, x2 = rnorm(n), x3 = rnorm(n))
y <- rbinom(n, 1, plogis(X[, 1] - 0.5 * X[, 2]))

fit <- fairgate(X, y, group = g, family = "binomial", reference_group = 0,
                base_learner = "xgboost")   # or "nn"
fit
head(fairgate_export(fit))
```

`fairgate()` uses the full protocol of the accompanying study by default (48
searched configurations, up to 300 epochs, 10 bootstrap replicates), which
takes minutes on tens of thousands of rows. For a first look pass, for
example, `n_draws = 4, epochs = 50, interval_reps = 2`.

The building blocks are exported for custom pipelines:
`fairgate_base_learner()` and `fairgate_tune_base_learner()` for the base
learner, `train_gnn()` for the correction with supplied base scores,
`fairgate_disparity_target()`, `fairgate_group_disparity()` and the penalty
functions, and `analyse_experts()`, `analyse_gnn_results()`, `plot_sankey()`
and `export_f360_csv()` (AI Fairness 360 format) for analysis.

See `vignette("introduction-to-fairGATE")` for a worked example on the Adult
income sample shipped with the package.

## Citation

Holland R, Iniesta R. fairGATE: Fair Gated Algorithm for Targeted Equity. R
package version 1.0.0. https://github.com/rhysholland/FairGATE

## Licence

MIT.
