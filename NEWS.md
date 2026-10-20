# fairGATE 1.0.0

This release turns fairGATE from a gated mixture of experts trained under a
fairness penalty into a method that corrects an independently fitted
prediction model, for binary and continuous outcomes and for any number of
protected groups. Every change since 0.1.1 is listed below. No exported
function or argument of 0.1.1 has been removed or renamed, and new arguments
of `train_gnn()` follow the old ones, so calls written for 0.1.1 still run.
The changes that can alter their results or the structure of their output
are listed under "Breaking changes".

Version numbers 0.2.0 to 0.5.0 that appear in the help pages and tests are
internal development versions that were not released on CRAN; 1.0.0 is the
first release to contain their changes.

## Correcting a base learner

* `fairgate_base_learner()` fits a cross-fitted base learner without the
  protected attribute: 5-fold out-of-fold scores for training rows and a fit on
  all training rows for held-out rows, so that no training row is scored by a
  model that saw its outcome. The learner is selectable:
  `base_learner = "xgboost"` (gradient-boosted trees, package xgboost) or
  `base_learner = "nn"` (a torch multilayer perceptron). Each has its own
  hyperparameter grid, listed by `fairgate_base_grid()`.
* `fairgate_tune_base_learner()` tunes either learner on an inner split, by
  grid or seeded random search with early stopping on the validation slice,
  and selects by validation loss; AUC or R-squared is reported beside it.
  Fixed hyperparameters can be passed instead.
* `train_gnn(base_score = , coupling = )` corrects the supplied base scores.
  With `coupling = "residual"` the gate weights the base learner and the
  experts, each expert proposing a correction to the base score; with
  `coupling = "expert"` the base score is one component of the mixture; with
  `coupling = "offset"` the mixture is added to the base score. The expert
  output layers start at zero, so the untrained model is exactly the base
  learner, and an individual routed entirely to the base learner receives its
  score back exactly.
* `fairgate()` runs the whole method in one call: the base learner, the
  disparity signal, a search over 48 seeded random configurations of
  `fairgate_space()` scored on an inner split, selection of the configuration
  with the smallest worst-group disparity among those within 0.01 of the best
  validation AUC or R-squared, and the final fit on every partition. The
  defaults are those of the accompanying study. `fairgate_draw_configs()`
  returns the configurations searched.

## Outcomes, groups and penalties

* Continuous outcomes (`family = "gaussian"`), with squared-error loss and a
  penalty on outcome-stratified conditional mean predictions (`n_strata`
  strata, five by default in `fairgate()`).
* Any number of protected groups, and intersections of two attributes through
  `crossed_groups` (`fairgate_crossed_plan()`, `fairgate_crossed_penalty()`).
* Three penalty families selected by `penalty`: the random-effects
  between-group variance component (DerSimonian and Laird), which corrects the
  spread of group means for their sampling noise and is the new default; the
  variance of the group means of earlier versions (`"variance"`); and a
  thresholded worst-group penalty (`"thresholded"`) on the gaps in soft true-
  and false-positive rates at the prevalence-matched threshold, or on the
  worst group's conditional mean gap for a continuous outcome
  (`fairgate_threshold_plan()`, `fairgate_threshold_penalty()`,
  `fairgate_group_gap_penalty()`). `lambda_thr` adds the thresholded term to a
  variance-component penalty; `reference_group` names the reference group.
* The penalties can be evaluated and differentiated outside a fit with
  `fairgate_penalty_from_cells()`, `fairgate_penalty_plan()` and
  `fairgate_fair_penalty()`.
* `intervention_cost` charges the mean absolute link-scale distance from the
  base learner, so that people are moved only where the fairness gain
  justifies it.

## Disparity signal and gate options

* `fairgate_disparity_target()` computes, for each training person, the signed
  direction in which moving their score would close their group's gap to the
  reference group, weighted by their leverage on the decision. A cross-fitted
  prediction of this target from the features and the base score can be given
  to the gate (`gate_signal`); the protected attribute itself is never an
  input to the base learner, the gate or the experts.
* Sparse gates (`gate = "sparsemax"` or `"entmax15"`, `fairgate_gate_map()`),
  a routing-balance regulariser (`balance`) and an adversary on the routing
  weights (`adversary`) are available as options.

## Training

* `splits` accepts precomputed partitions, so several models can be compared
  on identical data.
* Standardisation and median imputation (`standardise`, `impute`) are computed
  from the training rows of each partition only.
* Early stopping (`early_stopping`, `es_prop`, `es_min_epochs`, `es_patience`)
  on a slice of the training rows stratified on outcome and group, with an
  optional trailing moving average of the monitored objective (`es_smooth`).
* `alpha` and `lambda_max` add an auxiliary per-expert objective on a ladder
  of fairness weights (`fairgate_lambda_ladder()`).

## Outputs and analyses

* `attribution`, a per-person export for every held-out row: base score,
  prediction, adjustment, base weight and routing weights.
* Per-person stability intervals for the adjustment from warm-started
  bootstrap replicates (`interval_reps`, `interval_epochs`, `interval_level`;
  columns `adj_lo`, `adj_hi`, `adj_sd`). `fairgate_export()` returns the
  export of a `fairgate()` fit, flags intervals that exclude zero, and
  optionally writes it to a CSV file.
* `fairgate_group_disparity()` reports per-group, worst-group, joint and
  pooled separation disparity (average odds difference, or conditional mean
  prediction difference in outcome standard deviations), with an optional
  minimum-cell rule.
* Routing analyses: the export carries the full routing vector, the weight
  on the base learner (`base_weight`) and, with sparse gates, the number of
  active components (`n_active`) and whether a person was routed to the base
  learner alone (`base_only`), so that who is corrected, by which expert and
  by how much can be tabulated by group. The interfaces of
  `analyse_experts()`, `analyse_gnn_results()` and `plot_sankey()` are
  unchanged.
* `performance_summary` reports the mean, standard deviation and 2.5th and
  97.5th percentiles across partitions, with per-partition values in
  `performance_by_split`.
* `train_gnn()` also returns `attribution`, `performance_by_split`,
  `early_stopping` and `settings`; the elements returned by 0.1.1 are kept.

## Breaking changes

* The default `penalty` is now `"random_effects"`. Pass
  `penalty = "variance"` to reproduce results of earlier versions.
* `performance_summary` from `train_gnn()` has columns `Metric`, `Value`,
  `SD`, `Lower`, `Upper` and `Splits`. `Value` is now the mean of the
  per-partition AUC (or R-squared and RMSE), where 0.1.1 reported a single
  AUC pooled over all held-out predictions.
* The Brier score is no longer part of `performance_summary`; discrimination
  is summarised by AUC (binary) or R-squared and RMSE (continuous).
  `analyse_gnn_results()` computes the Brier score from the held-out
  predictions instead, so `$brier` and the calibration plot subtitle are
  still filled for a binary outcome.

## Bug fixes

* `export_f360_csv()` now merges gate weights onto predictions by partition
  and person; previously, with repeated splits, routing diagnostics could be
  attached to the wrong person.
* Without a `subjectid` column, predictions are now identified by the row of
  the supplied data rather than by their position within a fold.

## Documentation

* Help pages with runnable examples for every exported function, and a new
  vignette on public data (the Adult income sample shipped with the package)
  in place of the earlier clinical example.

# fairGATE 0.1.1

* First CRAN release.
