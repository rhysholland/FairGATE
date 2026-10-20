## Submission

This is an update of fairGATE from 0.1.1 to 1.0.0. The maintainer and the
maintainer address are unchanged (Rhys Holland <rhys.holland@icloud.com>), as
is the author list (Rhys Holland, Raquel Iniesta).

The jump from 0.1.1 to 1.0.0 reflects a change in what the package does: it
now corrects an independently fitted, cross-fitted prediction model (XGBoost
or a torch neural network, both in Suggests and checked at run time) with a
gated mixture of experts, for binary and continuous outcomes and any number of
protected groups, and it adds further fairness penalties, per-person
adjustment intervals and a one-call wrapper, `fairgate()`. This is the version
described in an accompanying journal submission. NEWS.md lists every
user-visible change.

No exported function, dataset or argument of 0.1.1 has been removed or
renamed, and the new arguments of `train_gnn()` follow the old ones, so code
written for 0.1.1 still runs. The breaking changes, listed in NEWS.md under
"Breaking changes", are: the default fairness penalty is now the
random-effects variance component (`penalty = "variance"` restores the 0.1.1
behaviour); and `train_gnn()$performance_summary` now reports the mean, SD and
2.5th and 97.5th percentiles of per-partition AUC (or R-squared and RMSE)
instead of a pooled AUC and Brier score.

## Test environments

* local macOS 27.0.1 (aarch64-apple-darwin20), R 4.4.3, with torch 0.17.0
  (LibTorch installed) and xgboost 3.4.2.1; TinyTeX 2026.10 for the PDF
  manual; HTML Tidy 5.9.14
* local macOS 27.0.1 (aarch64-apple-darwin20), R 4.5.3, without torch and
  xgboost (no Suggests other than knitr, rmarkdown and testthat); TinyTeX
  2026.10; HTML Tidy 5.9.14; V8 8.2.0 for the math rendering check
* local macOS 27.0.1, R 4.4.3, without torch and xgboost

All checks were `R CMD check --as-cran` on the built tarball, with the PDF
manual and with `_R_CHECK_CRAN_INCOMING_REMOTE_=true`.

## R CMD check results

0 errors | 0 warnings | 1 note (R 4.5.3)

* checking for future file timestamps ... NOTE
  unable to verify current time

  The test machine cannot reach the time server queried by this check; the
  note is about the local check environment, not the package.

On R 4.4.3 with all Suggests installed, the only other note was "Skipping
checking math rendering: package 'V8' unavailable" (V8 was not installed in
that library); the math rendering check passed on R 4.5.3.

## Check without suggested packages

With torch and xgboost not installed and `_R_CHECK_FORCE_SUGGESTS_=false`, the
check gives 0 errors and 0 warnings (only the notes above and the
information that torch and xgboost are not available). Examples run, tests
that need torch or xgboost are skipped (testthat: 39 passed, 9 skipped,
0 failed) and the vignette rebuilds, showing its model-fitting code without
evaluating it.

## Reverse dependencies

There are no reverse dependencies on CRAN
(`tools::package_dependencies("fairGATE", reverse = TRUE, which = "all")`).

## Notes for the reviewers

* torch and xgboost are in Suggests. Every function that needs them checks
  for them with `requireNamespace()` and stops with an informative message.
  Examples that need torch are wrapped in `\donttest{}` and guarded by
  `requireNamespace("torch")` and `torch::torch_is_installed()`; the xgboost
  examples are guarded by `requireNamespace("xgboost")`.
* The vignette evaluates its model-fitting chunks only when torch (with
  LibTorch) and xgboost are available; otherwise it shows the code.
* Tests that need torch or xgboost skip when they are unavailable; the slower
  ones use `skip_on_cran()`.
* Nothing is written outside `tempdir()`: `train_gnn(save_outputs = FALSE)` is
  the default and `fairgate_export()` writes only to a path the user gives.
* All functions that set a random seed restore the caller's random number
  state on exit, except `train_gnn(seed = )`, which keeps its 0.1.1 behaviour.
