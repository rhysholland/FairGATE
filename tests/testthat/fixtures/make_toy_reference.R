# Reference fits for test-reproduction.R, made with the fairGATE 0.5.0 package
# (the version used in the accompanying study) on simulated toy data, one
# torch thread. Run from tests/testthat:
#   R_LIBS_USER=<library holding fairGATE 0.5.0 and torch> Rscript fixtures/make_toy_reference.R
suppressPackageStartupMessages({ library(torch); library(fairGATE) })
stopifnot(as.character(packageVersion("fairGATE")) == "0.5.0")
torch_set_num_threads(1L)
source("helper-toy.R")
ref <- lapply(toy_cases(), function(cs) {
  f <- fit_toy_case(cs)
  list(prob = f$final_results$prob,
       attribution = as.data.frame(f$attribution)[, intersect(c("base", "pred", "adjustment", "base_weight",
                                                                "adj_lo", "adj_hi", "adj_sd"),
                                                              names(f$attribution))])
})
ref$version <- as.character(packageVersion("fairGATE"))
ref$torch <- as.character(packageVersion("torch"))
saveRDS(ref, "fixtures/toy_reference.rds", version = 2)
