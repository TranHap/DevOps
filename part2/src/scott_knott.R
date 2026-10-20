#!/usr/bin/env Rscript
# Scott-Knott ESD ranking of the 4 ranked models over the resamples (brief Section 8).
#
#   Rscript src/scott_knott.R [results_dir]        (default: results; run from part2/)
#
# Input : <results_dir>/runs.csv (one row per seed x model; written by src/run_resamples.py)
# Output: <results_dir>/sk_{f1,auc}.csv                 groups (1 = best) with median / mean per model
#         <results_dir>/sk_input_{f1,auc}.csv           the wide data frame given to sk_esd (rows = seeds)
#         <results_dir>/sk_{f1,auc}_effect_sizes.csv    checkDifference() (Cohen's d) + Cliff's delta per pair
#         <results_dir>/figures/sk_{f1,auc}.png         the package's plot
#         <results_dir>/sk_session_info.txt             R and package versions
suppressPackageStartupMessages({
  library(ScottKnottESD)
  library(effsize)
})
stopifnot(packageVersion("ScottKnottESD") >= "3.0.0")

args <- commandArgs(trailingOnly = TRUE)
res_dir <- if (length(args) >= 1) args[1] else "results"
fig_dir <- file.path(res_dir, "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

ranked <- c(rf = "RF_MCV", lstm = "LSTM_BERT", pca = "PCA", loggpt = "LogGPT")
runs <- read.csv(file.path(res_dir, "runs.csv"), stringsAsFactors = FALSE)
runs <- runs[runs$model %in% names(ranked), ]

for (metric in c("f1", "auc")) {
  wide <- reshape(runs[, c("seed", "model", metric)], idvar = "seed", timevar = "model", direction = "wide")
  wide <- wide[order(wide$seed), ]
  seeds <- wide$seed
  x <- data.frame(row.names = seeds, check.names = FALSE)
  for (m in names(ranked)) x[[ranked[[m]]]] <- wide[[paste0(metric, ".", m)]]
  stopifnot(!anyNA(x), nrow(x) >= 2)
  write.csv(cbind(seed = seeds, x), file.path(res_dir, sprintf("sk_input_%s.csv", metric)), row.names = FALSE)

  sk <- sk_esd(x, version = "np", alpha = 0.05)
  groups <- sk$groups
  out <- data.frame(model = names(groups), group = as.integer(groups),
                    median = sapply(names(groups), function(m) median(x[[m]])),
                    mean = sapply(names(groups), function(m) mean(x[[m]])),
                    n_resamples = nrow(x), row.names = NULL)
  write.csv(out, file.path(res_dir, sprintf("sk_%s.csv", metric)), row.names = FALSE)

  eff <- checkDifference(sk$groups, x)
  # Cliff's delta: the effect size the non-parametric version uses to merge groups
  eff$cliff_delta <- NA_real_
  eff$cliff_magnitude <- NA_character_
  for (r in seq_len(nrow(eff))) {
    cd <- cliff.delta(x[[eff$i[r]]], x[[eff$j[r]]])
    eff$cliff_delta[r] <- round(cd$estimate, 3)
    eff$cliff_magnitude[r] <- as.character(cd$magnitude)
  }
  names(eff)[names(eff) == "mag"] <- "cohen_d_magnitude"
  names(eff)[names(eff) == "est"] <- "cohen_d_abs"
  write.csv(eff, file.path(res_dir, sprintf("sk_%s_effect_sizes.csv", metric)), row.names = FALSE)

  png(file.path(fig_dir, sprintf("sk_%s.png", metric)), width = 1400, height = 1000, res = 200)
  plot(sk, ylab = toupper(metric), title = sprintf("Scott-Knott ESD (np): %s over %d resamples", toupper(metric), nrow(x)))
  dev.off()

  cat(sprintf("== %s\n", toupper(metric)))
  print(out)
}

writeLines(c(R.version.string, paste("ScottKnottESD", packageVersion("ScottKnottESD")),
             paste("effsize", packageVersion("effsize"))), file.path(res_dir, "sk_session_info.txt"))
