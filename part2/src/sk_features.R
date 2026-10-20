#!/usr/bin/env Rscript
# Optional E3 step: Scott-Knott ESD (np) ranking of the correlation clusters by their per-resample
# mean |grouped SHAP| (the use case ScottKnottESD was designed for: ranking features).
#   Rscript src/sk_features.R results/explain
# Input : <dir>/sk_feature_importance_input.csv (rows = seeds, columns = clusters)
# Output: <dir>/sk_feature_importance.csv, <dir>/sk_feature_importance.png
suppressPackageStartupMessages(library(ScottKnottESD))
stopifnot(packageVersion("ScottKnottESD") >= "3.0.0")
d <- commandArgs(trailingOnly = TRUE)[1]
x <- read.csv(file.path(d, "sk_feature_importance_input.csv"), check.names = FALSE)
x$seed <- NULL
x <- x[, colSums(x) > 0, drop = FALSE]          # clusters with zero importance in every resample cannot be ranked
sk <- sk_esd(x, version = "np", alpha = 0.05)
out <- data.frame(group = names(sk$groups), sk_rank = as.integer(sk$groups),
                  median = sapply(names(sk$groups), function(m) median(x[[m]])), row.names = NULL)
write.csv(out, file.path(d, "sk_feature_importance.csv"), row.names = FALSE)
png(file.path(d, "sk_feature_importance.png"), width = 2000, height = 1000, res = 200)
plot(sk, ylab = "mean |grouped SHAP|", title = "Scott-Knott ESD (np) of cluster importance over resamples", id.las = 2)
dev.off()
