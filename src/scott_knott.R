# Scott-Knott ESD ranking (Tantithamthavorn et al., TSE 2017), ScottKnottESD R package.
# Usage: Rscript src/scott_knott.R <wide.csv> <out.csv>
#   wide.csv: one column per model, one row per resample (seed); higher = better.
# Uses the non-parametric version (version="np", default of ScottKnottESD >= 2.0): it ranks on
# rank-transformed data and merges groups whose difference has a negligible Cliff's delta.
lib <- Sys.getenv("R_LIBS_USER"); .libPaths(c(lib, .libPaths()))
suppressMessages(library(ScottKnottESD))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1], check.names = FALSE)
sk <- sk_esd(d, version = "np")
# sk_esd mangles column names with make.names(); map them back to the original model names
orig <- names(d)
out <- data.frame(model = orig[match(names(sk$groups), make.names(orig))], rank = as.integer(sk$groups))
write.csv(out, args[2], row.names = FALSE)
print(out)
