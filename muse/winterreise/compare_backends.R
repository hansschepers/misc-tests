# Compare embedding backends for one cycle: reads out/<cycle>/*/vectors.rds
# Usage: Rscript compare_backends.R [cycle]
setwd("C:/Users/Lenovo/Documents/misc-tests/muse/winterreise")
args <- commandArgs(trailingOnly = TRUE)
CYCLE <- if (length(args) >= 1) args[1] else "winterreise"
suppressPackageStartupMessages({ library(data.table); library(glue) })

runs <- list.files(file.path("out", CYCLE), pattern = "^vectors\\.rds$", recursive = TRUE, full.names = TRUE)
R <- lapply(runs, readRDS)
names(R) <- basename(dirname(runs))
sink(file.path("out", CYCLE, "backend_comparison.txt"), split = TRUE)
cat(glue("=== {CYCLE}: {length(R)} runs: {paste(names(R), collapse = ', ')} ==="), "\n")
if (length(R) < 2) { sink(); stop("need >= 2 runs") }

upper <- function(M) M[upper.tri(M)]
pairs <- combn(names(R), 2, simplify = FALSE)
for (pr in pairs) {
  a <- R[[pr[1]]]
  b <- R[[pr[2]]]
  stopifnot(identical(a$stanzas$text, b$stanzas$text))
  cat(glue("\n##### {pr[1]}  vs  {pr[2]}"), "\n")
  # 1. RSA: do they agree on which songs / stanzas are alike? (Spearman, upper triangles)
  rs_song <- cor(upper(a$V %*% t(a$V)), upper(b$V %*% t(b$V)), method = "spearman")
  rs_st <- cor(upper(a$E %*% t(a$E)), upper(b$E %*% t(b$E)), method = "spearman")
  cat(glue("RSA songs  (n = {nrow(a$V)}):   rho = {round(rs_song, 3)}"), "\n")
  cat(glue("RSA stanzas (n = {nrow(a$E)}):  rho = {round(rs_st, 3)}"), "\n")
  # 2. axis correspondence: |cor| of PC song scores (rows = first run, cols = second)
  C <- cor(a$pc$song, b$pc$song)
  dimnames(C) <- list(glue("{pr[1]}:PC{1:3}"), glue("PC{1:3}"))
  cat("PC correspondence (signed cor of song scores; |r| ~ 1 = same axis, off-diagonal = swapped):\n")
  print(round(C, 2))
  cat(glue("variance explained: {paste(round(100 * a$pc$var_explained, 1), collapse = '/')} %  vs  ",
           "{paste(round(100 * b$pc$var_explained, 1), collapse = '/')} %"), "\n")
  # 3. step profile: do both see the same big jumps?
  cat(glue("step-distance profile cor (consecutive songs): ",
           "{round(cor(a$songs$step_dist, b$songs$step_dist, use = 'complete.obs'), 3)}"), "\n")
}
cat("\nPath tests:\n")
print(rbindlist(lapply(names(R), function(n) R[[n]]$path_test[, run := n]))[
  , .(run, order, path_length = round(path_length, 3), z = round(z, 2), p_shorter = round(p_shorter, 3))])
sink()
