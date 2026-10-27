# =============================================================================
# run_demo.R  --  reproduce every dualctl demo figure from one seed
#
# Usage (from the dualctl/ directory, after installing the package so that
# parallel workers can load it):
#   R CMD INSTALL .
#   Rscript run_demo.R [seed] [workers] [config]
# e.g.
#   Rscript run_demo.R 20261027 8 fouling_default.yaml   # full: 50 seeds + ablations
#   Rscript run_demo.R 1 1 fouling_quick.yaml            # quick smoke run
#
# Output: demo_output/ with PNG figures, CSV tables, summary.md and results.rds.
# Runtime of the full run has NOT been measured (the code was written without
# being executed); expect hours on one core, use several workers.
# =============================================================================

args    <- commandArgs(trailingOnly = TRUE)
seed    <- if (length(args) >= 1) as.integer(args[1]) else 20261027L
workers <- if (length(args) >= 2) as.integer(args[2]) else max(1L, parallel::detectCores() - 1L)
config  <- if (length(args) >= 3) args[3] else "fouling_default.yaml"

if (requireNamespace("dualctl", quietly = TRUE)) {
  library(dualctl)
} else {
  # development fallback; parallel workers will not see an unloaded package
  pkgload::load_all(".")
  workers <- 1L
}

use_parallel <- workers > 1 && requireNamespace("future.apply", quietly = TRUE)
if (use_parallel) future::plan(future::multisession, workers = workers)

pList <- list()                                   # all figures end up here
out <- run_demo(config = config, seed = seed, out_dir = "demo_output", parallel = use_parallel)
pList <- out$pList
results <- out$results

if (use_parallel) future::plan(future::sequential)
cat("figures:", paste(names(pList), collapse = ", "), "\n")
cat("summary: demo_output/summary.md\n")
