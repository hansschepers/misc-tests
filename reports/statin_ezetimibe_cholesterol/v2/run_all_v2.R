# =============================================================================
# run_all_v2.R
# Meal pattern, meal composition and pill timing in the v2 cholesterol /
# bile-acid / rosuvastatin / ezetimibe model.
#
# Usage: setwd("<repo>/reports/statin_ezetimibe_cholesterol/v2"); source("run_all_v2.R")
#        res <- run_all_v2(); res$pList$p1
# Run time: ~35 unique scenarios x 40 simulated days with a 12-min rk4 step;
# expect several minutes. Lower n_days (>= 25) or raise step_d (<= 1/96) to speed up.
# =============================================================================

library(data.table)
library(deSolve)
library(ggplot2)

model_dir <- "."
source(file.path(model_dir, "R", "01_parameters_v2.R"))
source(file.path(model_dir, "R", "02_model_v2.R"))
source(file.path(model_dir, "R", "03_scenarios_v2.R"))
source(file.path(model_dir, "R", "04_plots_v2.R"))

run_all_v2 <- function(n_days = 40, burn_in_d = 60, step_d = 1 / 120) {
  pList   <- list()
  results <- list()

  # ---- parameters: reference patient, mean-field calibration, PD calibration ----
  parms <- make_parms_v2()
  parms <- derive_parms_v2(parms)
  parms <- calibrate_pd_v2(parms)
  results$parms <- parms

  # ---- burn-in: reference diet (3 meals), no drug, to periodic steady state ----
  init0   <- make_init_v2(parms)
  burn_in <- simulate_v2(parms, init = init0, n_days = burn_in_d, step_d = step_d)
  init    <- end_state(out_dt = burn_in)
  results$burn_in_summary <- summarise_periodic(out_dt = burn_in, n_last_days = 3)
  # compare with calibration targets: LDL0 = 3 mmol/L, f_abs0 = 0.44, B0 = 2.6, BA0 = 1.2

  # ---- all experiments ----
  scenarios <- make_scenarios_v2()
  sim <- run_scenarios(parms, scenarios = scenarios, init = init, n_days = n_days,
                       step_d = step_d)
  results$summary <- sim$summary
  results$traces  <- sim$traces

  # ---- plots ----
  p1 <- plot_pattern_effects(summary_dt = sim$summary)
  p2 <- plot_pattern_mechanism(summary_dt = sim$summary)
  p3 <- plot_pattern_traces(summary_dt = sim$summary, traces_dt = sim$traces, treatment_sel = "eze")
  p4 <- plot_pill_timing(summary_dt = sim$summary)
  p5 <- plot_composition(summary_dt = sim$summary)
  p6 <- plot_addons(summary_dt = sim$summary)
  p7 <- plot_lithogenicity(summary_dt = sim$summary, traces_dt = sim$traces)
  pList$p1 <- p1
  pList$p2 <- p2
  pList$p3 <- p3
  pList$p4 <- p4
  pList$p5 <- p5
  pList$p6 <- p6
  pList$p7 <- p7

  list(pList = pList, results = results)
}

res <- run_all_v2()
# res$results$burn_in_summary   # how close the pulsed reference is to the mean-field targets
# res$results$summary[experiment == "B_pill_timing"]
# plot_name <- names(res$pList)[1]
# for (plot_name in names(res$pList)) {
#   ggsave(filename = file.path(model_dir, paste0(plot_name, ".png")), plot = res$pList[[plot_name]],
#          width = 10, height = 7, dpi = 150)
# }
