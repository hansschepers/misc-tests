# =============================================================================
# run_all.R
# Ezetimibe LDL-lowering as a function of system state and patient parameters
# in a coupled cholesterol / rosuvastatin / ezetimibe ODE model.
#
# Usage: setwd("<repo>/reports/statin_ezetimibe_cholesterol"); source("run_all.R")
#        res <- run_all(); res$pList$p1
# Run time: steady-state scans take seconds; dynamic runs ~ tens of seconds each.
# =============================================================================

library(data.table)
library(deSolve)
library(ggplot2)

model_dir <- "."
source(file.path(model_dir, "R", "01_parameters.R"))
source(file.path(model_dir, "R", "02_model.R"))
source(file.path(model_dir, "R", "03_steady_state.R"))
source(file.path(model_dir, "R", "04_plots.R"))

run_all <- function(run_dynamic = TRUE) {
  pList   <- list()
  results <- list()

  # ---- parameters: reference patient, calibrated PD ----
  parms <- make_parms()
  parms <- derive_parms(parms)
  parms <- calibrate_pd(parms)
  results$parms <- parms

  # ---- 1. ezetimibe alone: HMGCR capacity x dietary cholesterol ----
  grid1 <- make_scan_grid(S_max_factor = c(0.5, 0.75, 1, 1.5, 2),
                          D_mg = c(0, 150, 300, 600, 900),
                          dose_rsv_mg = 10, dose_eze_mg = 10)
  res1 <- scan_grid(parms, grid = grid1)
  results$synth_x_diet <- res1
  p1 <- plot_heatmap(dt = res1, x_col = "D_mg", y_col = "S_max_factor", fill_col = "pct_eze_mono",
                     x_lab = "dietary cholesterol [mg/d]", y_lab = "hepatic HMGCR capacity factor [-]",
                     fill_lab = "LDL-C change [%]",
                     title = "Ezetimibe 10 mg alone: LDL-C change at steady state")
  p2 <- plot_heatmap(dt = res1, x_col = "D_mg", y_col = "S_max_factor", fill_col = "pct_eze_addon",
                     x_lab = "dietary cholesterol [mg/d]", y_lab = "hepatic HMGCR capacity factor [-]",
                     fill_lab = "LDL-C change [%]",
                     title = "Ezetimibe 10 mg added to rosuvastatin 10 mg")
  pList$p1 <- p1
  pList$p2 <- p2

  # ---- 2. absorber x synthesizer phenotype ----
  grid2 <- make_scan_grid(S_max_factor = c(0.5, 0.75, 1, 1.5, 2),
                          k_abs_factor = c(0.4, 0.7, 1, 1.4, 2),
                          dose_rsv_mg = 10, dose_eze_mg = 10)
  res2 <- scan_grid(parms, grid = grid2)
  results$synth_x_absorber <- res2
  p3 <- plot_heatmap(dt = res2, x_col = "k_abs_factor", y_col = "S_max_factor", fill_col = "pct_eze_mono",
                     x_lab = "NPC1L1 absorption capacity factor [-]",
                     y_lab = "hepatic HMGCR capacity factor [-]", fill_lab = "LDL-C change [%]",
                     title = "Ezetimibe alone: absorber vs synthesizer phenotype")
  pList$p3 <- p3

  # ---- 3. statin background ----
  grid3 <- make_scan_grid(S_max_factor = c(0.5, 1, 2),
                          dose_rsv_mg = c(0, 5, 10, 20, 40), dose_eze_mg = 10)
  res3 <- scan_grid(parms, grid = grid3)
  results$statin_background <- res3
  p4 <- plot_addon_vs_rsv(dt = res3)
  pList$p4 <- p4

  # ---- 4. one-at-a-time sensitivity of the ezetimibe effect ----
  grid4 <- make_tornado_grid()
  res4 <- scan_grid(parms, grid = grid4)
  results$tornado <- res4
  p5 <- plot_tornado(dt = res4)
  pList$p5 <- p5

  # ---- 5. who compensates the lost absorption? ----
  prof_ref <- pk_unit_profile(parms)
  I_E_ref  <- mean_inhibition(conc_per_mg = prof_ref$E_gut_per_mg, dose_mg = parms$dose_ref_eze,
                              Imax = parms$Emax_E, IC50 = parms$EC50_E)
  comp <- compensation_table(parms, S_max_factors = c(0.5, 1, 2), I_E = I_E_ref)
  results$compensation <- comp
  p6 <- plot_compensation(dt = comp)
  pList$p6 <- p6

  # ---- 6. dynamic simulations (three synthesizer phenotypes) ----
  if (run_dynamic) {
    start_rsv_d <- 14
    start_eze_d <- 56
    phenotypes  <- c(low_synthesizer = 0.5, reference = 1, high_synthesizer = 2)
    dyn_list    <- list()
    ph <- names(phenotypes)[1]
    for (ph in names(phenotypes)) {
      parms_ph <- derive_parms(modifyList(parms, list(S_max_factor = unname(phenotypes[ph]))))
      dyn_ph   <- simulate_dynamic(parms = parms_ph, dose_rsv_mg = 10, dose_eze_mg = 10,
                                   start_rsv_d = start_rsv_d, start_eze_d = start_eze_d, end_d = 98)
      dyn_ph[, patient := ph]
      dyn_list[[ph]] <- dyn_ph
    }
    dyn <- rbindlist(dyn_list)
    results$dynamic <- dyn
    p7 <- plot_ldl_timecourse(dt = dyn, start_rsv_d = start_rsv_d, start_eze_d = start_eze_d)
    p8 <- plot_daily_fluxes(dt = dyn[patient == "reference"])
    p9 <- plot_intraday(dt = dyn[patient == "reference"], n_last_days = 3)
    pList$p7 <- p7
    pList$p8 <- p8
    pList$p9 <- p9
  }

  list(pList = pList, results = results)
}

res <- run_all()
# res$pList$p1 ; res$results$tornado
# optional: save all plots
# plot_name <- names(res$pList)[1]
# for (plot_name in names(res$pList)) {
#   ggsave(filename = file.path(model_dir, paste0(plot_name, ".png")), plot = res$pList[[plot_name]],
#          width = 9, height = 6, dpi = 150)
# }
