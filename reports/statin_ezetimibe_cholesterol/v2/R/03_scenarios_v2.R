# =============================================================================
# 03_scenarios_v2.R
# Meal schedules, scenario table, periodic-steady-state runs and summaries.
# Every scenario runs from the burned-in reference state to a periodic steady
# state (n_days), then the last n_last_days are averaged.
# =============================================================================

# meal clock and per-meal shares of cholesterol, fat, fibre, plant sterols
make_meal_schedule <- function(pattern = "three_meals", composition = "spread") {
  schedule <- switch(pattern,
    three_meals = list(meal_h = c(8, 13, 19),  share = c(0.25, 0.35, 0.40)),
    tre_16_8    = list(meal_h = c(12, 19),     share = c(0.45, 0.55)),       # eating window 12-20 h
    omad        = list(meal_h = 18,            share = 1),                   # one meal a day
    stop("unknown pattern: ", pattern))
  chol_frac <- schedule$share
  fat_frac  <- schedule$share
  if (composition != "spread") {
    if (pattern != "three_meals") stop("compositions are defined for three_meals only")
    fracs <- switch(composition,
      chol_breakfast_fatty    = list(chol = c(1, 0, 0), fat = c(0.25, 0.35, 0.40)),  # eggs + fat at breakfast
      chol_breakfast_low_fat  = list(chol = c(1, 0, 0), fat = c(0.05, 0.45, 0.50)),  # cholesterol without fat
      chol_dinner_fatty       = list(chol = c(0, 0, 1), fat = c(0.10, 0.20, 0.70)),  # one rich dinner
      stop("unknown composition: ", composition))
    chol_frac <- fracs$chol
    fat_frac  <- fracs$fat
  }
  list(meal_h    = schedule$meal_h,    # [h]
       chol_frac = chol_frac,          # [-]
       fat_frac  = fat_frac,           # [-]
       fib_frac  = schedule$share,     # [-]
       ps_frac   = schedule$share)     # [-]
}

TREATMENTS <- data.table(treatment   = c("none", "eze", "rsv", "rsv_eze"),
                         dose_rsv_mg = c(0, 0, 10, 10),        # [mg/d]
                         dose_eze_mg = c(0, 10, 0, 10))        # [mg/d]

# all experiments in one table; duplicates are simulated once (see run_scenarios)
make_scenarios_v2 <- function() {
  # A: meal pattern at equal daily intake
  exp_A <- CJ(pattern = c("three_meals", "tre_16_8", "omad"), treatment = TREATMENTS$treatment)
  exp_A[, `:=`(experiment = "A_meal_pattern", composition = "spread", dose_hour = 8,
               occupancy_kinetic = TRUE, fib_daily_g = 0, ps_daily_g = 0)]

  # B: pill timing x NPC1L1 binding kinetics x meal pattern
  exp_B <- CJ(pattern = c("three_meals", "tre_16_8"), dose_hour = c(8, 14, 22),
              occupancy_kinetic = c(TRUE, FALSE), treatment = "eze")
  exp_B[, `:=`(experiment = "B_pill_timing", composition = "spread",
               fib_daily_g = 0, ps_daily_g = 0)]
  exp_B0 <- CJ(pattern = c("three_meals", "tre_16_8"), treatment = "none")
  exp_B0[, `:=`(experiment = "B_pill_timing", composition = "spread", dose_hour = 8,
                occupancy_kinetic = TRUE, fib_daily_g = 0, ps_daily_g = 0)]

  # C: where the cholesterol and fat are eaten (three meals)
  exp_C <- CJ(composition = c("spread", "chol_breakfast_fatty", "chol_breakfast_low_fat",
                              "chol_dinner_fatty"),
              treatment = c("none", "eze"))
  exp_C[, `:=`(experiment = "C_composition", pattern = "three_meals", dose_hour = 8,
               occupancy_kinetic = TRUE, fib_daily_g = 0, ps_daily_g = 0)]

  # D: add-ons: viscous fibre 10 g/d, plant sterols 2 g/d
  exp_D <- CJ(fib_daily_g = c(0, 10), ps_daily_g = c(0, 2), treatment = c("none", "eze"))
  exp_D[, `:=`(experiment = "D_addons", pattern = "three_meals", composition = "spread",
               dose_hour = 8, occupancy_kinetic = TRUE)]

  sc <- rbindlist(list(exp_A, exp_B, exp_B0, exp_C, exp_D), use.names = TRUE)
  sc <- merge(sc, TREATMENTS, by = "treatment")
  # a drug-free run does not depend on pill time or binding mode
  sc[treatment == "none", `:=`(dose_hour = 8, occupancy_kinetic = TRUE)]
  sc[, run_key := paste(pattern, composition, dose_hour, occupancy_kinetic, fib_daily_g,
                        ps_daily_g, dose_rsv_mg, dose_eze_mg, sep = "|")]
  sc[, diet_key := paste(pattern, composition, fib_daily_g, ps_daily_g, sep = "|")]
  setcolorder(sc, c("experiment", "treatment", "pattern", "composition"))
  sc[]
}

# parameters for one scenario row
scenario_parms <- function(parms, row) {
  schedule <- make_meal_schedule(pattern = row$pattern, composition = row$composition)
  modifyList(parms, c(schedule,
                      list(dose_hour         = row$dose_hour,           # [h]
                           occupancy_kinetic = row$occupancy_kinetic,   # [-]
                           fib_daily_g       = row$fib_daily_g,         # [g/d]
                           ps_daily_g        = row$ps_daily_g,          # [g/d]
                           dose_rsv_mg       = row$dose_rsv_mg,         # [mg/d]
                           dose_eze_mg       = row$dose_eze_mg,         # [mg/d]
                           start_rsv_d       = 0,                       # [d]
                           start_eze_d       = 0)))                     # [d]
}

# daily means over the last n_last_days of a periodic steady state
summarise_periodic <- function(out_dt, n_last_days = 3) {
  end_d <- max(out_dt$time)
  win   <- out_dt[time > end_d - n_last_days]
  win[, hour_of_day := (time %% 1) * 24]
  win[, .(LDL          = mean(LDL),                          # [mmol/L]
          LDL_mgdL     = mean(LDL_mgdL),                     # [mg/dL]
          C_L          = mean(C_L),                          # [mmol]
          D_in         = mean(D_in),                         # [mmol/d]
          B            = mean(B),                            # biliary cholesterol [mmol/d]
          B_night      = mean(B[hour_of_day >= 2 & hour_of_day < 6]),  # [mmol/d]
          A            = mean(A),                            # absorption [mmol/d]
          f_abs        = mean(A) / (mean(D_in) + mean(B)),   # [-]
          S_hep        = mean(S_hep),                        # [mmol/d]
          J_sec        = mean(J_sec),                        # hepatic BA secretion [mmol/d]
          J_syn        = mean(J_syn),                        # BA synthesis (CYP7A1) [mmol/d]
          J_colB       = mean(J_colB),                       # faecal BA [mmol/d]
          BA_pool      = mean(BA_pool),                      # [mmol]
          BA_cycles    = mean(J_sec) / mean(BA_pool),        # enterohepatic cycles [1/d]
          litho_mean   = mean(litho_GB),                     # cholesterol/BA in gallbladder [mol/mol]
          litho_max    = max(litho_GB),                      # [mol/mol]
          I_E          = mean(I_E),                          # [-]
          I_S          = mean(I_S))]                         # [-]
}

# runs every unique scenario; returns per-scenario summary + within-day traces
run_scenarios <- function(parms, scenarios, init, n_days = 40, step_d = 1 / 120,
                          n_last_days = 3, trace_days = 2) {
  runs <- unique(scenarios, by = "run_key")
  summary_list <- vector("list", nrow(runs))
  trace_list   <- vector("list", nrow(runs))

  i <- 1
  for (i in seq_len(nrow(runs))) {
    row     <- runs[i]
    parms_i <- scenario_parms(parms, row = row)
    out_i   <- simulate_v2(parms = parms_i, init = init, n_days = n_days, step_d = step_d)
    summary_list[[i]] <- cbind(data.table(run_key = row$run_key),
                               summarise_periodic(out_dt = out_i, n_last_days = n_last_days))
    trace_i <- out_i[time > n_days - trace_days]
    trace_i[, `:=`(run_key = row$run_key, time = time - (n_days - trace_days))]  # re-based [d]
    trace_list[[i]] <- trace_i
    message(sprintf("run %d/%d done: %s", i, nrow(runs), row$run_key))
  }
  run_summary <- rbindlist(summary_list)
  res <- merge(scenarios, run_summary, by = "run_key")

  # LDL change vs the drug-free run on the same diet
  base <- res[treatment == "none", .(diet_key, LDL_base = LDL, f_abs_base = f_abs)]
  base <- unique(base, by = "diet_key")
  res  <- merge(res, base, by = "diet_key")
  res[, pct_LDL := 100 * (LDL / LDL_base - 1)]                # [%]

  list(summary = res, traces = rbindlist(trace_list))
}
