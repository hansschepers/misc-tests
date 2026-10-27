# =============================================================================
# 03_steady_state.R
# Semi-analytical steady state and scenario scans.
#
# At steady state all LDL produced is cleared, and everything taken up by the
# periphery returns by RCT, so the LDL and RCT routes cancel in the liver
# balance.  The whole system then reduces to ONE scalar equation in C_L:
#
#   f_abs * (D + k_B C_L) + S_max (1 - I_S) g(C_L) + S_P = (k_B + k_BA) C_L
#   (absorbed)              (hepatic synthesis)   (periph.)  (bile + bile acids)
#
# with f_abs = k_abs (1 - I_E) / (k_abs (1 - I_E) + k_fec).
# LDL is then a READ-OUT of C_L:
#   LDL = P_V0 (C_L/C_L0)^q_V / (V_p (k_U (g/g0)^m_R + k_nR))
#
# Drug effects enter as time-averaged inhibitions over a dosing interval
# (cholesterol pools respond over days; PK fluctuates within hours).
# =============================================================================

solve_steady_state <- function(parms, I_S = 0, I_E = 0) {
  list2env(parms, envir = environment())

  k_abs_eff <- k_abs * (1 - I_E)                               # [1/d]
  f_abs     <- k_abs_eff / (k_abs_eff + k_fec)                 # fractional absorption [-]
  g_fun     <- function(C) 1 / (1 + (C / K_C)^n_g)             # SREBP2 activity [-]
  balance   <- function(C) {
    f_abs * (D_daily + k_B * C) + S_max * (1 - I_S) * g_fun(C) + S_P - (k_B + k_BA) * C  # [mmol/d]
  }
  C_L <- uniroot(balance, lower = 1e-6, upper = 100 * C_L0, tol = 1e-10)$root  # [mmol]

  g      <- g_fun(C_L)                                         # [-]
  R_LDLR <- (g / g0)^m_R                                       # [-]
  P_V    <- P_V0 * (C_L / C_L0)^q_V                            # [mmol/d]
  LDL    <- P_V / (V_p * (k_U * R_LDLR + k_nR))                # [mmol/L]
  C_P    <- (S_P + k_nR * LDL * V_p) / k_RCT                   # [mmol]
  B      <- k_B * C_L                                          # [mmol/d]
  C_lum  <- (D_daily + B) / (k_abs_eff + k_fec)                # [mmol]
  A      <- k_abs_eff * C_lum                                  # [mmol/d]
  S_hep  <- S_max * (1 - I_S) * g                              # [mmol/d]
  BA     <- k_BA * C_L                                         # [mmol/d]

  # loop-gain diagnostics (linearisation around this steady state)
  phi_R    <- k_U * R_LDLR / (k_U * R_LDLR + k_nR)             # share of LDL cleared by LDLR [-]
  eps_LDL  <- q_V + phi_R * m_R * n_g * (1 - g)                # elasticity dlnLDL/dlnC_L [-]
  gain_S   <- S_hep * n_g * (1 - g)                            # -C_L dS/dC_L, synthesis buffer [mmol/d]
  gain_out <- (k_B * (1 - f_abs) + k_BA) * C_L                 # C_L d(out - reabsorbed)/dC_L [mmol/d]

  data.table(C_L, g, R_LDLR, LDL, LDL_mgdL = LDL * MW_chol / 10, C_P, C_lum,
             A, S_hep, B, BA, P_V, f_abs, I_S, I_E,
             phi_R, eps_LDL, gain_S, gain_out)
}

# four steady states for one patient/diet/dose combination
evaluate_scenario <- function(parms, dose_rsv_mg, dose_eze_mg, prof_rsv, prof_eze) {
  I_S <- mean_inhibition(conc_per_mg = prof_rsv$C_h_per_mg, dose_mg = dose_rsv_mg,
                         Imax = parms$Imax_S, IC50 = parms$IC50_S)   # [-]
  I_E <- mean_inhibition(conc_per_mg = prof_eze$E_gut_per_mg, dose_mg = dose_eze_mg,
                         Imax = parms$Emax_E, IC50 = parms$EC50_E)   # [-]

  ss_none <- solve_steady_state(parms, I_S = 0,   I_E = 0)
  ss_rsv  <- solve_steady_state(parms, I_S = I_S, I_E = 0)
  ss_eze  <- solve_steady_state(parms, I_S = 0,   I_E = I_E)
  ss_both <- solve_steady_state(parms, I_S = I_S, I_E = I_E)

  data.table(
    I_S            = I_S,                                        # [-]
    I_E            = I_E,                                        # [-]
    LDL_base       = ss_none$LDL,                                # [mmol/L]
    f_abs_base     = ss_none$f_abs,                              # [-]
    S_hep_base     = ss_none$S_hep,                              # [mmol/d]
    eps_LDL_base   = ss_none$eps_LDL,                            # [-]
    pct_rsv        = 100 * (ss_rsv$LDL  / ss_none$LDL - 1),      # [%]
    pct_eze_mono   = 100 * (ss_eze$LDL  / ss_none$LDL - 1),      # [%]
    pct_eze_addon  = 100 * (ss_both$LDL / ss_rsv$LDL  - 1),      # [%] relative to statin alone
    pct_both       = 100 * (ss_both$LDL / ss_none$LDL - 1),      # [%]
    pct_dS_eze     = 100 * (ss_eze$S_hep / ss_none$S_hep - 1)    # synthesis rise on ezetimibe [%]
  )
}

# scenario grid: every column is needed by scan_grid()
make_scan_grid <- function(S_max_factor = 1, K_C_factor = 1, k_abs_factor = 1, k_B_factor = 1,
                           f_OATP = 1, m_R = 1, q_V = 0.5, D_mg = 300,
                           dose_rsv_mg = 10, dose_eze_mg = 10) {
  CJ(S_max_factor = S_max_factor,   # [-]
     K_C_factor   = K_C_factor,     # [-]
     k_abs_factor = k_abs_factor,   # [-]
     k_B_factor   = k_B_factor,     # [-]
     f_OATP       = f_OATP,         # [-]
     m_R          = m_R,            # [-]
     q_V          = q_V,            # [-]
     D_mg         = D_mg,           # dietary cholesterol [mg/d]
     dose_rsv_mg  = dose_rsv_mg,    # [mg/d]
     dose_eze_mg  = dose_eze_mg)    # [mg/d]
}

SCAN_COLS <- c("S_max_factor", "K_C_factor", "k_abs_factor", "k_B_factor", "f_OATP",
               "m_R", "q_V", "D_mg", "dose_rsv_mg", "dose_eze_mg")

# parms must be calibrated (calibrate_pd) before scanning
scan_grid <- function(parms, grid) {
  missing_cols <- setdiff(SCAN_COLS, names(grid))
  if (length(missing_cols) > 0) stop("grid lacks columns: ", paste(missing_cols, collapse = ", "))
  if (is.na(parms$IC50_S) || is.na(parms$EC50_E)) stop("run calibrate_pd() first")

  prof_eze       <- pk_unit_profile(parms)   # ezetimibe PK does not depend on scanned factors
  prof_rsv_cache <- list()                   # rosuvastatin PK depends on f_OATP only
  res_list       <- vector("list", nrow(grid))

  i <- 1
  for (i in seq_len(nrow(grid))) {
    row <- grid[i]
    parms_i <- modifyList(parms, list(S_max_factor = row$S_max_factor,
                                      K_C_factor   = row$K_C_factor,
                                      k_abs_factor = row$k_abs_factor,
                                      k_B_factor   = row$k_B_factor,
                                      f_OATP       = row$f_OATP,
                                      m_R          = row$m_R,
                                      q_V          = row$q_V,
                                      D_daily      = row$D_mg / parms$MW_chol))  # [mmol/d]
    parms_i <- derive_parms(parms_i)
    key <- as.character(row$f_OATP)
    if (is.null(prof_rsv_cache[[key]])) prof_rsv_cache[[key]] <- pk_unit_profile(parms = parms_i)
    res_list[[i]] <- cbind(row, evaluate_scenario(parms = parms_i,
                                                  dose_rsv_mg = row$dose_rsv_mg,
                                                  dose_eze_mg = row$dose_eze_mg,
                                                  prof_rsv = prof_rsv_cache[[key]],
                                                  prof_eze = prof_eze))
  }
  rbindlist(res_list)
}

# one-at-a-time sensitivity of the ezetimibe effect to patient parameters
make_tornado_grid <- function() {
  specs <- data.table(
    parameter = c("S_max_factor", "K_C_factor", "k_B_factor", "m_R", "q_V",
                  "f_OATP", "k_abs_factor", "D_mg"),
    label     = c("HMGCR capacity (liver)", "SREBP2 set point (liver)",
                  "ABCG5/G8 biliary secretion (liver)", "LDLR elasticity, PCSK9 (liver)",
                  "VLDL production elasticity (liver)", "OATP1B1 rosuvastatin uptake (liver)",
                  "NPC1L1 absorption capacity (gut)", "dietary cholesterol (w)"),
    low       = c(0.5, 0.8, 0.5, 0.5, 0.0, 0.5, 0.5, 150),
    high      = c(2.0, 1.25, 2.0, 1.5, 1.0, 1.5, 1.5, 600)
  )
  grid_list <- list(cbind(make_scan_grid(), parameter = "reference", label = "reference", level = "reference"))
  j <- 1
  for (j in seq_len(nrow(specs))) {
    lev <- "low"
    for (lev in c("low", "high")) {
      args <- list()
      args[[specs$parameter[j]]] <- specs[[lev]][j]
      grid_j <- do.call(make_scan_grid, args)
      grid_list[[length(grid_list) + 1]] <- cbind(grid_j, parameter = specs$parameter[j],
                                                  label = specs$label[j], level = lev)
    }
  }
  rbindlist(grid_list)
}

# steady-state flux changes caused by ezetimibe: shows who compensates the
# lost absorption (balance: dA + dS_hep = dB + dBA)
compensation_table <- function(parms, S_max_factors, I_E) {
  res_list <- list()
  k <- 1
  for (k in seq_along(S_max_factors)) {
    parms_k <- derive_parms(modifyList(parms, list(S_max_factor = S_max_factors[k])))
    ss0 <- solve_steady_state(parms = parms_k, I_S = 0, I_E = 0)
    ss1 <- solve_steady_state(parms = parms_k, I_S = 0, I_E = I_E)
    res_list[[k]] <- data.table(
      S_max_factor = S_max_factors[k],
      flux  = c("absorption A", "hepatic synthesis S_hep", "biliary cholesterol B", "bile acids BA"),
      delta = c(ss1$A - ss0$A, ss1$S_hep - ss0$S_hep, ss1$B - ss0$B, ss1$BA - ss0$BA),  # [mmol/d]
      pct_LDL = 100 * (ss1$LDL / ss0$LDL - 1)                                            # [%]
    )
  }
  rbindlist(res_list)
}
