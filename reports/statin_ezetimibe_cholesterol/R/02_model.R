# =============================================================================
# 02_model.R
# Drivers w(t), dosing u(t), PK, coupled ODE and dynamic simulation.
# Units: [mmol], [mg], [L], [d] (see 01_parameters.R)
# =============================================================================

# ---------------------------- drivers w(t) -----------------------------------

# dietary cholesterol intake as smooth Gaussian meal pulses [mmol/d];
# integral over one day equals D_daily
meal_intake_rate <- function(time, D_daily, meal_h, meal_frac, meal_sd_h) {
  hour_of_day <- (time %% 1) * 24                              # [h]
  delta_h     <- ((hour_of_day - meal_h + 12) %% 24) - 12       # signed distance to each meal [h]
  pulse       <- meal_frac * stats::dnorm(delta_h, sd = meal_sd_h) * 24  # [1/d]
  D_daily * sum(pulse)                                         # [mmol/d]
}

# gallbladder emptying rate: basal + meal-induced (CCK) pulses [1/d]
gallbladder_emptying_rate <- function(time, k_gb_basal, k_gb_peak, meal_h, gb_sd_h) {
  hour_of_day <- (time %% 1) * 24                              # [h]
  delta_h     <- ((hour_of_day - meal_h + 12) %% 24) - 12       # [h]
  k_gb_basal + k_gb_peak * sum(exp(-0.5 * (delta_h / gb_sd_h)^2))  # [1/d]
}

# ---------------------------- control u(t) -----------------------------------

# daily oral dose delivered as a smooth Gaussian pulse (tablet dissolution and
# gastric emptying, sd = dose_sd_h) around dose_hour, from day start_d on [mg/d];
# integral over one day equals dose_mg.  Smooth input keeps fixed-step rk4 exact
# enough without deSolve events.
dosing_rate <- function(time, dose_mg, start_d, dose_hour, dose_sd_h) {
  if (dose_mg <= 0) return(0)
  hour_of_day <- (time %% 1) * 24                              # [h]
  delta_h     <- ((hour_of_day - dose_hour + 12) %% 24) - 12    # distance to nearest dose [h]
  time_dose   <- time - delta_h / 24                           # time of that dose [d]
  if (time_dose < start_d - 1e-9) return(0)
  dose_mg * stats::dnorm(delta_h, sd = dose_sd_h) * 24         # [mg/d]
}

# ---------------------------- pharmacokinetics -------------------------------

# derivatives of the 7 PK states (order = PK_STATES); shared by full and PK-only ODE
pk_derivs <- function(time, state, parms) {
  list2env(c(parms, as.list(state)), envir = environment())

  # control u(t): oral intake rates [mg/d]
  u_rsv <- dosing_rate(time, dose_mg = dose_rsv_mg, start_d = start_rsv_d,
                       dose_hour = dose_hour, dose_sd_h = dose_sd_h)  # [mg/d]
  u_eze <- dosing_rate(time, dose_mg = dose_eze_mg, start_d = start_eze_d,
                       dose_hour = dose_hour, dose_sd_h = dose_sd_h)  # [mg/d]

  # rosuvastatin [mg]
  C_c_rsv   <- A_rc / V_c_rsv                                  # plasma conc. [mg/L]
  C_per_rsv <- A_rp / V_per_rsv                                # peripheral conc. [mg/L]
  upt_rsv   <- CL_upt_rsv * f_OATP * C_c_rsv                   # OATP1B1 hepatic uptake [mg/d]
  dA_rg <- F_rsv * u_rsv - ka_rsv * A_rg                      # [mg/d]
  dA_rc <- ka_rsv * A_rg - upt_rsv + k_hc_rsv * A_rh -
           CL_ren_rsv * C_c_rsv - Q_rsv * (C_c_rsv - C_per_rsv) # [mg/d]
  dA_rp <- Q_rsv * (C_c_rsv - C_per_rsv)                       # [mg/d]
  dA_rh <- upt_rsv - (k_hc_rsv + k_bil_rsv) * A_rh             # [mg/d]

  # ezetimibe with enterohepatic circulation [mg]
  k_gb <- gallbladder_emptying_rate(time, k_gb_basal = k_gb_basal, k_gb_peak = k_gb_peak,
                                    meal_h = meal_h, gb_sd_h = gb_sd_h)  # [1/d]
  dE_gut <- u_eze + k_gb * E_gb - (ka_eze + k_fec_eze) * E_gut # [mg/d]
  dE_pl  <- ka_eze * E_gut - (k_bile_eze + k_ren_eze) * E_pl   # [mg/d]
  dE_gb  <- k_bile_eze * E_pl - k_gb * E_gb                    # [mg/d]

  list(derivs  = c(dA_rg, dA_rc, dA_rp, dA_rh, dE_gut, dE_pl, dE_gb),
       C_c_rsv = C_c_rsv,                                      # [mg/L]
       C_h_rsv = A_rh / V_h,                                   # hepatic conc. [mg/L]
       k_gb    = k_gb,                                         # [1/d]
       u_rsv   = u_rsv,                                        # [mg/d]
       u_eze   = u_eze)                                        # [mg/d]
}

pk_only_ode <- function(time, state, parms) {
  pk <- pk_derivs(time, state, parms)
  list(pk$derivs, c(C_h_rsv = pk$C_h_rsv, C_c_rsv = pk$C_c_rsv))
}

# PK at periodic steady state for 1 mg/d of each drug (PK is linear in dose):
# returns the last dosing interval, time re-based to 0..1 [d]
pk_unit_profile <- function(parms, n_days = 14, step_d = 1 / 240) {
  parms_unit <- modifyList(parms, list(dose_rsv_mg = 1, dose_eze_mg = 1,     # [mg/d]
                                       start_rsv_d = 0, start_eze_d = 0))   # [d]
  init  <- c(A_rg = 0, A_rc = 0, A_rp = 0, A_rh = 0, E_gut = 0, E_pl = 0, E_gb = 0)  # [mg]
  times <- seq(0, n_days, by = step_d)                                       # [d]
  out <- deSolve::ode(y = init, times = times, func = pk_only_ode, parms = parms_unit,
                      method = "rk4")
  out_dt   <- as.data.table(as.data.frame(out))
  last_day <- out_dt[time >= n_days - 1 - 1e-9 & time < n_days - 1e-9]
  data.table(time         = last_day$time - (n_days - 1),     # [d]
             C_h_per_mg   = last_day$C_h_rsv,                 # hepatic rosuvastatin [mg/L per mg/d]
             E_gut_per_mg = last_day$E_gut)                   # gut ezetimibe [mg per mg/d]
}

# time-averaged fractional inhibition over a dosing interval [-]
mean_inhibition <- function(conc_per_mg, dose_mg, Imax, IC50) {
  if (dose_mg <= 0) return(0)
  conc <- dose_mg * conc_per_mg
  mean(Imax * conc / (IC50 + conc))
}

# set IC50_S and EC50_E so the reference doses give I_S_ref and I_E_ref
# (calibrated in the reference patient, f_OATP = 1)
calibrate_pd <- function(parms) {
  parms_ref <- modifyList(parms, list(f_OATP = 1))
  prof <- pk_unit_profile(parms = parms_ref)

  root_S <- uniroot(function(log_IC50) {
    mean_inhibition(conc_per_mg = prof$C_h_per_mg, dose_mg = parms$dose_ref_rsv,
                    Imax = parms$Imax_S, IC50 = exp(log_IC50)) - parms$I_S_ref
  }, lower = log(1e-9), upper = log(1e4))
  root_E <- uniroot(function(log_EC50) {
    mean_inhibition(conc_per_mg = prof$E_gut_per_mg, dose_mg = parms$dose_ref_eze,
                    Imax = parms$Emax_E, IC50 = exp(log_EC50)) - parms$I_E_ref
  }, lower = log(1e-9), upper = log(1e4))

  modifyList(parms, list(IC50_S = exp(root_S$root),           # [mg/L]
                         EC50_E = exp(root_E$root)))          # [mg]
}

# ---------------------------- coupled ODE ------------------------------------

chol_drug_ode <- function(time, state, parms) {
  list2env(c(parms, as.list(state)), envir = environment())

  # PK and drivers
  pk   <- pk_derivs(time, state, parms)
  D_in <- meal_intake_rate(time, D_daily = D_daily, meal_h = meal_h,
                           meal_frac = meal_frac, meal_sd_h = meal_sd_h)  # [mmol/d]

  # PD
  I_S <- Imax_S * pk$C_h_rsv / (IC50_S + pk$C_h_rsv)           # HMGCR inhibition [-]
  I_E <- Emax_E * E_gut / (EC50_E + E_gut)                     # NPC1L1 inhibition [-]

  # hepatic sterol sensing (SREBP2): negative feedback of C_L
  g        <- 1 / (1 + (C_L / K_C)^n_g)                        # SREBP2 activity [-]
  R_target <- (g / g0)^m_R                                     # LDLR target activity [-]

  # fluxes [mmol/d]
  S_hep <- S_max * g * (1 - I_S)                               # hepatic synthesis
  B     <- k_B * C_L                                           # biliary cholesterol
  BA    <- k_BA * C_L                                          # bile-acid synthesis
  A     <- k_abs * (1 - I_E) * C_lum                           # absorption (chylomicrons -> liver)
  F_neu <- k_fec * C_lum                                       # faecal neutral sterols
  P_V   <- P_V0 * (C_L / C_L0)^q_V                             # VLDL -> LDL production
  U_hep <- k_U * R_LDLR * LDL * V_p                            # hepatic LDLR uptake
  U_per <- k_nR * LDL * V_p                                    # peripheral LDL uptake
  RCT   <- k_RCT * C_P                                         # reverse cholesterol transport

  dC_lum  <- D_in + B - A - F_neu                              # [mmol/d]
  dC_L    <- A + S_hep + U_hep + RCT - B - BA - P_V            # [mmol/d]
  dR_LDLR <- (R_target - R_LDLR) / tau_R                       # [1/d]
  dLDL    <- (P_V - U_hep - U_per) / V_p                       # [mmol/L/d]
  dC_P    <- S_P + U_per - RCT                                 # [mmol/d]

  list(c(dC_lum, dC_L, dR_LDLR, dLDL, dC_P, pk$derivs),
       c(D_in = D_in, I_S = I_S, I_E = I_E, g = g, S_hep = S_hep, A = A, B = B, BA = BA,
         F_neu = F_neu, P_V = P_V, U_hep = U_hep, U_per = U_per, RCT = RCT,
         C_h_rsv = pk$C_h_rsv, C_c_rsv = pk$C_c_rsv, k_gb = pk$k_gb,
         u_rsv = pk$u_rsv, u_eze = pk$u_eze,
         LDL_mgdL = LDL * MW_chol / 10))                       # [mg/dL]
}

# ---------------------------- dynamic simulation -----------------------------

# starts from the patient's own drug-free steady state (from 03_steady_state.R);
# parms must be derived (derive_parms) and calibrated (calibrate_pd)
simulate_dynamic <- function(parms, dose_rsv_mg = 10, dose_eze_mg = 10,
                             start_rsv_d = 14, start_eze_d = 56, end_d = 98,
                             step_d = 1 / 120) {
  parms_dyn <- modifyList(parms, list(dose_rsv_mg = dose_rsv_mg,   # [mg/d]
                                      dose_eze_mg = dose_eze_mg,   # [mg/d]
                                      start_rsv_d = start_rsv_d,   # [d]
                                      start_eze_d = start_eze_d))  # [d]
  ss0 <- solve_steady_state(parms = parms_dyn, I_S = 0, I_E = 0)
  init <- c(C_lum  = unname(ss0$C_lum),   # [mmol]
            C_L    = unname(ss0$C_L),     # [mmol]
            R_LDLR = unname(ss0$R_LDLR),  # [-]
            LDL    = unname(ss0$LDL),     # [mmol/L]
            C_P    = unname(ss0$C_P),     # [mmol]
            A_rg = 0, A_rc = 0, A_rp = 0, A_rh = 0,   # [mg]
            E_gut = 0, E_pl = 0, E_gb = 0)            # [mg]
  stopifnot(identical(names(init), c(CHOL_STATES, PK_STATES)))

  times <- seq(0, end_d, by = step_d)                                        # [d]
  out <- deSolve::ode(y = init, times = times, func = chol_drug_ode, parms = parms_dyn,
                      method = "rk4")
  as.data.table(as.data.frame(out))
}
