# =============================================================================
# 02_model_v2.R
# Drivers w(t), dosing u(t), gut/PK sub-model, full ODE, calibration.
# Units: [mmol], [g], [mg], [L], [d] (see 01_parameters_v2.R)
# =============================================================================

# ---------------------------- drivers w(t) -----------------------------------

# nutrient delivery pulse of every meal (gastric emptying) [1/d];
# each pulse integrates to 1 over a day
meal_pulses <- function(time, meal_h, meal_sd_h) {
  hour_of_day <- (time %% 1) * 24                              # [h]
  delta_h     <- ((hour_of_day - meal_h + 12) %% 24) - 12       # signed distance to each meal [h]
  stats::dnorm(delta_h, sd = meal_sd_h) * 24                   # [1/d]
}

# ---------------------------- control u(t) -----------------------------------

# daily oral dose as smooth pulse around dose_hour from day start_d on [mg/d]
dosing_rate <- function(time, dose_mg, start_d, dose_hour, dose_sd_h) {
  if (dose_mg <= 0) return(0)
  hour_of_day <- (time %% 1) * 24                              # [h]
  delta_h     <- ((hour_of_day - dose_hour + 12) %% 24) - 12    # [h]
  time_dose   <- time - delta_h / 24                           # [d]
  if (time_dose < start_d - 1e-9) return(0)
  dose_mg * stats::dnorm(delta_h, sd = dose_sd_h) * 24         # [mg/d]
}

# ---------------------------- gut nutrients + drug PK ------------------------

# derivatives of GUT_PK_STATES plus the signals the rest of the model needs
gut_pk_derivs <- function(time, state, parms) {
  list2env(c(parms, as.list(state)), envir = environment())

  # w(t): nutrient inputs
  pulse  <- meal_pulses(time, meal_h = meal_h, meal_sd_h = meal_sd_h)  # [1/d] per meal
  D_in   <- D_daily * sum(chol_frac * pulse)                   # cholesterol [mmol/d]
  fat_in <- fat_daily_g * sum(fat_frac * pulse)                # fat [g/d]
  fib_in <- fib_daily_g * sum(fib_frac * pulse)                # fibre [g/d]
  ps_in  <- ps_daily_g * 1000 / MW_ps * sum(ps_frac * pulse)   # plant sterols [mmol/d]

  # fat -> CCK -> gallbladder contraction and sphincter of Oddi relaxation
  dFat_J <- fat_in - k_fat * Fat_J                             # [g/d]
  dCCK   <- (Fat_J / Fat_ref_cck - CCK) / tau_cck              # [1/d]
  a_cck  <- CCK / (K_cck + CCK)                                # CCK activity [-]
  phi_GB <- phi_fast * (1 - a_cck)                             # hepatic bile to gallbladder [-]
  k_gb   <- k_gb_mmc + k_gb_max * a_cck                        # gallbladder emptying [1/d]

  # fibre (binds BA in the ileum) and plant sterols (compete in micelles)
  dFib_I <- fib_in - k_tr_fib * Fib_I                          # [g/d]
  b_fib  <- b_max_fib * Fib_I / (K_fib + Fib_I)                # bound BA fraction [-]
  dPS_J  <- ps_in - k_tr * PS_J                                # [mmol/d]

  # u(t)
  u_rsv <- dosing_rate(time, dose_mg = dose_rsv_mg, start_d = start_rsv_d,
                       dose_hour = dose_hour, dose_sd_h = dose_sd_h)  # [mg/d]
  u_eze <- dosing_rate(time, dose_mg = dose_eze_mg, start_d = start_eze_d,
                       dose_hour = dose_hour, dose_sd_h = dose_sd_h)  # [mg/d]

  # rosuvastatin [mg]
  C_c_rsv   <- A_rc / V_c_rsv                                  # [mg/L]
  C_per_rsv <- A_rp / V_per_rsv                                # [mg/L]
  upt_rsv   <- CL_upt_rsv * f_OATP * C_c_rsv                   # [mg/d]
  dA_rg <- F_rsv * u_rsv - ka_rsv * A_rg                       # [mg/d]
  dA_rc <- ka_rsv * A_rg - upt_rsv + k_hc_rsv * A_rh -
           CL_ren_rsv * C_c_rsv - Q_rsv * (C_c_rsv - C_per_rsv) # [mg/d]
  dA_rp <- Q_rsv * (C_c_rsv - C_per_rsv)                       # [mg/d]
  dA_rh <- upt_rsv - (k_hc_rsv + k_bil_rsv) * A_rh             # [mg/d]

  # ezetimibe: biliary excretion follows the hepatic bile (gallbladder or gut)
  J_E_bile <- k_bile_eze * E_pl                                # [mg/d]
  dE_gut <- u_eze + (1 - phi_GB) * J_E_bile + k_gb * E_gb -
            (ka_eze + k_fec_eze) * E_gut                       # [mg/d]
  dE_pl  <- ka_eze * E_gut - (k_bile_eze + k_ren_eze) * E_pl   # [mg/d]
  dE_gb  <- phi_GB * J_E_bile - k_gb * E_gb                    # [mg/d]

  # NPC1L1 occupancy (receptor amount negligible vs drug amount)
  dO_NPC <- k_on_NPC * E_gut * (1 - O_NPC) - k_off_NPC * O_NPC # [1/d]

  list(derivs  = c(dFat_J, dCCK, dFib_I, dPS_J, dA_rg, dA_rc, dA_rp, dA_rh,
                   dE_gut, dE_pl, dE_gb, dO_NPC),
       D_in = D_in, fat_in = fat_in,                           # [mmol/d], [g/d]
       a_cck = a_cck, phi_GB = phi_GB, k_gb = k_gb, b_fib = b_fib,  # [-], [-], [1/d], [-]
       C_h_rsv = A_rh / V_h, C_c_rsv = C_c_rsv,                # [mg/L]
       u_rsv = u_rsv, u_eze = u_eze)                           # [mg/d]
}

gut_pk_ode <- function(time, state, parms) {
  gut <- gut_pk_derivs(time, state, parms)
  list(gut$derivs, c(C_h_rsv = gut$C_h_rsv, k_gb = gut$k_gb, a_cck = gut$a_cck))
}

# ---------------------------- full ODE ---------------------------------------

chol_v2_ode <- function(time, state, parms) {
  list2env(c(parms, as.list(state)), envir = environment())
  gut <- gut_pk_derivs(time, state, parms)
  phi_GB <- gut$phi_GB                                         # [-]
  k_gb   <- gut$k_gb                                           # [1/d]

  # pharmacodynamics
  I_S <- Imax_S * gut$C_h_rsv / (IC50_S + gut$C_h_rsv)         # HMGCR inhibition [-]
  I_E <- if (occupancy_kinetic) Emax_E * O_NPC else Emax_E * E_gut / (EC50_E + E_gut)  # [-]

  # bile acids; hepatocyte BA pool quasi-steady: secretion = portal return + synthesis
  J_asbt <- k_asbt * (1 - gut$b_fib) * BA_I                    # ileal reabsorption [mmol/d]
  h_Z    <- (1 + K_Z^n_Z) / (1 + (K_Z * Z_FXR)^n_Z)            # CYP7A1 de-repression [-]
  J_syn  <- k_syn_BA * C_L * h_Z                               # BA synthesis = cholesterol use [mmol/d]
  J_sec  <- J_asbt + J_syn                                     # hepatic BA secretion [mmol/d]
  J_colB <- (k_col + k_asbt * gut$b_fib) * BA_I                # faecal BA loss [mmol/d]
  dBA_GB <- phi_GB * J_sec - k_gb * BA_GB                      # [mmol/d]
  dBA_J  <- (1 - phi_GB) * J_sec + k_gb * BA_GB - k_tr * BA_J  # [mmol/d]
  dBA_I  <- k_tr * BA_J - J_asbt - J_colB                      # [mmol/d]
  dZ_FXR <- (J_asbt / J_asbt_ref - Z_FXR) / tau_Z              # [1/d]

  # biliary cholesterol, coupled to BA secretion and hepatic cholesterol
  B <- B0 * k_B_factor * (C_L / C_L0) * (J_sec / J_BA0)^nu_B   # [mmol/d]

  # micelle-gated absorption (smooth min of lumen cholesterol and micellar capacity)
  cap   <- r_mic * BA_J * (m0_mic + (1 - m0_mic) * Fat_J / (K_fat_mic + Fat_J))  # [mmol]
  C_mic <- C_lum * cap / (C_lum + a_ps * PS_J + cap + 1e-12)   # micellar cholesterol [mmol]
  A     <- k_abs * (1 - I_E) * C_mic                           # absorption [mmol/d]
  F_neu <- k_tr * C_lum                                        # to colon / faeces [mmol/d]

  dC_GB  <- phi_GB * B - k_gb * C_GB                           # [mmol/d]
  dC_lum <- gut$D_in + (1 - phi_GB) * B + k_gb * C_GB - A - F_neu  # [mmol/d]

  # hepatic regulation and synthesis
  hour_of_day <- (time %% 1) * 24                              # [h]
  circ   <- 1 + a_circ * cos(2 * pi * (hour_of_day - circ_peak_h) / 24)  # [-]
  g      <- 1 / (1 + (C_L / K_C)^n_g)                          # SREBP2 activity [-]
  S_hep  <- S_max * g * (1 - I_S) * circ                       # [mmol/d]
  R_target <- (g / g0)^m_R                                     # [-]

  # lipoproteins and periphery
  P_V   <- P_V0 * (C_L / C_L0)^q_V                             # [mmol/d]
  U_hep <- k_U * R_LDLR * LDL * V_p                            # [mmol/d]
  U_per <- k_nR * LDL * V_p                                    # [mmol/d]
  RCT   <- if (rct_quasi_ss) S_P + U_per else k_RCT * C_P      # [mmol/d]

  dC_L    <- A + S_hep + U_hep + RCT - B - J_syn - P_V         # [mmol/d]
  dR_LDLR <- (R_target - R_LDLR) / tau_R                       # [1/d]
  dLDL    <- (P_V - U_hep - U_per) / V_p                       # [mmol/L/d]
  dC_P    <- if (rct_quasi_ss) 0 else S_P + U_per - RCT        # [mmol/d]

  list(c(gut$derivs, dBA_GB, dBA_J, dBA_I, dZ_FXR, dC_GB, dC_lum, dC_L, dR_LDLR, dLDL, dC_P),
       c(D_in = gut$D_in, fat_in = gut$fat_in, a_cck = gut$a_cck, phi_GB = phi_GB, k_gb = k_gb,
         I_S = I_S, I_E = I_E, J_asbt = J_asbt, J_syn = J_syn, J_sec = J_sec, J_colB = J_colB,
         h_Z = h_Z, B = B, cap = cap, C_mic = C_mic, A = A, F_neu = F_neu,
         S_hep = S_hep, P_V = P_V, U_hep = U_hep,
         BA_pool = BA_GB + BA_J + BA_I,                        # [mmol]
         litho_GB = C_GB / (BA_GB + 1e-9),                     # cholesterol/BA in gallbladder [mol/mol]
         LDL_mgdL = LDL * MW_chol / 10))                       # [mg/dL]
}

# ---------------------------- initial state and simulation -------------------

# rough initial state from the mean-field calibration; refine with a burn-in
make_init_v2 <- function(parms) {
  init <- c(Fat_J = 0, CCK = 0, Fib_I = 0, PS_J = 0,           # [g], [-], [g], [mmol]
            A_rg = 0, A_rc = 0, A_rp = 0, A_rh = 0,            # [mg]
            E_gut = 0, E_pl = 0, E_gb = 0, O_NPC = 0,          # [mg], [-]
            BA_GB = 3,                                         # [mmol]
            BA_J  = unname(parms$BA_J0),                       # [mmol]
            BA_I  = unname(parms$J_BA0 / parms$k_ileum),       # [mmol]
            Z_FXR = 1,                                         # [-]
            C_GB  = 0.2,                                       # [mmol]
            C_lum = unname(parms$C_lum0),                      # [mmol]
            C_L   = unname(parms$C_L0),                        # [mmol]
            R_LDLR = 1,                                        # [-]
            LDL   = unname(parms$LDL0),                        # [mmol/L]
            C_P   = unname(parms$C_P0))                        # [mmol]
  stopifnot(identical(names(init), ALL_STATES_V2))
  init
}

simulate_v2 <- function(parms, init, n_days = 40, step_d = 1 / 120) {
  times <- seq(0, n_days, by = step_d)                         # [d]
  out <- deSolve::ode(y = init, times = times, func = chol_v2_ode, parms = parms,
                      method = "rk4")
  as.data.table(as.data.frame(out))
}

# final state of a run, as named initial-state vector for the next run
end_state <- function(out_dt) {
  last_row <- out_dt[.N]
  init <- unlist(last_row[, ..ALL_STATES_V2])
  init[c("A_rg", "A_rc", "A_rp", "A_rh", "E_gut", "E_pl", "E_gb", "O_NPC")] <- 0  # drug-free start
  init
}

# ---------------------------- PD calibration ---------------------------------

# periodic-steady-state profiles over the last day for 1 mg/d of each drug
gut_pk_profile <- function(parms, n_days = 14, step_d = 1 / 240) {
  parms_unit <- modifyList(parms, list(dose_rsv_mg = 1, dose_eze_mg = 1,
                                       start_rsv_d = 0, start_eze_d = 0, k_on_NPC = 0))
  init  <- c(Fat_J = 0, CCK = 0, Fib_I = 0, PS_J = 0, A_rg = 0, A_rc = 0, A_rp = 0, A_rh = 0,
             E_gut = 0, E_pl = 0, E_gb = 0, O_NPC = 0)
  stopifnot(identical(names(init), GUT_PK_STATES))
  times <- seq(0, n_days, by = step_d)                         # [d]
  out <- deSolve::ode(y = init, times = times, func = gut_pk_ode, parms = parms_unit,
                      method = "rk4")
  out_dt   <- as.data.table(as.data.frame(out))
  last_day <- out_dt[time >= n_days - 1 - 1e-9 & time < n_days - 1e-9]
  data.table(time         = last_day$time - (n_days - 1),     # [d]
             C_h_per_mg   = last_day$C_h_rsv,                 # [mg/L per mg/d]
             E_gut_per_mg = last_day$E_gut)                   # [mg per mg/d]
}

# time-averaged Emax-type inhibition (instantaneous equilibrium) [-]
mean_inhibition <- function(conc_per_mg, dose_mg, Imax, IC50) {
  if (dose_mg <= 0) return(0)
  conc <- dose_mg * conc_per_mg
  mean(Imax * conc / (IC50 + conc))
}

# time-averaged NPC1L1 occupancy at periodic steady state [-];
# exact exponential update per step (E_gut constant within a step)
occupancy_mean <- function(E_gut_per_mg, dose_mg, k_on, k_off, step_d, n_cycles = 8) {
  E_gut   <- dose_mg * E_gut_per_mg                            # [mg]
  occ     <- 0                                                 # [-]
  occ_day <- numeric(length(E_gut))
  cyc <- 1
  for (cyc in seq_len(n_cycles)) {
    i <- 1
    for (i in seq_along(E_gut)) {
      k_tot  <- k_on * E_gut[i] + k_off                        # [1/d]
      occ_eq <- k_on * E_gut[i] / k_tot                        # [-]
      occ    <- occ_eq + (occ - occ_eq) * exp(-k_tot * step_d)
      occ_day[i] <- occ
    }
  }
  mean(occ_day)
}

# IC50_S, EC50_E (equilibrium mode) and k_on_NPC (kinetic mode) such that the
# reference regimen (3 meals, pill 08:00, 10 mg) gives I_S_ref and I_E_ref
calibrate_pd_v2 <- function(parms, step_d = 1 / 240) {
  parms_ref <- modifyList(parms, list(f_OATP = 1))
  prof <- gut_pk_profile(parms = parms_ref, step_d = step_d)

  root_S <- uniroot(function(log_IC50) {
    mean_inhibition(conc_per_mg = prof$C_h_per_mg, dose_mg = parms$dose_ref_rsv,
                    Imax = parms$Imax_S, IC50 = exp(log_IC50)) - parms$I_S_ref
  }, lower = log(1e-9), upper = log(1e4))
  root_E <- uniroot(function(log_EC50) {
    mean_inhibition(conc_per_mg = prof$E_gut_per_mg, dose_mg = parms$dose_ref_eze,
                    Imax = parms$Emax_E, IC50 = exp(log_EC50)) - parms$I_E_ref
  }, lower = log(1e-9), upper = log(1e4))
  root_on <- uniroot(function(log_kon) {
    parms$Emax_E * occupancy_mean(E_gut_per_mg = prof$E_gut_per_mg, dose_mg = parms$dose_ref_eze,
                                  k_on = exp(log_kon), k_off = parms$k_off_NPC,
                                  step_d = step_d) - parms$I_E_ref
  }, lower = log(1e-6), upper = log(1e6))

  modifyList(parms, list(IC50_S   = exp(root_S$root),         # [mg/L]
                         EC50_E   = exp(root_E$root),         # [mg]
                         k_on_NPC = exp(root_on$root)))       # [1/(mg d)]
}
