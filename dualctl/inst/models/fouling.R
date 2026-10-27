# =============================================================================
# fouling.R  --  dualctl model plug-in: cooling-water heat-exchanger fouling
#
# ILLUSTRATIVE, NOT VALIDATED. Plausible SI values, no plant data behind them.
#
# A shell-and-tube cooler: a process stream (shell side, C_h [kW/K], inlet
# T_h_in [degC]) is cooled by cooling water in the tubes (velocity v [m/s],
# inlet T_cw_in [degC]). Scale deposits on the water side (inverse-solubility
# salts: warmer surface -> faster deposition).
#
# Fitted model (one state):
#   dRf/dt = dep - rem                                         [m2K/kW/d]
#   dep    = alpha_dep * (Re/Re_ref)^-0.66 * (Pr/Pr_ref)^-0.33
#            * exp(-E_act/R * (1/T_film - 1/T_ref)) * c_f * exp(-k_inh * dose)
#   rem    = gamma_rem * (tau_w/tau_ref) * Rf/(Rf + Rf_eps)
# Ebert & Panchal (1997) threshold form (deposition minus wall-shear removal),
# re-parameterised around reference conditions so that alpha_dep and E_act
# are less correlated. The factor Rf/(Rf + Rf_eps) lets removal vanish on a
# clean surface (keeps Rf >= 0 smoothly). Cleaning: Rf -> (1 - eta_clean) Rf.
#
# Truth (two states): adds deposit hardness H [-] (ageing). Hard deposit is
# not removed by shear and resists cleaning:
#   dH/dt  = k_age (1 - H) - H (dep - rem)/(Rf + Rf_eps)
#   rem    = rem_fitted * (1 - mismatch * H)
#   clean  : Rf -> (1 - eta_clean (1 - mismatch * hard_clean_penalty * H)) Rf
# `mismatch` in [0, 1] sets the strength of the structural error the fitted
# model cannot represent (0 = H has no effect).
#
# Derived quantities: Dittus-Boelter tube-side film coefficient, Blasius
# friction, effectiveness-NTU counterflow duty, outlet temperatures, film
# temperature T_film = T_bulk + f_film * q / h_i (Ebert-Panchal convention),
# tube narrowing by the deposit (thickness = Rf * lambda_dep) which raises the
# pressure drop, pump power over exchanger + circuit losses.
#
# Costs [EUR/d]: cooling shortfall (duty below the requirement, smooth max),
# pumping electricity, inhibitor (dose x treated make-up volume); cleaning is
# a lump event (labour + downtime valued as lost duty).
#
# References (for the functional forms only):
#   Ebert W, Panchal CB (1997) Analysis of Exxon crude-oil-slip stream coking
#     data. In: Fouling Mitigation of Industrial Heat-Exchange Equipment,
#     Begell House, 451-460.
#   Incropera FP et al. (2007) Fundamentals of Heat and Mass Transfer, 6th ed.
#     (Dittus-Boelter, effectiveness-NTU, Blasius).
# =============================================================================

# water properties as functions of temperature [degC] (fits valid ~ 0-60 degC)
water_props <- function(T_C) {
  T_K <- T_C + 273.15                                       # [K]
  rho <- 999.8 - 0.0045 * (T_C - 4)^2                       # [kg/m3] density
  mu  <- 2.414e-5 * 10^(247.8 / (T_K - 140))                # [Pa s] viscosity (Vogel-type fit)
  tr  <- T_K / 298.15                                       # [-]
  k_w <- 0.6065 * (-1.48445 + 4.12292 * tr - 1.63866 * tr^2) # [W/(m K)] conductivity
  list(rho = rho, mu = mu, k_w = k_w)
}

# all algebraic quantities of the exchanger for state x, controls u, disturbances d
hx_derived <- function(x, u, d, parms) {
  list2env(c(parms, as.list(x), as.list(u), as.list(d)), envir = environment())

  wp  <- water_props(T_cw_in)
  rho <- wp$rho                                             # [kg/m3]
  mu  <- wp$mu                                              # [Pa s]
  k_w <- wp$k_w                                             # [W/(m K)]
  Pr  <- cp_w * 1000 * mu / k_w                             # [-] Prandtl number

  # deposit narrows the tubes
  delta_dep <- max(Rf, 0) / 1000 * lambda_dep               # [m] (Rf [m2K/kW] -> [m2K/W])
  D_eff <- max(D_i - 2 * delta_dep, 0.5 * D_i)              # [m]
  v_eff <- v * (D_i / D_eff)^2                              # [m/s] continuity

  Re     <- rho * v * D_i / mu                              # [-] clean-tube Reynolds number
  Re_eff <- rho * v_eff * D_eff / mu                        # [-]
  f_F    <- 0.079 * Re_eff^(-0.25)                          # [-] Fanning friction factor (Blasius)
  tau_w  <- 0.5 * f_F * rho * v_eff^2                       # [Pa] wall shear at deposit surface

  h_i <- 0.023 * Re^0.8 * Pr^0.4 * k_w / D_i / 1000         # [kW/(m2 K)] Dittus-Boelter
  U0  <- f_U0 / (1 / h_i + 1 / h_o + R_wall)                # [kW/(m2 K)] clean overall coefficient
  U   <- 1 / (1 / U0 + max(Rf, 0))                          # [kW/(m2 K)] fouled

  Vdot  <- v * A_flow                                       # [m3/s] cooling-water flow
  C_c   <- rho * Vdot * cp_w                                # [kW/K]
  C_h   <- C_h_nom * load                                   # [kW/K]
  C_min <- min(C_c, C_h)                                    # [kW/K]
  Cr    <- C_min / max(C_c, C_h)                            # [-]
  NTU   <- U * A_ht / C_min                                 # [-]
  eps   <- if (abs(1 - Cr) < 1e-6) NTU / (1 + NTU) else
    (1 - exp(-NTU * (1 - Cr))) / (1 - Cr * exp(-NTU * (1 - Cr)))   # [-] counterflow effectiveness
  Q       <- eps * C_min * (T_h_in - T_cw_in)               # [kW] duty
  T_h_out <- T_h_in - Q / C_h                               # [degC]
  T_c_out <- T_cw_in + Q / C_c                              # [degC]
  q_flux  <- Q / A_ht                                       # [kW/m2]
  T_film  <- 0.5 * (T_cw_in + T_c_out) + f_film * q_flux / h_i  # [degC] film temperature

  dP_hx  <- (4 * f_F * L_tube / D_eff + K_minor) * rho * v_eff^2 / 2 / 1000  # [kPa]
  dP_sys <- K_sys * rho * v^2 / 2 / 1000                    # [kPa] rest of the circuit
  P_pump <- (dP_hx + dP_sys) * Vdot / eta_pump              # [kW] (kPa * m3/s = kW)
  Q_req  <- C_h * (T_h_in - T_h_out_target)                 # [kW] required duty

  list(Re = Re, Pr = Pr, tau_w = tau_w, h_i = h_i, U0 = U0, U = U, Q = Q,
       T_h_out = T_h_out, T_film = T_film, dP_hx = dP_hx, P_pump = P_pump,
       Q_req = Q_req)
}

# deposition and removal rates [m2K/kW/d]
hx_rates <- function(x, u, d, parms, alg) {
  list2env(c(parms, as.list(x), as.list(u), alg), envir = environment())
  dep <- alpha_dep * (Re / Re_ref)^(-0.66) * (Pr / Pr_ref)^(-0.33) *
    exp(-E_act * 1000 / R_gas * (1 / (T_film + 273.15) - 1 / T_ref_K)) *
    d[["c_f"]] * exp(-k_inh * dose)                         # [m2K/kW/d]
  rem <- gamma_rem * (tau_w / tau_ref) * max(Rf, 0) / (max(Rf, 0) + Rf_eps)  # [m2K/kW/d]
  c(dep = dep, rem = rem)
}

rhs_fitted <- function(time, x, u, d, parms) {
  alg <- hx_derived(x, u, d, parms)
  r <- hx_rates(x, u, d, parms, alg)
  dRf <- r[["dep"]] - r[["rem"]]                            # [m2K/kW/d]
  c(dRf)
}

rhs_truth <- function(time, x, u, d, parms) {
  alg <- hx_derived(x, u, d, parms)
  r <- hx_rates(x, u, d, parms, alg)
  list2env(c(parms, as.list(x)), envir = environment())
  H_c <- min(max(H, 0), 1)                                  # [-]
  rem <- r[["rem"]] * (1 - mismatch * H_c)                  # [m2K/kW/d] hard deposit resists shear
  dRf <- r[["dep"]] - rem                                   # [m2K/kW/d]
  dH  <- k_age * (1 - H_c) - H_c * dRf / (max(Rf, 0) + Rf_eps)  # [1/d]
  if (H >= 1 && dH > 0) dH <- 0
  if (H <= 0 && dH < 0) dH <- 0
  c(dRf, dH)
}

hx_outputs <- function(time, x, u, d, parms) {
  alg <- hx_derived(x, u, d, parms)
  list2env(c(parms, as.list(x), as.list(u), alg), envir = environment())
  c(U_obs          = U,                                     # [kW/(m2 K)] from temperatures and flows
    dP_obs         = dP_hx,                                 # [kPa] exchanger pressure drop
    Rf_probe       = max(Rf, 0),                            # [m2K/kW] direct fouling monitor
    T_h_out_obs    = T_h_out,                               # [degC]
    duty           = Q,                                     # [kW]
    cost_shortfall = c_short * 24 * softplus(Q_req - Q, s = s_soft),  # [EUR/d]
    cost_pumping   = c_el * 24 * P_pump,                    # [EUR/d]
    cost_dosing    = c_inh * dose * V_treat / 1000,         # [EUR/d] (g/m3 * m3/d -> kg/d)
    g_T_h_out      = T_h_out - T_h_out_max,                 # [degC] > 0 violates
    g_dP           = dP_hx - dP_max)                        # [kPa]  > 0 violates
}

# cleaning: lump cost = labour + downtime valued as lost required duty [EUR]
clean_cost <- function(x, parms, u, d) {
  list2env(parms, envir = environment())
  Q_req_nom <- C_h_nom * (T_h_in - T_h_out_target)          # [kW]
  c_clean_labor + downtime_d * 24 * c_short * Q_req_nom
}

clean_fitted <- function(x, parms) {
  list2env(c(parms, as.list(x)), envir = environment())
  c((1 - eta_clean) * Rf)
}

clean_truth <- function(x, parms) {
  list2env(c(parms, as.list(x)), envir = environment())
  H_c <- min(max(H, 0), 1)
  eta_eff <- eta_clean * (1 - mismatch * hard_clean_penalty * H_c)  # [-]
  c((1 - eta_eff) * Rf, H)
}

# synthetic weather: seasonal cooling-water temperature with AR(1) noise and
# storms (cold, turbid water), foulant index with slow drift and storm spikes,
# process load with a weekly cycle
generate_weather <- function(times, cfg, seed) {
  dc <- cfg$model$disturbances
  with_seed(seed, {
    n <- length(times)
    Tc <- dc$T_cw_in
    T_clim <- Tc$mean + Tc$amplitude * cos(2 * pi * (times - Tc$peak_day) / 365)   # [degC]
    storms <- storm_series(times, rate_per_day = dc$storms$rate_per_day,
                           duration_d = dc$storms$duration_d)                       # [-]
    T_cw_in <- T_clim + ar1_series(n, phi = Tc$ar_phi, sd = Tc$ar_sd) -
      dc$storms$T_drop * storms                                                     # [degC]
    cf <- dc$c_f
    c_f <- cf$mean * exp(ou_series(times, theta = cf$ou_theta, sd = cf$ou_sd)) *
      (1 + dc$storms$c_f_spike * storms)                                            # [-]
    ld <- dc$load
    load_clim <- 1 + ld$weekly_amplitude * sin(2 * pi * times / 7)                  # [-]
    load <- load_clim + ar1_series(n, phi = ld$ar_phi, sd = ld$ar_sd)               # [-]
    data.table::data.table(time = times, T_cw_in = T_cw_in, c_f = c_f, load = load,
                           T_cw_in_clim = T_clim, c_f_clim = cf$mean, load_clim = load_clim)
  })
}

build_model <- function(cfg, variant = "fitted") {
  mc <- cfg$model
  const <- lapply(mc$constants, as.numeric)
  # geometry-derived constants
  const$A_ht   <- pi * const$D_i * const$L_tube * const$N_tubes      # [m2] heat-transfer area
  const$A_flow <- const$N_tubes * pi * const$D_i^2 / 4               # [m2] tube flow area
  states <- mc$states
  rhs <- rhs_fitted
  clean_apply <- clean_fitted
  if (identical(variant, "truth")) {
    const  <- c(const, lapply(mc$truth_constants, as.numeric))
    states <- c(states, mc$truth_states)
    rhs <- rhs_truth
    clean_apply <- clean_truth
  }
  ev_cfg <- mc$events$clean
  model_spec(
    name = mc$name, variant = variant,
    description = "Cooling-water heat-exchanger fouling (Ebert-Panchal style); illustrative values",
    states = states, params = mc$parameters, constants = const, controls = mc$controls,
    events = list(clean = list(apply = clean_apply, cost = clean_cost,
                               change_cost = ev_cfg$change_cost,
                               min_interval_epochs = ev_cfg$min_interval_epochs)),
    disturbances = list(names = c("T_cw_in", "c_f", "load"),
                        units = c("degC", "-", "-"), generate = generate_weather),
    rhs = rhs, outputs = hx_outputs, sensor_sets = mc$sensor_sets,
    cost_outputs = c("cost_shortfall", "cost_pumping", "cost_dosing"),
    constraint_outputs = c("g_T_h_out", "g_dP"))
}
