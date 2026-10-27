# =============================================================================
# toy_decay.R  --  minimal dualctl model plug-in (used by the tests)
#
# A well-mixed tank with first-order decay, a dosed input and an inflow load:
#   dconc/dt = -k_el * conc + b_in * u_in + inflow          [mg/L/d]
# Observation: conc (noisy). Cost: quadratic tracking of conc_target plus
# input cost. Event "flush": conc -> (1 - flush_frac) * conc with a lump cost.
# For constant u_in and inflow the analytic solution is
#   conc(t) = c_ss + (conc_0 - c_ss) exp(-k_el t),  c_ss = (b_in u_in + inflow)/k_el
# which the ODE test uses. The file exists to prove that a new model needs
# only a plug-in file plus a config, with no engine changes.
# =============================================================================

toy_rhs <- function(time, x, u, d, parms) {
  list2env(c(parms, as.list(x), as.list(u), as.list(d)), envir = environment())
  dconc <- -k_el * conc + b_in * u_in + inflow              # [mg/L/d]
  c(dconc)
}

toy_outputs <- function(time, x, u, d, parms) {
  list2env(c(parms, as.list(x), as.list(u), as.list(d)), envir = environment())
  c(conc_obs   = conc,                                      # [mg/L]
    cost_track = w_track * (conc - conc_target)^2,          # [EUR/d]
    cost_input = c_input * u_in,                            # [EUR/d]
    g_conc     = conc - conc_max)                           # [mg/L] > 0 violates
}

toy_flush <- function(x, parms) {
  list2env(c(parms, as.list(x)), envir = environment())
  c((1 - flush_frac) * conc)
}

toy_flush_cost <- function(x, parms, u, d) {
  list2env(parms, envir = environment())
  c_flush
}

toy_inflow <- function(times, cfg, seed) {
  dc <- cfg$model$disturbances$inflow
  with_seed(seed, {
    clim <- dc$mean + dc$amplitude * sin(2 * pi * times / dc$period_d)   # [mg/L/d]
    data.table::data.table(time = times,
                           inflow = clim + ar1_series(length(times), phi = dc$ar_phi, sd = dc$ar_sd),
                           inflow_clim = clim)
  })
}

build_model <- function(cfg, variant = "fitted") {
  mc <- cfg$model
  model_spec(
    name = mc$name, variant = variant, description = "toy first-order decay tank",
    states = mc$states, params = mc$parameters,
    constants = lapply(mc$constants, as.numeric), controls = mc$controls,
    events = list(flush = list(apply = toy_flush, cost = toy_flush_cost,
                               change_cost = mc$events$flush$change_cost,
                               min_interval_epochs = mc$events$flush$min_interval_epochs)),
    disturbances = list(names = "inflow", units = "mg/L/d", generate = toy_inflow),
    rhs = toy_rhs, outputs = toy_outputs, sensor_sets = mc$sensor_sets,
    cost_outputs = c("cost_track", "cost_input"), constraint_outputs = "g_conc")
}
