test_that("both RK4 engines match the analytic solution of the toy model", {
  model <- load_model(toy_cfg(), variant = "fitted")
  parms <- model$parms_nominal                               # k_el = 0.2 [1/d], b_in = 1 [-]
  u <- c(u_in = 1.5)                                         # [mg/L/d]
  inflow <- 0.3                                              # [mg/L/d]
  dfun <- const_dfun(inflow = inflow)
  x0 <- c(conc = 2)                                          # [mg/L]
  grid <- make_grid(0, 10, dt = 0.1)                         # [d]
  k <- parms$k_el
  c_ss <- (parms$b_in * u[["u_in"]] + inflow) / k            # [mg/L]
  analytic <- c_ss + (x0[["conc"]] - c_ss) * exp(-k * grid)
  y_int <- integrate_states(model, x0 = x0, parms = parms, u = u, dfun = dfun,
                            grid = grid, engine = "rk4_internal")
  y_des <- integrate_states(model, x0 = x0, parms = parms, u = u, dfun = dfun,
                            grid = grid, engine = "deSolve")
  expect_equal(y_int[, 1], analytic, tolerance = 1e-7)
  expect_equal(y_des[, 1], analytic, tolerance = 1e-7)
  expect_equal(y_int[, 1], y_des[, 1], tolerance = 1e-10)
})

test_that("simulate_unit integrates costs and applies events", {
  model <- load_model(toy_cfg(), variant = "fitted")
  parms <- model$parms_nominal
  dfun <- const_dfun(inflow = 0.3)
  u_plan <- make_u_plan(times = 0, U = c(u_in = 1), control_names = "u_in")
  res <- simulate_unit(model, x0 = c(conc = 4), parms = parms, u_plan = u_plan, dfun = dfun,
                       t0 = 0, t1 = 5, dt = 0.05, out_times = c(1, 5))
  # input cost is constant: c_input * u_in * duration
  expect_equal(res$components[["cost_input"]], parms$c_input * 1 * 5, tolerance = 1e-10)
  expect_equal(nrow(res$y_out), 2)
  expect_equal(res$y_out$time, c(1, 5))
  ev <- data.table::data.table(time = 0, event = "flush")
  res_f <- simulate_unit(model, x0 = c(conc = 4), parms = parms, u_plan = u_plan, dfun = dfun,
                         t0 = 0, t1 = 5, dt = 0.05, events_plan = ev, out_times = 0)
  expect_equal(res_f$y_out$conc[1], 4 * (1 - parms$flush_frac))
  expect_equal(res_f$cost_events, parms$c_flush)
})
