summer <- function() const_dfun(T_cw_in = 25, c_f = 1, load = 1)

test_that("fouling model loads in both variants", {
  cfg <- fouling_cfg()
  fit <- load_model(cfg, variant = "fitted")
  tru <- load_model(cfg, variant = "truth")
  expect_equal(fit$state_names, "Rf")
  expect_equal(tru$state_names, c("Rf", "H"))
  expect_equal(fit$phi_names, tru$phi_names)
})

test_that("clean exchanger has plausible U, pressure drop and duty", {
  fit <- load_model(fouling_cfg(), variant = "fitted")
  out <- fit$outputs(0, c(Rf = 0), c(v = 1.5, dose = 2), summer()(0), fit$parms_nominal)
  expect_gt(out[["U_obs"]], 1.0)                             # [kW/(m2 K)]
  expect_lt(out[["U_obs"]], 3.0)
  expect_gt(out[["dP_obs"]], 5)                              # [kPa]
  expect_lt(out[["dP_obs"]], 30)
  expect_gt(out[["duty"]], 3000)                             # [kW]
})

test_that("fouling grows without inhibitor and is slowed by velocity", {
  fit <- load_model(fouling_cfg(), variant = "fitted")
  grid <- make_grid(0, 60, dt = 1)
  run <- function(v) {
    y <- integrate_states(fit, x0 = c(Rf = 0.02), parms = fit$parms_nominal,
                          u = c(v = v, dose = 0), dfun = summer(), grid = grid)
    y[nrow(y), 1]
  }
  expect_gt(run(1.0), 0.02)
  expect_lt(run(2.5), run(1.0))
})

test_that("cleaning lowers Rf; with mismatch 0 truth and fitted Rf agree", {
  cfg <- fouling_cfg()
  fit <- load_model(cfg, variant = "fitted")
  tru <- load_model(cfg, variant = "truth")
  expect_lt(fit$events$clean$apply(c(Rf = 0.4), fit$parms_nominal), 0.4)
  p_tru <- tru$parms_nominal
  p_tru$mismatch <- 0
  grid <- make_grid(0, 30, dt = 1)
  y_fit <- integrate_states(fit, x0 = c(Rf = 0.1), parms = fit$parms_nominal,
                            u = c(v = 1.5, dose = 1), dfun = summer(), grid = grid)
  y_tru <- integrate_states(tru, x0 = c(Rf = 0.1, H = 0.5), parms = p_tru,
                            u = c(v = 1.5, dose = 1), dfun = summer(), grid = grid)
  expect_equal(y_tru[, 1], y_fit[, 1], tolerance = 1e-10)
})
