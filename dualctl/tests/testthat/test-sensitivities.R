test_that("forward sensitivity equations agree with finite differences (toy, with event)", {
  model <- load_model(toy_cfg(), variant = "fitted")
  parms <- phi_to_parms(model, phi = c(k_el = log(0.25), b_in = log(1.2)))
  dfun <- const_dfun(inflow = 0.3)
  u_plan <- make_u_plan(times = c(0, 5), U = c(0.5, 2.0), control_names = "u_in")
  ev <- data.table::data.table(time = 5, event = "flush")
  obs <- seq(1, 10, by = 1)
  args <- list(model = model, x0 = c(conc = 2), parms = parms, u_plan = u_plan, dfun = dfun,
               t0 = 0, t1 = 10, obs_times = obs, outputs = "conc_obs", events_plan = ev,
               h = 1e-4, dt = 0.25)
  s_fd <- do.call(compute_sensitivities, c(args, list(method = "fd_central")))
  s_od <- do.call(compute_sensitivities, c(args, list(method = "forward_ode")))
  expect_equal(s_od$times, s_fd$times)
  expect_equal(s_od$y, s_fd$y, tolerance = 1e-8)
  expect_equal(s_od$S, s_fd$S, tolerance = 1e-5)
})

test_that("forward sensitivity equations agree with finite differences (fouling)", {
  model <- load_model(fouling_cfg(), variant = "fitted")
  dfun <- const_dfun(T_cw_in = 24, c_f = 1.1, load = 1)
  u_plan <- make_u_plan(times = c(0, 15), U = rbind(c(1.4, 1), c(1.8, 3)),
                        control_names = c("v", "dose"))
  args <- list(model = model, x0 = c(Rf = 0.1), parms = model$parms_nominal, u_plan = u_plan,
               dfun = dfun, t0 = 0, t1 = 30, obs_times = seq(2, 30, by = 2),
               outputs = c("U_obs", "dP_obs", "Rf_probe"), h = 1e-4, dt = 1)
  s_fd <- do.call(compute_sensitivities, c(args, list(method = "fd_central")))
  s_od <- do.call(compute_sensitivities, c(args, list(method = "forward_ode")))
  expect_equal(s_od$S, s_fd$S, tolerance = 1e-4)
})

test_that("Fisher information is symmetric PSD and CRBs are finite for the toy model", {
  model <- load_model(toy_cfg(), variant = "fitted")
  dfun <- const_dfun(inflow = 0.3)
  u_plan <- make_u_plan(times = c(0, 10, 20), U = c(0.5, 2.5, 1.0), control_names = "u_in")
  sens <- compute_sensitivities(model, x0 = c(conc = 2), parms = model$parms_nominal,
                                u_plan = u_plan, dfun = dfun, t0 = 0, t1 = 30,
                                obs_times = 1:30, outputs = "conc_obs", dt = 0.5)
  FIM <- fisher_information(sens$S, sd = 0.1)
  expect_equal(FIM, t(FIM))
  expect_true(all(eigen(FIM, symmetric = TRUE)$values > -1e-8))
  isp <- information_spectrum(FIM, prior_sd = model$phi_prior_sd)
  expect_true(all(is.finite(isp$bounds$crb_sd)))
  expect_true(all(isp$bounds$bcrb_sd <= isp$bounds$prior_sd))
})

test_that("expected information gain grows with a probing input", {
  model <- load_model(toy_cfg(), variant = "fitted")
  dfun <- const_dfun(inflow = 0.3)
  flat <- make_u_plan(times = 0, U = 1, control_names = "u_in")
  probe <- make_u_plan(times = c(0, 2, 4, 6, 8), U = c(0, 3, 0, 3, 0), control_names = "u_in")
  Sigma <- diag(model$phi_prior_sd^2)
  ig <- function(plan) expected_info_gain(model, x0 = c(conc = 5), parms = model$parms_nominal,
                                          phi_cov = Sigma, u_plan = plan, dfun = dfun, t0 = 0,
                                          t1 = 10, sensor_set = "routine", dt = 0.5)
  expect_gt(ig(probe), ig(flat))
  expect_gte(ig(flat), 0)
})

test_that("identifiability report runs and classifies every parameter", {
  cfg <- toy_cfg()
  rep <- identifiability_report(cfg, seed = 1)
  expect_setequal(rep$classification$param, c("k_el", "b_in"))
  expect_true(all(c("disturbances", "probing_routine", "probing") %in% names(rep$fim)))
  expect_true(all(rep$bounds$ratio <= 1 + 1e-9))
})
