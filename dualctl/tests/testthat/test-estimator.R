# run a single-unit filter on noise-free toy data with an exciting input
run_toy_filter <- function(est, model, parms_true, n_days, sensor_set = "routine") {
  dfun <- const_dfun(inflow = 0.3)                           # [mg/L/d]
  x <- c(conc = 2)                                           # [mg/L] equals the prior initial state
  t_prev <- 0
  day <- 1                                                   # debugging line
  for (day in seq_len(n_days)) {
    u <- c(u_in = if ((day %/% 5) %% 2 == 0) 0.5 else 2.5)   # [mg/L/d] square wave
    plan <- make_u_plan(times = t_prev, U = u, control_names = "u_in")
    sim <- simulate_unit(model, x0 = x, parms = parms_true, u_plan = plan, dfun = dfun,
                         t0 = t_prev, t1 = day, dt = 0.1, out_times = day)
    x <- sim$x_end
    est$predict(t_prev, day, u_list = list(u), dfun = dfun)
    y <- c(conc_obs = sim$y_out$conc_obs[1])
    est$update(day, y_list = list(y), sensor_sets = sensor_set, u_list = list(u), dfun = dfun)
    t_prev <- day
  }
  est
}

test_that("EKF recovers parameters in a noise-free case", {
  cfg <- toy_cfg(list(sensors = list(noise_mult = 0.2)))
  model <- load_model(cfg, variant = "fitted")
  phi_true <- c(k_el = log(0.3), b_in = log(1.3))            # ~1 prior sd from the prior mean
  parms_true <- phi_to_parms(model, phi = phi_true)
  est <- EstimatorEKF$new(model, cfg = cfg, n_units = 1, seed = 1)
  prior_sd <- est$posterior(1)$sd
  est <- run_toy_filter(est, model = model, parms_true = parms_true, n_days = 60)
  po <- est$posterior(1)
  expect_lt(abs(po$mean[["k_el"]] - phi_true[["k_el"]]), 0.05)
  expect_lt(abs(po$mean[["b_in"]] - phi_true[["b_in"]]), 0.05)
  expect_true(all(po$sd < prior_sd))
})

test_that("EnKF moves towards the truth (robustness check)", {
  cfg <- toy_cfg(list(sensors = list(noise_mult = 0.2), estimator = list(n_ensemble = 30)))
  model <- load_model(cfg, variant = "fitted")
  phi_true <- c(k_el = log(0.3), b_in = log(1.3))
  parms_true <- phi_to_parms(model, phi = phi_true)
  est <- EstimatorEnKF$new(model, cfg = cfg, n_units = 1, seed = 2)
  err0 <- sum(abs(est$posterior(1)$mean - phi_true))
  est <- run_toy_filter(est, model = model, parms_true = parms_true, n_days = 30)
  err1 <- sum(abs(est$posterior(1)$mean - phi_true))
  expect_lt(err1, err0)
  expect_lt(err1, 0.3)
})

test_that("hierarchical model transfers information between units", {
  cfg <- toy_cfg(list(sensors = list(noise_mult = 0.2)))
  model <- load_model(cfg, variant = "fitted")
  est <- EstimatorEKF$new(model, cfg = cfg, n_units = 2, seed = 1)
  ld0 <- logdet(est$posterior(2)$cov)
  dfun <- const_dfun(inflow = 0.3)
  u_list <- list(c(u_in = 2), c(u_in = 1))
  day <- 1                                                   # debugging line
  for (day in 1:10) {
    est$predict(day - 1, day, u_list = u_list, dfun = dfun)
    # only unit 1 is observed
    est$update(day, y_list = list(c(conc_obs = 2 + 0.5 * day), NULL),
               sensor_sets = c("routine", "routine"), u_list = u_list, dfun = dfun)
  }
  expect_lt(logdet(est$posterior(2)$cov), ld0)
})

test_that("state tracking without parameter learning keeps parameters at the prior", {
  cfg <- toy_cfg(list(sensors = list(noise_mult = 0.2)))
  model <- load_model(cfg, variant = "fitted")
  phi_true <- c(k_el = log(0.3), b_in = log(1.3))
  est <- EstimatorEKF$new(model, cfg = cfg, n_units = 1, seed = 1, learn_params = FALSE)
  est <- run_toy_filter(est, model = model, parms_true = phi_to_parms(model, phi = phi_true), n_days = 10)
  expect_equal(est$posterior(NULL)$mean, model$phi_prior_mean, tolerance = 1e-12)
})
