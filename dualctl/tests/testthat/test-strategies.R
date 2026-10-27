# a valid strategy context on the toy model
toy_context <- function(cfg, sensor_set = "intensive") {
  fit <- load_model(cfg, variant = "fitted")
  tru <- load_model(cfg, variant = "truth")
  plant <- Plant$new(tru, cfg = cfg, seed = 1, n_units = 1)
  est <- EstimatorEKF$new(fit, cfg = cfg, n_units = 1, seed = 1)
  po <- est$posterior(1)
  ctl <- Controller$new(fit, cfg = cfg)
  fc <- forecast_disturbances(fit, realized = plant$dist, t_now = 0, t_end = ctl$H * ctl$E, cfg = cfg)
  state <- list(unit = 1, epoch = 0, n_epochs = 8, season_frac = 0, x_mean = po$x_mean,
                x_cov = po$x_cov, u_prev = fit$control_default,
                last_event_epoch = stats::setNames(rep(-Inf, length(fit$event_names)), fit$event_names),
                last_cost_rate = NA_real_, last_obs = c(conc_obs = 2), sensor_set = sensor_set,
                arm = "probe", seed = 11,
                truth = list(x = plant$x[[1]], parms = plant$unit_parms[[1]]))
  list(fit = fit, tru = tru, posterior = po, forecast = fc, state = state)
}

test_that("every registered strategy conforms to the act() interface", {
  cfg <- toy_cfg()
  ctx <- toy_context(cfg)
  fit <- ctx$fit
  nm <- list_strategies()[1]                                 # debugging line
  for (nm in list_strategies()) {
    s <- make_strategy(nm, model = fit, cfg = cfg, unit = 1, seed = 1, model_truth = ctx$tru)
    expect_true(inherits(s, "ProbingStrategy"), info = nm)
    res <- s$act(ctx$state, ctx$posterior, ctx$forecast, 0)
    expect_named(res, c("u", "events", "info"), info = nm)
    expect_equal(names(res$u), fit$control_names, info = nm)
    expect_true(all(is.finite(res$u)), info = nm)
    expect_true(is.character(res$events), info = nm)
    expect_true(all(res$events %in% fit$event_names), info = nm)
    # the engine clamps to bounds and rate limits; the result must be usable
    u_ok <- dualctl:::enforce_control_limits(res$u, u_prev = ctx$state$u_prev, model = fit)
    expect_true(all(u_ok >= fit$controls$lower & u_ok <= fit$controls$upper), info = nm)
  }
})

test_that("required strategies are registered", {
  expect_true(all(c("none", "ce", "calendar", "dither", "infompc", "thompson", "evsi",
                    "hillclimb", "oracle") %in% list_strategies()))
})

test_that("controller plans respect bounds and rate limits", {
  cfg <- toy_cfg()
  ctx <- toy_context(cfg)
  ctl <- Controller$new(ctx$fit, cfg = cfg)
  pl <- ctl$plan(x0 = ctx$state$x_mean, parms = ctx$fit$parms_nominal, u_prev = c(u_in = 1),
                 forecast_dt = ctx$forecast, t0 = 0, epoch = 0, seed = 3)
  expect_equal(dim(pl$U), c(ctl$H, 1L))
  expect_true(all(pl$U >= 0 & pl$U <= 3))
  expect_true(all(abs(diff(c(1, pl$U[, 1]))) <= 1 + 1e-9))
  expect_true(is.finite(pl$J))
  # the plan should not be worse than holding the current recipe
  hold <- ctl$evaluate(ctl$hold_dec(c(u_in = 1)), ev = NULL, x0 = ctx$state$x_mean,
                       parms = ctx$fit$parms_nominal, u_prev = c(u_in = 1),
                       dfun = make_interp_fun(ctx$forecast, cols = "inflow"), t0 = 0)$J
  expect_lte(pl$J, hold + 1e-9)
})

test_that("advisor returns a ranked list and a sequence", {
  cfg <- toy_cfg()
  ctx <- toy_context(cfg)
  adv <- advise(ctx$fit, cfg = cfg, posterior = ctx$posterior, x_mean = ctx$state$x_mean,
                u_prev = c(u_in = 1), forecast_dt = ctx$forecast, t0 = 0, epoch = 0,
                last_event_epoch = ctx$state$last_event_epoch, seed = 1)
  expect_true(data.table::is.data.table(adv$ranking))
  expect_true(all(c("control", "step", "saving_mean", "score", "rank") %in% names(adv$ranking)))
  expect_equal(adv$ranking$rank, seq_len(nrow(adv$ranking)))
  expect_true(all(diff(adv$ranking$score) <= 1e-12))
})
