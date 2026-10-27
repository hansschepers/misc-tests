# =============================================================================
# identifiability.R
# Output sensitivities to the (transformed) parameters along a trajectory,
# Fisher information, eigen-spectrum, Cramer-Rao bounds, expected information
# gain of a plan, a report of which parameters the disturbances alone
# identify versus which need probing, and profile likelihoods.
# =============================================================================

#' Output sensitivities to the estimated parameters
#'
#' `fd_central`/`fd_forward`: finite differences of whole simulations.
#' `forward_ode`: forward sensitivity equations dS/dt = f_x S + f_phi
#' integrated with RK4 alongside the states (Jacobians of f by central
#' differences at every stage, event jumps S+ = g_x S + g_phi), output
#' sensitivities h_x S + h_phi at the observation times.
#'
#' @param model a `dc_model`.
#' @param x0 initial state.
#' @param parms natural parameters (the estimated ones are perturbed in the
#'   transformed space).
#' @param u_plan control plan table.
#' @param dfun disturbance function.
#' @param t0,t1 window.
#' @param obs_times observation times (sorted, within the window).
#' @param outputs output names.
#' @param events_plan NULL or data.table(time, event).
#' @param method `fd_central`, `fd_forward` or `forward_ode`.
#' @param h step in the transformed parameter space.
#' @param dt integration step.
#' @param engine solver engine for the finite-difference methods.
#' @return list(times, y (n_t x n_o), S (array n_t x n_o x n_p)).
#' @export
compute_sensitivities <- function(model, x0, parms, u_plan, dfun, t0, t1, obs_times, outputs,
                                  events_plan = NULL, method = "fd_central", h = 1e-4,
                                  dt = 1, engine = "rk4_internal") {
  phi0 <- parms_to_phi(model, parms = parms)
  n_p <- model$n_p
  if (method == "forward_ode") {
    return(forward_sensitivities(model, x0 = x0, parms = parms, u_plan = u_plan, dfun = dfun,
                                 t0 = t0, t1 = t1, obs_times = obs_times, outputs = outputs,
                                 events_plan = events_plan, dt = dt, h_jac = h))
  }
  sim_y <- function(phi) {
    p <- phi_to_parms(model, phi = phi, base_parms = parms)
    res <- simulate_unit(model, x0 = x0, parms = p, u_plan = u_plan, dfun = dfun, t0 = t0, t1 = t1,
                         dt = dt, events_plan = events_plan, out_times = obs_times, engine = engine)
    list(times = res$y_out$time, y = as.matrix(res$y_out[, outputs, with = FALSE]))
  }
  base <- sim_y(phi0)
  y0 <- base$y
  S <- array(0, dim = c(nrow(y0), length(outputs), n_p),
             dimnames = list(NULL, outputs, model$phi_names))
  j <- 1                                              # debugging line
  for (j in seq_len(n_p)) {
    phi_p <- phi0; phi_p[j] <- phi_p[j] + h
    if (method == "fd_central") {
      phi_m <- phi0; phi_m[j] <- phi_m[j] - h
      S[, , j] <- (sim_y(phi_p)$y - sim_y(phi_m)$y) / (2 * h)
    } else if (method == "fd_forward") {
      S[, , j] <- (sim_y(phi_p)$y - y0) / h
    } else {
      stop("unknown sensitivity method: ", method)
    }
  }
  list(times = base$times, y = y0, S = S)
}

# forward sensitivity equations (see compute_sensitivities)
forward_sensitivities <- function(model, x0, parms, u_plan, dfun, t0, t1, obs_times, outputs,
                                  events_plan, dt, h_jac) {
  n_x <- model$n_x; n_p <- model$n_p
  phi0 <- parms_to_phi(model, parms = parms)
  p_of <- function(phi) phi_to_parms(model, phi = phi, base_parms = parms)
  steps_x <- function(x) pmax(1e-7, h_jac * abs(x))
  steps <- function(x) c(steps_x(x), rep(h_jac, n_p))
  cn <- model$control_names
  sn <- model$state_names

  breaks <- c(t0, u_plan$time[u_plan$time > t0 + 1e-9 & u_plan$time < t1 - 1e-9])
  if (!is.null(events_plan) && nrow(events_plan) > 0) {
    et <- events_plan$time
    breaks <- c(breaks, et[et > t0 + 1e-9 & et < t1 - 1e-9])
  }
  breaks <- sort(unique(round(c(breaks, t1), 9)))
  obs_times <- obs_times[obs_times >= t0 - 1e-9 & obs_times <= t1 + 1e-9]

  x <- as.numeric(x0)
  Sx <- matrix(0, nrow = n_x, ncol = n_p)
  out_t <- c(); out_y <- list(); out_S <- list()

  s <- 1                                              # debugging line
  for (s in seq_len(length(breaks) - 1L)) {
    ta <- breaks[s]; tb <- breaks[s + 1L]
    u <- plan_controls(u_plan, time = ta, control_names = cn)
    if (!is.null(events_plan) && nrow(events_plan) > 0) {
      hit <- which(abs(events_plan$time - ta) < 1e-9)
      k <- hit[1]                                     # debugging line
      for (k in hit) {
        en <- events_plan$event[k]
        g <- function(v) as.numeric(model$events[[en]]$apply(stats::setNames(v[seq_len(n_x)], sn),
                                                            p_of(v[n_x + seq_len(n_p)])))
        jac <- fd_jacobian(g, v0 = c(x, phi0), steps = steps(x), central = TRUE)
        Sx <- jac$J[, seq_len(n_x), drop = FALSE] %*% Sx + jac$J[, n_x + seq_len(n_p), drop = FALSE]
        x <- as.numeric(clamp_states(jac$f0, model))
      }
    }
    f_aug <- function(time, w) {
      xs <- w[seq_len(n_x)]
      S_cur <- matrix(w[-seq_len(n_x)], nrow = n_x, ncol = n_p)
      d <- dfun(time)
      fv <- function(v) as.numeric(model$rhs(time, stats::setNames(v[seq_len(n_x)], sn), u, d,
                                             p_of(v[n_x + seq_len(n_p)])))
      jac <- fd_jacobian(fv, v0 = c(xs, phi0), steps = steps(xs), central = TRUE)
      dS <- jac$J[, seq_len(n_x), drop = FALSE] %*% S_cur + jac$J[, n_x + seq_len(n_p), drop = FALSE]
      c(jac$f0, as.numeric(dS))
    }
    grid <- make_grid(ta, tb, dt, extra = obs_times)
    W <- rk4_grid(f_aug, y0 = c(x, as.numeric(Sx)), grid = grid)
    sel <- which(vapply(grid, function(gt) any(abs(obs_times - gt) < 1e-9), logical(1)) &
                   (grid > ta + 1e-9 | (s == 1L & grid <= t0 + 1e-9)))
    r <- sel[1]                                       # debugging line
    for (r in sel) {
      xr <- W[r, seq_len(n_x)]
      Sr <- matrix(W[r, -seq_len(n_x)], nrow = n_x, ncol = n_p)
      d <- dfun(grid[r])
      hv <- function(v) as.numeric(model$outputs(grid[r], stats::setNames(v[seq_len(n_x)], sn), u, d,
                                                  p_of(v[n_x + seq_len(n_p)]))[outputs])
      jh <- fd_jacobian(hv, v0 = c(xr, phi0), steps = steps(xr), central = TRUE)
      out_t <- c(out_t, grid[r])
      out_y[[length(out_y) + 1L]] <- hv(c(xr, phi0))
      out_S[[length(out_S) + 1L]] <- jh$J[, seq_len(n_x), drop = FALSE] %*% Sr +
        jh$J[, n_x + seq_len(n_p), drop = FALSE]
    }
    x <- as.numeric(clamp_states(W[nrow(W), seq_len(n_x)], model))
    Sx <- matrix(W[nrow(W), -seq_len(n_x)], nrow = n_x, ncol = n_p)
  }
  n_t <- length(out_t)
  y <- matrix(as.numeric(unlist(out_y)), nrow = n_t, ncol = length(outputs), byrow = TRUE,
              dimnames = list(NULL, outputs))
  S <- array(0, dim = c(n_t, length(outputs), n_p), dimnames = list(NULL, outputs, model$phi_names))
  k <- 1                                              # debugging line
  for (k in seq_len(n_t)) S[k, , ] <- out_S[[k]]
  list(times = out_t, y = y, S = S)
}

#' Fisher information from output sensitivities
#'
#' FIM = sum_k S_k' R^-1 S_k with independent Gaussian errors of sd `sd`.
#'
#' @param S array n_t x n_o x n_p.
#' @param sd observation sd per output (length n_o).
#' @export
fisher_information <- function(S, sd) {
  n_o <- dim(S)[2]; n_p <- dim(S)[3]
  FIM <- matrix(0, nrow = n_p, ncol = n_p, dimnames = list(dimnames(S)[[3]], dimnames(S)[[3]]))
  k <- 1                                              # debugging line
  for (k in seq_len(dim(S)[1])) {
    Sk <- matrix(S[k, , ], nrow = n_o, ncol = n_p)
    FIM <- FIM + t(Sk) %*% (Sk / sd^2)
  }
  (FIM + t(FIM)) / 2
}

#' Eigen-spectrum and Cramer-Rao bounds of a Fisher information matrix
#'
#' @param FIM Fisher information (transformed parameter space).
#' @param prior_sd prior sd per parameter (for the Bayesian bound).
#' @return list(spectrum = data.table(k, eigenvalue, param, loading),
#'   bounds = data.table(param, prior_sd, crb_sd, bcrb_sd, ratio)).
#'   `crb_sd` is the classical bound (Inf where the FIM is singular),
#'   `bcrb_sd` the posterior-sd bound with the prior, `ratio` = bcrb/prior.
#' @export
information_spectrum <- function(FIM, prior_sd) {
  pn <- rownames(FIM)
  ev <- eigen(FIM, symmetric = TRUE)
  spec <- data.table::rbindlist(lapply(seq_along(ev$values), function(k) {
    data.table::data.table(k = k, eigenvalue = ev$values[k], param = pn, loading = ev$vectors[, k])
  }))
  lam_max <- max(abs(ev$values), 1e-300)
  ok <- ev$values > 1e-10 * lam_max
  crb <- rep(Inf, length(pn))
  if (any(ok)) {
    # pseudo-inverse on the identifiable subspace; parameters loading on a
    # null direction have an infinite classical bound
    V <- ev$vectors[, ok, drop = FALSE]
    pinv_diag <- rowSums(V^2 / matrix(ev$values[ok], nrow = nrow(V), ncol = sum(ok), byrow = TRUE))
    null_load <- if (any(!ok)) rowSums(ev$vectors[, !ok, drop = FALSE]^2) else rep(0, length(pn))
    crb <- ifelse(null_load > 1e-6, Inf, sqrt(pinv_diag))
  }
  post <- solve(FIM + diag(1 / prior_sd^2, nrow = length(pn)))
  bcrb <- sqrt(diag(post))
  bounds <- data.table::data.table(param = pn, prior_sd = as.numeric(prior_sd), crb_sd = crb,
                                   bcrb_sd = as.numeric(bcrb), ratio = as.numeric(bcrb / prior_sd))
  list(spectrum = spec, bounds = bounds)
}

#' Expected information gain of a plan
#'
#' 0.5 * log det(I + L FIM L') with the current posterior covariance
#' Sigma = L'L of the unit's parameters and FIM predicted along the planned
#' trajectory for the unit's sensor set (forward finite differences). This is
#' the expected log-det reduction of the parameter covariance (in nats).
#'
#' @param model a `dc_model`.
#' @param x0 initial state.
#' @param parms natural parameters (posterior mean).
#' @param phi_cov posterior covariance of the transformed parameters.
#' @param u_plan control plan.
#' @param dfun disturbance (forecast) function.
#' @param t0,t1 information window.
#' @param sensor_set sensor set name.
#' @param events_plan optional events.
#' @param noise_mult sensor noise multiplier.
#' @param dt integration step.
#' @param engine solver engine.
#' @param h FD step.
#' @export
expected_info_gain <- function(model, x0, parms, phi_cov, u_plan, dfun, t0, t1, sensor_set,
                               events_plan = NULL, noise_mult = 1, dt = 1,
                               engine = "rk4_internal", h = 1e-4) {
  ss <- model$sensor_sets[[sensor_set]]
  if (t0 + ss$interval > t1 + 1e-9) return(0)
  obs <- seq(t0 + ss$interval, t1 + 1e-9, by = ss$interval)
  sens <- compute_sensitivities(model, x0 = x0, parms = parms, u_plan = u_plan, dfun = dfun,
                                t0 = t0, t1 = t1, obs_times = obs, outputs = ss$outputs,
                                events_plan = events_plan, method = "fd_forward", h = h,
                                dt = dt, engine = engine)
  FIM <- fisher_information(sens$S, sd = ss$sd[ss$outputs] * noise_mult)
  L <- chol(make_pd(phi_cov))
  0.5 * logdet(diag(model$n_p) + L %*% FIM %*% t(L))
}

#' Which parameters do the disturbances identify, which need probing?
#'
#' Computes the Fisher information over one season for a single unit at the
#' prior mean in three scenarios: `disturbances` (calendar recipe, routine
#' sensors: only weather/load and scheduled events excite the system),
#' `probing_routine` (PRBS dither on every control, routine sensors) and
#' `probing` (dither plus the intensive probe sensors). A parameter counts as
#' identified when its posterior-sd bound is below
#' `cfg$identifiability$threshold` times its prior sd.
#'
#' @param cfg scenario config.
#' @param model fitted model (loaded if NULL).
#' @param seed weather seed.
#' @param parms parameters to linearise at (default prior mean).
#' @return list(fim, spectrum, bounds, classification).
#' @export
identifiability_report <- function(cfg, model = NULL, seed = 1L, parms = NULL) {
  if (is.null(model)) model <- load_model(cfg, variant = "fitted")
  if (is.null(parms)) parms <- model$parms_nominal
  E <- cfg$season$epoch_days
  n_ep <- floor(cfg$season$length_days / E)
  T_end <- n_ep * E
  times <- seq(0, T_end + E, by = cfg$disturbances$dt)
  dist <- get_disturbances(model, cfg = cfg, times = times, seed = derive_seed(seed, "weather"))
  dfun <- make_interp_fun(dist, cols = model$dist_names)
  cal <- calendar_schedule(model, cfg = cfg, n_epochs = n_ep)
  ev_plan <- if (nrow(cal$events) > 0) data.table::data.table(time = cal$events$epoch * E, event = cal$events$event) else NULL
  base_plan <- make_u_plan(times = 0, U = cal$controls, control_names = model$control_names)
  ic <- cfg$identifiability
  dith <- dither_plan(model, base_u = cal$controls, n_epochs = n_ep, epoch_days = E,
                      amplitude = ic$dither_amplitude, hold = ic$dither_hold_epochs,
                      seed = derive_seed(seed, "ident"))
  sets <- cfg$experiment$sensor_sets
  scen <- list(disturbances    = list(plan = base_plan, sensor = sets$production),
               probing_routine = list(plan = dith, sensor = sets$production),
               probing         = list(plan = dith, sensor = sets$probe))
  x0 <- stats::setNames(model$states$init, model$state_names)
  nm <- cfg$sensors$noise_mult %||% 1
  fims <- list(); specs <- list(); bnds <- list()
  sc <- names(scen)[1]                                # debugging line
  for (sc in names(scen)) {
    ss <- model$sensor_sets[[scen[[sc]]$sensor]]
    obs <- seq(ss$interval, T_end, by = ss$interval)
    sens <- compute_sensitivities(model, x0 = x0, parms = parms, u_plan = scen[[sc]]$plan, dfun = dfun,
                                  t0 = 0, t1 = T_end, obs_times = obs, outputs = ss$outputs,
                                  events_plan = ev_plan, method = "fd_central", h = ic$fd_step,
                                  dt = cfg$solver$dt_inner, engine = cfg$solver$inner_engine)
    FIM <- fisher_information(sens$S, sd = ss$sd[ss$outputs] * nm)
    isp <- information_spectrum(FIM, prior_sd = model$phi_prior_sd)
    fims[[sc]] <- FIM
    specs[[sc]] <- isp$spectrum[, scenario := sc]
    bnds[[sc]] <- isp$bounds[, scenario := sc]
  }
  bounds <- data.table::rbindlist(bnds)
  thr <- ic$threshold
  wide <- data.table::dcast(bounds, param ~ scenario, value.var = "ratio")
  wide[, identified_by_disturbances := disturbances < thr]
  wide[, identified_with_probing := probing < thr]
  wide[, verdict := ifelse(identified_by_disturbances, "identified by disturbances alone",
                           ifelse(identified_with_probing, "needs probing (and/or probe sensors)",
                                  "weakly identified even with probing"))]
  list(fim = fims, spectrum = data.table::rbindlist(specs), bounds = bounds, classification = wide)
}

#' Profile likelihood of one parameter
#'
#' For each grid value of `param` (transformed space) the other estimated
#' parameters are re-optimised (Nelder-Mead) to minimise the Gaussian negative
#' log-likelihood of `data` (columns `time` and the sensor outputs).
#'
#' @param model a `dc_model`.
#' @param data data.table of observations.
#' @param x0 initial state.
#' @param u_plan control plan.
#' @param dfun disturbance function.
#' @param t0,t1 window.
#' @param sensor_set sensor set of the data.
#' @param param parameter name.
#' @param phi_start starting transformed parameters.
#' @param cfg scenario config (`cfg$profile`).
#' @param events_plan optional events.
#' @return data.table(param, value, nll, delta_nll).
#' @export
profile_likelihood <- function(model, data, x0, u_plan, dfun, t0, t1, sensor_set, param,
                               phi_start = model$phi_prior_mean, cfg, events_plan = NULL) {
  ss <- model$sensor_sets[[sensor_set]]
  sd <- ss$sd[ss$outputs] * (cfg$sensors$noise_mult %||% 1)
  Y <- as.matrix(data[, ss$outputs, with = FALSE])
  nll <- function(phi) {
    p <- phi_to_parms(model, phi = phi)
    res <- simulate_unit(model, x0 = x0, parms = p, u_plan = u_plan, dfun = dfun, t0 = t0, t1 = t1,
                         dt = cfg$solver$dt_inner, events_plan = events_plan, out_times = data$time,
                         engine = cfg$solver$inner_engine)
    Yh <- as.matrix(res$y_out[, ss$outputs, with = FALSE])
    0.5 * sum(sweep(Y - Yh, 2, sd, "/")^2)
  }
  j <- match(param, model$phi_names)
  pc <- cfg$profile
  grid <- phi_start[j] + seq(-pc$width_sd, pc$width_sd, length.out = pc$n_grid) * model$phi_prior_sd[j]
  others <- setdiff(seq_len(model$n_p), j)
  vals <- vapply(grid, function(g) {
    f <- function(q) { phi <- phi_start; phi[j] <- g; phi[others] <- q; nll(phi) }
    if (length(others) == 0) return(f(numeric(0)))
    if (length(others) == 1) {
      w <- 3 * model$phi_prior_sd[others]
      return(stats::optimize(f, interval = phi_start[others] + c(-w, w))$objective)
    }
    stats::optim(phi_start[others], f, method = "Nelder-Mead", control = list(maxit = pc$maxit))$value
  }, numeric(1))
  data.table::data.table(param = param, value = grid, nll = vals, delta_nll = vals - min(vals))
}
