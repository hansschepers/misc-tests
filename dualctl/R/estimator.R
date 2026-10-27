# =============================================================================
# estimator.R
# Hierarchical joint state/parameter estimation across all units.
#
# Augmented state z = [x_1, ..., x_N, phi_s, b_1, ..., b_N]
#   x_i   states of unit i (natural units)
#   phi_s shared (population) parameters, transformed space
#   b_i   random effects of unit i for the parameters with unit_sd > 0
# Unit i's effective parameters: phi_i = phi_s + M b_i  (M selects the
# random-effect parameters). Prior: phi_s ~ N(prior mean, prior_sd^2),
# b_i ~ N(0, unit_sd^2). All units are fitted in one model, so what the probe
# units teach about phi_s transfers to the production units.
#
# Interface (all estimators):
#   init(), predict(t0, t1, u_list, dfun), update(time, y_list, sensor_sets,
#   u_list, dfun), apply_event(unit, event, time, u, dfun),
#   posterior(unit), sample(n, unit), cov()
# =============================================================================

# index layout of the augmented state
make_layout <- function(model, n_units) {
  pt <- model$params[match(model$phi_names, model$params$name), ]
  re_names <- model$phi_names[pt$unit_sd > 0]
  n_x <- model$n_x; n_p <- model$n_p; n_re <- length(re_names)
  idx_x <- lapply(seq_len(n_units), function(i) (i - 1L) * n_x + seq_len(n_x))
  off_p <- n_units * n_x
  idx_p <- off_p + seq_len(n_p)
  off_b <- off_p + n_p
  idx_b <- lapply(seq_len(n_units), function(i) off_b + (i - 1L) * n_re + seq_len(n_re))
  M <- matrix(0, nrow = n_p, ncol = n_re)
  if (n_re > 0) M[cbind(match(re_names, model$phi_names), seq_len(n_re))] <- 1
  list(n_units = n_units, n_x = n_x, n_p = n_p, n_re = n_re, n_z = off_b + n_units * n_re,
       idx_x = idx_x, idx_p = idx_p, idx_b = idx_b, M = M, re_names = re_names,
       re_sd = pt$unit_sd[pt$unit_sd > 0])
}

#' Estimator base class
#'
#' @description Common fields and helpers of the hierarchical estimators.
#' @export
Estimator <- R6::R6Class("Estimator",
  public = list(
    #' @field model fitted `dc_model`.
    model = NULL,
    #' @field cfg scenario config.
    cfg = NULL,
    #' @field layout index layout of the augmented state.
    layout = NULL,
    #' @field learn_params if FALSE parameters stay at the prior (state tracking only).
    learn_params = TRUE,
    #' @field noise_mult sensor noise multiplier assumed by the estimator.
    noise_mult = 1,
    #' @field seed seed for stochastic estimators.
    seed = 1L,

    #' @description Create an estimator.
    #' @param model fitted `dc_model`.
    #' @param cfg scenario config.
    #' @param n_units number of units.
    #' @param seed seed.
    #' @param learn_params learn parameters (TRUE) or track states only.
    initialize = function(model, cfg, n_units, seed = 1L, learn_params = TRUE) {
      self$model <- model
      self$cfg <- cfg
      self$layout <- make_layout(model, n_units = n_units)
      self$learn_params <- learn_params
      self$noise_mult <- cfg$sensors$noise_mult %||% 1
      self$seed <- seed
      self$init()
    },

    #' @description Prior mean and covariance of the augmented state.
    prior = function() {
      L <- self$layout
      m <- self$model
      z <- numeric(L$n_z)
      sd <- numeric(L$n_z)
      i <- 1                                          # debugging line
      for (i in seq_len(L$n_units)) {
        z[L$idx_x[[i]]]  <- m$states$init
        sd[L$idx_x[[i]]] <- m$states$init_sd
        sd[L$idx_b[[i]]] <- L$re_sd
      }
      z[L$idx_p]  <- m$phi_prior_mean
      sd[L$idx_p] <- m$phi_prior_sd
      sd[sd <= 0] <- self$cfg$estimator$fd_step_state_abs
      list(z = z, P = diag(sd^2, nrow = L$n_z))
    },

    #' @description Initialise (abstract).
    init = function() stop("abstract"),

    #' @description Effective parameters of a unit from an augmented vector.
    #' @param z augmented state vector.
    #' @param unit unit index.
    unit_phi = function(z, unit) {
      L <- self$layout
      stats::setNames(z[L$idx_p] + as.numeric(L$M %*% z[L$idx_b[[unit]]]), self$model$phi_names)
    },

    #' @description Linear map from z to the effective parameters of a unit.
    #' @param unit unit index.
    unit_map = function(unit) {
      L <- self$layout
      Tm <- matrix(0, nrow = L$n_p, ncol = L$n_z)
      Tm[, L$idx_p] <- diag(L$n_p)
      if (L$n_re > 0) Tm[, L$idx_b[[unit]]] <- L$M
      Tm
    },

    #' @description One-step state transition of a unit (x, phi) -> x.
    #' @param v c(x, phi) of one unit.
    #' @param u controls.
    #' @param dfun disturbance function.
    #' @param t0,t1 window.
    transition = function(v, u, dfun, t0, t1) {
      L <- self$layout
      x <- v[seq_len(L$n_x)]
      phi <- v[L$n_x + seq_len(L$n_p)]
      parms <- phi_to_parms(self$model, phi = phi)
      grid <- make_grid(t0, t1, dt = self$cfg$solver$dt_inner)
      Y <- integrate_states(self$model, x0 = x, parms = parms, u = u, dfun = dfun,
                            grid = grid, engine = self$cfg$solver$inner_engine)
      as.numeric(Y[nrow(Y), ])
    },

    #' @description Measurement function of a unit (x, phi) -> outputs.
    #' @param v c(x, phi).
    #' @param time time.
    #' @param u controls.
    #' @param d disturbances at `time`.
    #' @param onames output names.
    measure = function(v, time, u, d, onames) {
      L <- self$layout
      x <- stats::setNames(v[seq_len(L$n_x)], self$model$state_names)
      parms <- phi_to_parms(self$model, phi = v[L$n_x + seq_len(L$n_p)])
      as.numeric(self$model$outputs(time, x, u, d, parms)[onames])
    },

    #' @description Event map of a unit (x, phi) -> x.
    #' @param v c(x, phi).
    #' @param event event name.
    event_map = function(v, event) {
      L <- self$layout
      x <- stats::setNames(v[seq_len(L$n_x)], self$model$state_names)
      parms <- phi_to_parms(self$model, phi = v[L$n_x + seq_len(L$n_p)])
      as.numeric(self$model$events[[event]]$apply(x, parms))
    },

    #' @description Finite-difference step sizes for c(x, phi) of a unit.
    #' @param v c(x, phi).
    fd_steps = function(v) {
      L <- self$layout
      ec <- self$cfg$estimator
      xs <- v[seq_len(L$n_x)]
      c(pmax(ec$fd_step_state_abs, ec$fd_step_state_rel * abs(xs)), rep(ec$fd_step_phi, L$n_p))
    },

    #' @description Assumed observation sd for a sensor set.
    #' @param sensor_set name.
    obs_sd = function(sensor_set) {
      ss <- self$model$sensor_sets[[sensor_set]]
      pmax(ss$sd[ss$outputs] * self$noise_mult, self$cfg$estimator$min_obs_sd)
    },

    #' @description Clamp parameters and random effects to sane ranges.
    #' @param z augmented vector.
    clamp_z = function(z) {
      L <- self$layout
      m <- self$model
      k <- self$cfg$estimator$phi_clamp_sd
      z[L$idx_p] <- clamp(z[L$idx_p], m$phi_prior_mean - k * m$phi_prior_sd,
                          m$phi_prior_mean + k * m$phi_prior_sd)
      i <- 1                                          # debugging line
      for (i in seq_len(L$n_units)) {
        z[L$idx_x[[i]]] <- clamp(z[L$idx_x[[i]]], m$states$lower, m$states$upper)
        if (L$n_re > 0) z[L$idx_b[[i]]] <- clamp(z[L$idx_b[[i]]], -k * L$re_sd, k * L$re_sd)
      }
      z
    },

    #' @description Indices of all parameter entries of z.
    param_idx = function() {
      L <- self$layout
      c(L$idx_p, unlist(L$idx_b))
    },

    #' @description Process-noise covariance for a step of length `dt`.
    #' @param dt step length.
    process_noise = function(dt) {
      L <- self$layout
      q <- numeric(L$n_z)
      i <- 1                                          # debugging line
      for (i in seq_len(L$n_units)) q[L$idx_x[[i]]] <- self$model$states$process_sd^2 * dt
      if (self$learn_params) q[self$param_idx()] <- (self$cfg$estimator$param_process_sd %||% 0)^2 * dt
      diag(q, nrow = L$n_z)
    },

    #' @description Augmented-state mean (abstract).
    mean_z = function() stop("abstract"),

    #' @description Augmented-state covariance (abstract).
    cov = function() stop("abstract"),

    #' @description Posterior of a unit's effective parameters (or the shared
    #'   parameters if `unit` is NULL) and of its states.
    #' @param unit unit index or NULL.
    #' @return list(mean, cov, sd, x_mean, x_cov, sample).
    posterior = function(unit = NULL) {
      z <- self$mean_z()
      P <- self$cov()
      L <- self$layout
      if (is.null(unit)) {
        mu <- stats::setNames(z[L$idx_p], self$model$phi_names)
        S <- P[L$idx_p, L$idx_p, drop = FALSE]
        x_mean <- NULL; x_cov <- NULL
      } else {
        Tm <- self$unit_map(unit)
        mu <- stats::setNames(as.numeric(Tm %*% z), self$model$phi_names)
        S <- Tm %*% P %*% t(Tm)
        ix <- L$idx_x[[unit]]
        x_mean <- stats::setNames(z[ix], self$model$state_names)
        x_cov <- P[ix, ix, drop = FALSE]
      }
      S <- make_pd(S)
      dimnames(S) <- list(self$model$phi_names, self$model$phi_names)
      list(mean = mu, cov = S, sd = sqrt(diag(S)), x_mean = x_mean, x_cov = x_cov,
           sample = function(n) rmvn(n, mean = mu, cov = S))
    },

    #' @description Draw parameter samples from the posterior.
    #' @param n number of draws.
    #' @param unit unit index or NULL (shared).
    sample = function(n, unit = NULL) self$posterior(unit)$sample(n)
  )
)

#' Extended Kalman filter on the parameter-augmented state
#'
#' @description Jacobians of the transition, measurement and event maps by
#'   finite differences; Joseph-form covariance update. With
#'   `learn_params = FALSE` the parameter rows of the gain are zeroed, so the
#'   filter tracks states with parameters frozen at the prior.
#' @export
EstimatorEKF <- R6::R6Class("EstimatorEKF", inherit = Estimator,
  public = list(
    #' @field z augmented mean.
    z = NULL,
    #' @field P augmented covariance.
    P = NULL,

    #' @description Set mean and covariance to the prior.
    init = function() {
      pr <- self$prior()
      self$z <- pr$z
      self$P <- pr$P
      invisible(self)
    },

    #' @description Mean of the augmented state.
    mean_z = function() self$z,

    #' @description Covariance of the augmented state.
    cov = function() self$P,

    #' @description Propagate all units from t0 to t1.
    #' @param t0,t1 window.
    #' @param u_list list of control vectors per unit.
    #' @param dfun disturbance function (realized, measured).
    predict = function(t0, t1, u_list, dfun) {
      if (t1 <= t0 + 1e-12) return(invisible(self))
      L <- self$layout
      Fm <- diag(L$n_z)
      z_new <- self$z
      i <- 1                                          # debugging line
      for (i in seq_len(L$n_units)) {
        ix <- L$idx_x[[i]]
        v0 <- c(self$z[ix], self$unit_phi(self$z, unit = i))
        jac <- fd_jacobian(function(v) self$transition(v, u = u_list[[i]], dfun = dfun, t0 = t0, t1 = t1),
                           v0 = v0, steps = self$fd_steps(v0))
        if (any(!is.finite(jac$J)) || any(!is.finite(jac$f0))) {
          warning("EKF predict: non-finite transition for unit ", i, " at time ", t1, "; state held")
          next
        }
        A_x <- jac$J[, seq_len(L$n_x), drop = FALSE]
        A_p <- jac$J[, L$n_x + seq_len(L$n_p), drop = FALSE]
        Fm[ix, ix] <- A_x
        Fm[ix, L$idx_p] <- A_p
        if (L$n_re > 0) Fm[ix, L$idx_b[[i]]] <- A_p %*% L$M
        z_new[ix] <- jac$f0
      }
      self$z <- self$clamp_z(z_new)
      self$P <- make_pd(Fm %*% self$P %*% t(Fm) + self$process_noise(t1 - t0))
      invisible(self)
    },

    #' @description Measurement update with the observations of any subset of units.
    #' @param time observation time.
    #' @param y_list list per unit of named observation vectors (NULL = none).
    #' @param sensor_sets sensor set name per unit.
    #' @param u_list controls per unit.
    #' @param dfun disturbance function.
    update = function(time, y_list, sensor_sets, u_list, dfun) {
      L <- self$layout
      d <- dfun(time)
      H_rows <- list(); innov <- c(); r_var <- c()
      i <- 1                                          # debugging line
      for (i in seq_len(L$n_units)) {
        y <- y_list[[i]]
        if (is.null(y) || length(y) == 0) next
        y <- y[!is.na(y)]
        if (length(y) == 0) next
        onames <- names(y)
        sd <- self$obs_sd(sensor_sets[i])[onames]
        ix <- L$idx_x[[i]]
        v0 <- c(self$z[ix], self$unit_phi(self$z, unit = i))
        jac <- fd_jacobian(function(v) self$measure(v, time = time, u = u_list[[i]], d = d, onames = onames),
                           v0 = v0, steps = self$fd_steps(v0))
        if (any(!is.finite(jac$J)) || any(!is.finite(jac$f0))) next
        Hi <- matrix(0, nrow = length(onames), ncol = L$n_z)
        H_p <- jac$J[, L$n_x + seq_len(L$n_p), drop = FALSE]
        Hi[, ix] <- jac$J[, seq_len(L$n_x), drop = FALSE]
        Hi[, L$idx_p] <- H_p
        if (L$n_re > 0) Hi[, L$idx_b[[i]]] <- H_p %*% L$M
        H_rows[[length(H_rows) + 1L]] <- Hi
        innov <- c(innov, as.numeric(y) - jac$f0)
        r_var <- c(r_var, sd^2)
      }
      if (length(H_rows) == 0) return(invisible(self))
      H <- do.call(rbind, H_rows)
      R <- diag(r_var, nrow = length(r_var))
      S <- make_pd(H %*% self$P %*% t(H) + R)
      K <- self$P %*% t(H) %*% solve(S)
      if (!self$learn_params) K[self$param_idx(), ] <- 0
      self$z <- self$clamp_z(as.numeric(self$z + K %*% innov))
      IKH <- diag(L$n_z) - K %*% H
      self$P <- make_pd(IKH %*% self$P %*% t(IKH) + K %*% R %*% t(K))
      invisible(self)
    },

    #' @description Apply a discrete event to one unit.
    #' @param unit unit index.
    #' @param event event name.
    #' @param time event time (unused by the EKF; interface symmetry).
    #' @param u controls (unused).
    #' @param dfun disturbance function (unused).
    apply_event = function(unit, event, time = NULL, u = NULL, dfun = NULL) {
      L <- self$layout
      ix <- L$idx_x[[unit]]
      v0 <- c(self$z[ix], self$unit_phi(self$z, unit = unit))
      jac <- fd_jacobian(function(v) self$event_map(v, event = event), v0 = v0, steps = self$fd_steps(v0))
      Fm <- diag(L$n_z)
      A_p <- jac$J[, L$n_x + seq_len(L$n_p), drop = FALSE]
      Fm[ix, ix] <- jac$J[, seq_len(L$n_x), drop = FALSE]
      Fm[ix, L$idx_p] <- A_p
      if (L$n_re > 0) Fm[ix, L$idx_b[[unit]]] <- A_p %*% L$M
      z_new <- self$z
      z_new[ix] <- jac$f0
      self$z <- self$clamp_z(z_new)
      self$P <- make_pd(Fm %*% self$P %*% t(Fm))
      invisible(self)
    }
  )
)

#' Create the estimator named in the config
#'
#' @param model fitted `dc_model`.
#' @param cfg scenario config (`cfg$estimator$type`: `ekf` or `enkf`).
#' @param n_units number of units.
#' @param seed seed.
#' @param learn_params learn parameters or track states only.
#' @export
make_estimator <- function(model, cfg, n_units, seed = 1L, learn_params = TRUE) {
  type <- cfg$estimator$type %||% "ekf"
  switch(type,
         ekf  = EstimatorEKF$new(model, cfg = cfg, n_units = n_units, seed = seed, learn_params = learn_params),
         enkf = EstimatorEnKF$new(model, cfg = cfg, n_units = n_units, seed = seed, learn_params = learn_params),
         stop("unknown estimator type: ", type))
}

#' Posterior summary table of an estimator
#'
#' @param est estimator.
#' @param level central interval level.
#' @param truth_unit matrix of true unit parameters (optional).
#' @param truth_pop named vector of true population parameters (optional).
#' @return data.table with one row per (unit or "shared", parameter).
#' @export
posterior_table <- function(est, level = 0.9, truth_unit = NULL, truth_pop = NULL) {
  zq <- stats::qnorm(1 - (1 - level) / 2)
  L <- est$layout
  rows <- list()
  i <- 0                                              # debugging line
  for (i in c(0, seq_len(L$n_units))) {
    po <- est$posterior(if (i == 0) NULL else i)
    tr <- if (i == 0) truth_pop else if (!is.null(truth_unit)) truth_unit[i, ] else NULL
    rows[[length(rows) + 1L]] <- data.table::data.table(
      unit = i, param = names(po$mean), mean = as.numeric(po$mean), sd = as.numeric(po$sd),
      lo = as.numeric(po$mean) - zq * as.numeric(po$sd), hi = as.numeric(po$mean) + zq * as.numeric(po$sd),
      truth = if (is.null(tr)) NA_real_ else as.numeric(tr[names(po$mean)]),
      logdet = logdet(po$cov))
  }
  data.table::rbindlist(rows)
}
