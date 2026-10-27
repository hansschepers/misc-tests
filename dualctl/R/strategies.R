# =============================================================================
# strategies.R
# ProbingStrategy interface:  act(state, posterior, forecast, time) -> control
#   state     list: unit, epoch, n_epochs, season_frac, x_mean, x_cov, u_prev,
#             last_event_epoch, last_cost_rate, last_obs, sensor_set, arm,
#             seed, and truth (only for strategies with needs_truth = TRUE)
#   posterior list: mean, cov, sd (transformed parameters), sample(n)
#   forecast  data.table of forecast disturbances from `time` over the horizon
#   time      decision time (the task's `t`; `t` is avoided as an object name)
# returns list(u = named control vector, events = character, info = list).
# One strategy object per unit, so strategies may keep internal state.
#
# Implemented: calendar, ce ("none": certainty equivalence, no probing),
# dither (PRBS / multisine aimed at cheap, informative directions), infompc
# (cost - lambda_t * expected log-det reduction), thompson, evsi (EVPI-scaled
# information value), hillclimb (naive one-at-a-time), oracle (true model and
# parameters). New strategies: subclass ProbingStrategy and register_strategy().
# =============================================================================

.dualctl_registry <- new.env(parent = emptyenv())

#' Register a probing strategy
#' @param name strategy name used in method configs.
#' @param generator R6 class generator inheriting from [ProbingStrategy].
#' @export
register_strategy <- function(name, generator) {
  assign(name, generator, envir = .dualctl_registry)
  invisible(name)
}

#' Names of the registered strategies
#' @export
list_strategies <- function() sort(ls(.dualctl_registry))

#' Instantiate a registered strategy
#' @param name strategy name.
#' @param model fitted `dc_model`.
#' @param cfg scenario config.
#' @param unit unit index.
#' @param seed seed.
#' @param model_truth truth model (needed by `oracle`).
#' @export
make_strategy <- function(name, model, cfg, unit = 1L, seed = 1L, model_truth = NULL) {
  if (!exists(name, envir = .dualctl_registry, inherits = FALSE)) stop("unknown strategy: ", name)
  gen <- get(name, envir = .dualctl_registry, inherits = FALSE)
  gen$new(model, cfg = cfg, unit = unit, seed = seed, model_truth = model_truth)
}

#' Probing strategy base class
#'
#' @description Holds the model, config and a [Controller]; provides the
#'   certainty-equivalent plan, probe candidates and information gains used
#'   by the concrete strategies.
#' @export
ProbingStrategy <- R6::R6Class("ProbingStrategy",
  public = list(
    #' @field name strategy name (key in `cfg$strategies`).
    name = "base",
    #' @field model fitted model.
    model = NULL,
    #' @field cfg scenario config.
    cfg = NULL,
    #' @field scfg this strategy's config block.
    scfg = NULL,
    #' @field controller receding-horizon controller.
    controller = NULL,
    #' @field unit unit index.
    unit = NULL,
    #' @field seed seed.
    seed = NULL,
    #' @field needs_truth TRUE only for the oracle.
    needs_truth = FALSE,
    #' @field uses_posterior FALSE for model-free baselines.
    uses_posterior = TRUE,

    #' @description Create a strategy.
    #' @param model fitted `dc_model`.
    #' @param cfg scenario config.
    #' @param unit unit index.
    #' @param seed seed.
    #' @param model_truth truth model (oracle only).
    initialize = function(model, cfg, unit = 1L, seed = 1L, model_truth = NULL) {
      self$model <- model
      self$cfg <- cfg
      self$unit <- unit
      self$seed <- seed
      self$scfg <- cfg$strategies[[self$name]] %||% list()
      self$controller <- Controller$new(model, cfg = cfg)
      self$setup(model_truth)
    },

    #' @description Strategy-specific set-up (hook).
    #' @param model_truth truth model.
    setup = function(model_truth) invisible(NULL),

    #' @description Choose the controls for the coming epoch.
    #' @param state,posterior,forecast,time see the file header.
    act = function(state, posterior, forecast, time) stop("abstract"),

    #' @description Certainty-equivalent plan for parameters `phi`.
    #' @param state strategy state.
    #' @param phi transformed parameters.
    #' @param forecast forecast table.
    #' @param time decision time.
    #' @param ... passed to `Controller$plan`.
    ce_plan = function(state, phi, forecast, time, ...) {
      parms <- phi_to_parms(self$model, phi = phi)
      self$controller$plan(x0 = state$x_mean, parms = parms, u_prev = state$u_prev,
                           forecast_dt = forecast, t0 = time, epoch = state$epoch,
                           last_event_epoch = state$last_event_epoch,
                           seed = derive_seed(state$seed, "plan"), ...)
    },

    #' @description Standard result list.
    #' @param u control vector.
    #' @param events event names.
    #' @param info diagnostics.
    result = function(u, events = character(0), info = list()) {
      cn <- self$model$control_names
      list(u = stats::setNames(as.numeric(u[cn]), cn), events = as.character(events), info = info)
    },

    #' @description Probe candidates around a plan: the plan itself plus sign
    #'   patterns of +-step on the first epoch (all patterns for <= 3 controls).
    #' @param U_ce plan control matrix.
    #' @param u_prev previous controls.
    #' @param step_mult step multiplier.
    probe_candidates = function(U_ce, u_prev, step_mult = 1) {
      n_u <- self$model$n_u
      step <- self$model$controls$step * step_mult
      signs <- if (n_u <= 3) as.matrix(expand.grid(rep(list(c(-1, 0, 1)), n_u))) else
        rbind(diag(n_u), -diag(n_u))
      signs <- signs[rowSums(abs(signs)) > 0, , drop = FALSE]
      out <- list(U_ce)
      r <- 1                                          # debugging line
      for (r in seq_len(nrow(signs))) {
        U <- U_ce
        U[1, ] <- U[1, ] + as.numeric(signs[r, ]) * step
        U <- self$controller$project_U(U, u_prev = u_prev)
        if (max(abs(U - U_ce)) > 1e-12) out[[length(out) + 1L]] <- U
      }
      out
    },

    #' @description Expected information gain of a plan over the info window.
    #' @param U plan control matrix.
    #' @param ev event option or NULL.
    #' @param state strategy state.
    #' @param posterior posterior list.
    #' @param parms natural parameters.
    #' @param dfun forecast function.
    #' @param time decision time.
    #' @param window_epochs information window in epochs.
    info_gain = function(U, ev, state, posterior, parms, dfun, time, window_epochs) {
      ctl <- self$controller
      t1 <- time + min(window_epochs, ctl$H) * ctl$E
      u_plan <- make_u_plan(times = time + (seq_len(ctl$H) - 1) * ctl$E, U = U,
                            control_names = self$model$control_names)
      ev_plan <- NULL
      if (!is.null(ev) && time + ev$tau * ctl$E < t1) {
        ev_plan <- data.table::data.table(time = time + ev$tau * ctl$E, event = ev$name)
      }
      expected_info_gain(self$model, x0 = state$x_mean, parms = parms, phi_cov = posterior$cov,
                         u_plan = u_plan, dfun = dfun, t0 = time, t1 = t1,
                         sensor_set = state$sensor_set, events_plan = ev_plan,
                         noise_mult = self$cfg$sensors$noise_mult %||% 1,
                         dt = self$cfg$solver$dt_inner, engine = self$cfg$solver$inner_engine,
                         h = self$cfg$identifiability$fd_step)
    },

    #' @description Score probe candidates: cost J and information gain.
    #' @param cands list of control matrices.
    #' @param ev event option.
    #' @param state,posterior,forecast,time see `act`.
    score_candidates = function(cands, ev, state, posterior, forecast, time) {
      parms <- phi_to_parms(self$model, phi = posterior$mean)
      dfun <- make_interp_fun(forecast, cols = self$model$dist_names)
      w <- self$scfg$info_window_epochs %||% 1
      sc <- vapply(cands, function(U) {
        J <- self$controller$evaluate_U(U, ev = ev, x0 = state$x_mean, parms = parms, dfun = dfun, t0 = time)$J
        ig <- self$info_gain(U, ev = ev, state = state, posterior = posterior, parms = parms,
                             dfun = dfun, time = time, window_epochs = w)
        c(J = J, ig = ig)
      }, numeric(2))
      matrix(sc, nrow = 2, dimnames = list(c("J", "ig"), NULL))
    }
  )
)

#' Certainty-equivalent MPC (probing strategy "none")
#' @description Plans with the posterior mean and applies the first epoch.
#' @export
StrategyCE <- R6::R6Class("StrategyCE", inherit = ProbingStrategy,
  public = list(
    #' @field name strategy name.
    name = "ce",
    #' @description Act.
    #' @param state,posterior,forecast,time see [ProbingStrategy].
    act = function(state, posterior, forecast, time) {
      pl <- self$ce_plan(state, phi = posterior$mean, forecast = forecast, time = time)
      self$result(pl$u_first, events = pl$events_first, info = list(J = pl$J, n_eval = pl$n_eval))
    }
  )
)

#' Fixed calendar schedule (baseline)
#' @description Fixed recipe and periodic events from `cfg$strategies$calendar`.
#' @export
StrategyCalendar <- R6::R6Class("StrategyCalendar", inherit = ProbingStrategy,
  public = list(
    #' @field name strategy name.
    name = "calendar",
    #' @field uses_posterior model-free.
    uses_posterior = FALSE,
    #' @field sched schedule.
    sched = NULL,
    #' @description Build the schedule.
    #' @param model_truth unused.
    setup = function(model_truth) {
      n_ep <- ceiling(self$cfg$season$length_days / self$cfg$season$epoch_days) + 1
      self$sched <- calendar_schedule(self$model, cfg = self$cfg, n_epochs = n_ep)
    },
    #' @description Act.
    #' @param state,posterior,forecast,time see [ProbingStrategy].
    act = function(state, posterior, forecast, time) {
      ev <- self$sched$events$event[self$sched$events$epoch == state$epoch]
      self$result(self$sched$controls, events = ev)
    }
  )
)

#' Thompson sampling
#' @description Plans with one posterior draw per epoch.
#' @export
StrategyThompson <- R6::R6Class("StrategyThompson", inherit = ProbingStrategy,
  public = list(
    #' @field name strategy name.
    name = "thompson",
    #' @description Act.
    #' @param state,posterior,forecast,time see [ProbingStrategy].
    act = function(state, posterior, forecast, time) {
      phi <- with_seed(derive_seed(state$seed, "thompson"), posterior$sample(1))[1, ]
      pl <- self$ce_plan(state, phi = phi, forecast = forecast, time = time)
      self$result(pl$u_first, events = pl$events_first, info = list(J = pl$J, phi = phi))
    }
  )
)

#' Information-aware MPC
#'
#' @description Minimises J - lambda_t * IG over the CE plan and its probe
#'   candidates; lambda_t = lambda0 * anneal(season fraction) \[currency/nat\].
#' @export
StrategyInfoMPC <- R6::R6Class("StrategyInfoMPC", inherit = ProbingStrategy,
  public = list(
    #' @field name strategy name.
    name = "infompc",
    #' @description Act.
    #' @param state,posterior,forecast,time see [ProbingStrategy].
    act = function(state, posterior, forecast, time) {
      sc <- self$scfg
      lam <- sc$lambda0 * anneal_schedule(state$season_frac %||% 0, schedule = sc$schedule,
                                          rate = sc$decay_rate %||% 3, step_at = sc$step_at %||% 0.5)
      ce <- self$ce_plan(state, phi = posterior$mean, forecast = forecast, time = time)
      if (lam <= 0) return(self$result(ce$u_first, events = ce$events_first, info = list(lambda = 0)))
      cands <- self$probe_candidates(ce$U, u_prev = state$u_prev, step_mult = sc$step_mult %||% 1)
      sm <- self$score_candidates(cands, ev = ce$event, state = state, posterior = posterior,
                                  forecast = forecast, time = time)
      obj <- sm["J", ] - lam * sm["ig", ]
      best <- which.min(obj)
      self$result(cands[[best]][1, ], events = ce$events_first,
                  info = list(lambda = lam, ig = sm["ig", best], ig_ce = sm["ig", 1],
                              extra_cost = sm["J", best] - sm["J", 1]))
    }
  )
)

#' Dither around the certainty-equivalent plan
#'
#' @description Bounded PRBS or multisine perturbation of the CE controls.
#'   Per control, the amplitude is weighted by information gain per unit of
#'   perturbation cost (flat-cost, high-sensitivity directions get the
#'   largest dither); weights are refreshed every `recompute_every_epochs`.
#' @export
StrategyDither <- R6::R6Class("StrategyDither", inherit = ProbingStrategy,
  public = list(
    #' @field name strategy name.
    name = "dither",
    #' @field weights per-control dither weights.
    weights = NULL,
    #' @field weights_epoch epoch of the last weight update.
    weights_epoch = -Inf,
    #' @field prbs PRBS matrix (epochs x controls).
    prbs = NULL,
    #' @field phases multisine phases.
    phases = NULL,
    #' @description Pre-compute the dither signals.
    #' @param model_truth unused.
    setup = function(model_truth) {
      n_ep <- ceiling(self$cfg$season$length_days / self$cfg$season$epoch_days) + 1
      n_u <- self$model$n_u
      hold <- self$scfg$prbs_hold_epochs %||% 1
      self$prbs <- vapply(seq_len(n_u), function(j) {
        prbs_signal(n_ep, hold = hold, seed = derive_seed(self$seed, "dither", self$unit, j))
      }, numeric(n_ep))
      self$prbs <- matrix(self$prbs, nrow = n_ep, ncol = n_u)
      self$phases <- with_seed(derive_seed(self$seed, "phase", self$unit), stats::runif(n_u, 0, 2 * pi))
    },
    #' @description Dither signal in \[-1, 1\] per control at epoch k.
    #' @param k epoch index.
    signal = function(k) {
      n_u <- self$model$n_u
      if (identical(self$scfg$signal, "multisine")) {
        per <- rep_len(as.numeric(unlist(self$scfg$multisine_periods)), n_u)
        return(multisine_signal(k, period = per, phase = self$phases))
      }
      self$prbs[min(k + 1, nrow(self$prbs)), ]
    },
    #' @description Information-per-cost weights of the control directions.
    #' @param ce CE plan.
    #' @param state,posterior,forecast,time see [ProbingStrategy].
    direction_weights = function(ce, state, posterior, forecast, time) {
      n_u <- self$model$n_u
      step <- self$model$controls$step * (self$scfg$amplitude %||% 1)
      cands <- list(ce$U)
      j <- 1                                          # debugging line
      for (j in seq_len(n_u)) {
        sgn <- -1                                     # debugging line
        for (sgn in c(-1, 1)) {
          U <- ce$U
          U[1, j] <- U[1, j] + sgn * step[j]
          cands[[length(cands) + 1L]] <- self$controller$project_U(U, u_prev = state$u_prev)
        }
      }
      sm <- self$score_candidates(cands, ev = ce$event, state = state, posterior = posterior,
                                  forecast = forecast, time = time)
      J0 <- sm["J", 1]; ig0 <- sm["ig", 1]
      floor_c <- self$scfg$min_cost_curvature %||% 1
      r <- vapply(seq_len(n_u), function(j) {
        k <- 1 + 2 * (j - 1) + c(1, 2)
        cost_j <- max(mean(sm["J", k]) - J0, floor_c)
        info_j <- max(mean(sm["ig", k]) - ig0, 0)
        info_j / cost_j
      }, numeric(1))
      if (max(r) <= 0) return(rep(1, n_u))
      r / max(r)
    },
    #' @description Act.
    #' @param state,posterior,forecast,time see [ProbingStrategy].
    act = function(state, posterior, forecast, time) {
      sc <- self$scfg
      ce <- self$ce_plan(state, phi = posterior$mean, forecast = forecast, time = time)
      amp <- (sc$amplitude %||% 1) * anneal_schedule(state$season_frac %||% 0, schedule = sc$schedule,
                                                     rate = sc$decay_rate %||% 3, step_at = sc$step_at %||% 0.5)
      if (amp <= 0) return(self$result(ce$u_first, events = ce$events_first, info = list(amp = 0)))
      if (is.null(self$weights) || state$epoch - self$weights_epoch >= (sc$recompute_every_epochs %||% 1)) {
        self$weights <- self$direction_weights(ce, state = state, posterior = posterior,
                                               forecast = forecast, time = time)
        self$weights_epoch <- state$epoch
      }
      u <- ce$u_first + amp * self$model$controls$step * self$weights * self$signal(state$epoch)
      self$result(u, events = ce$events_first, info = list(amp = amp, weights = self$weights))
    }
  )
)

#' EVSI-targeted probing (heuristic)
#'
#' @description Expected value of perfect information over the horizon,
#'   EVPI = E_theta\[J(u_CE; theta) - J(u*_theta; theta)\] from posterior draws,
#'   is scaled by the fraction of parameter variance a candidate is expected
#'   to resolve, 1 - exp(-2 IG / n_p), and by the remaining season
#'   (min(remaining, cap_epochs) / H). The candidate minimising
#'   J - EVSI is applied. A cheap approximation, not a full preposterior
#'   analysis.
#' @export
StrategyEVSI <- R6::R6Class("StrategyEVSI", inherit = ProbingStrategy,
  public = list(
    #' @field name strategy name.
    name = "evsi",
    #' @description Act.
    #' @param state,posterior,forecast,time see [ProbingStrategy].
    act = function(state, posterior, forecast, time) {
      sc <- self$scfg
      ce <- self$ce_plan(state, phi = posterior$mean, forecast = forecast, time = time)
      M <- sc$n_samples %||% 4
      thetas <- with_seed(derive_seed(state$seed, "evsi"), posterior$sample(M))
      dfun <- make_interp_fun(forecast, cols = self$model$dist_names)
      gaps <- vapply(seq_len(M), function(m) {
        pm <- phi_to_parms(self$model, phi = thetas[m, ])
        J_ce <- self$controller$evaluate_U(ce$U, ev = ce$event, x0 = state$x_mean, parms = pm,
                                           dfun = dfun, t0 = time)$J
        J_opt <- self$controller$plan(x0 = state$x_mean, parms = pm, u_prev = state$u_prev,
                                      forecast_dt = forecast, t0 = time, epoch = state$epoch,
                                      last_event_epoch = state$last_event_epoch,
                                      seed = derive_seed(state$seed, "evsi_opt", m),
                                      n_candidates = sc$n_candidates, maxit = sc$maxit)$J
        max(J_ce - J_opt, 0)
      }, numeric(1))
      evpi <- mean(gaps)
      remaining <- (state$n_epochs %||% self$controller$H) - state$epoch
      value_factor <- min(remaining, sc$cap_epochs %||% self$controller$H) / self$controller$H
      if (evpi <= 0) return(self$result(ce$u_first, events = ce$events_first, info = list(evpi = 0)))
      cands <- self$probe_candidates(ce$U, u_prev = state$u_prev, step_mult = sc$step_mult %||% 1)
      sm <- self$score_candidates(cands, ev = ce$event, state = state, posterior = posterior,
                                  forecast = forecast, time = time)
      frac <- 1 - exp(-2 * sm["ig", ] / self$model$n_p)
      evsi <- evpi * value_factor * frac
      best <- which.min(sm["J", ] - evsi)
      self$result(cands[[best]][1, ], events = ce$events_first,
                  info = list(evpi = evpi, evsi = evsi[best], extra_cost = sm["J", best] - sm["J", 1]))
    }
  )
)

#' Naive one-at-a-time hill climbing (baseline)
#'
#' @description Model-free: changes one control by one step, keeps the change
#'   if the measured operating cost rate of the next epoch fell, otherwise
#'   tries the other direction and then the next control. Weather confounds
#'   the comparison (that is the point of this baseline). Events follow a
#'   threshold rule on a measured output (`cfg$strategies$hillclimb$event_rule`).
#' @export
StrategyHillClimb <- R6::R6Class("StrategyHillClimb", inherit = ProbingStrategy,
  public = list(
    #' @field name strategy name.
    name = "hillclimb",
    #' @field uses_posterior model-free.
    uses_posterior = FALSE,
    #' @field u_acc accepted recipe.
    u_acc = NULL,
    #' @field ref_cost cost rate of the accepted recipe.
    ref_cost = NULL,
    #' @field j control index being tuned.
    j = 1L,
    #' @field dir step direction.
    dir = 1,
    #' @field tried directions tried for control j.
    tried = 0L,
    #' @field phase "baseline" or "trial".
    phase = "baseline",
    #' @field ref_obs reference output for the event rule.
    ref_obs = NULL,
    #' @description Act.
    #' @param state,posterior,forecast,time see [ProbingStrategy].
    act = function(state, posterior, forecast, time) {
      m <- self$model
      cn <- m$control_names
      ct <- m$controls
      step <- ct$step * (self$scfg$step_mult %||% 1)
      u_prev <- stats::setNames(as.numeric(state$u_prev[cn]), cn)
      if (is.null(self$u_acc)) self$u_acc <- u_prev
      c_last <- state$last_cost_rate
      if (!is.null(c_last) && length(c_last) == 1 && is.finite(c_last)) {
        if (is.null(self$ref_cost)) {
          self$ref_cost <- c_last
        } else if (identical(self$phase, "trial")) {
          if (c_last < self$ref_cost) {
            self$u_acc <- u_prev                      # keep the applied trial
            self$ref_cost <- c_last
            self$tried <- 0L
          } else {
            self$tried <- self$tried + 1L
            self$dir <- -self$dir
            if (self$tried >= 2L) {
              self$j <- self$j %% length(cn) + 1L
              self$tried <- 0L
            }
          }
        }
      }
      u_try <- self$u_acc
      if (!is.null(self$ref_cost)) {
        jj <- self$j
        u_try[jj] <- clamp(u_try[jj] + self$dir * step[jj], ct$lower[jj], ct$upper[jj])
        self$phase <- "trial"
      } else {
        self$phase <- "baseline"
      }
      events <- character(0)
      rule <- self$scfg$event_rule
      if (!is.null(rule) && !is.null(state$last_obs) && rule$output %in% names(state$last_obs)) {
        y <- state$last_obs[[rule$output]]
        if (is.null(self$ref_obs)) self$ref_obs <- y
        last <- if (rule$event %in% names(state$last_event_epoch)) state$last_event_epoch[[rule$event]] else -Inf
        allowed <- state$epoch - last >= m$events[[rule$event]]$min_interval_epochs
        fire <- (!is.null(rule$below_fraction) && y < rule$below_fraction * self$ref_obs) ||
          (!is.null(rule$above_fraction) && y > rule$above_fraction * self$ref_obs)
        if (allowed && fire) {
          events <- rule$event
          self$ref_obs <- NULL
          self$ref_cost <- NULL
          self$phase <- "baseline"
          u_try <- self$u_acc
        }
      }
      self$result(u_try, events = events, info = list(phase = self$phase, control = cn[self$j]))
    }
  )
)

#' Oracle: MPC with the true model, true parameters and true state
#' @description Benchmark for regret. Uses the same forecasts as everyone else.
#' @export
StrategyOracle <- R6::R6Class("StrategyOracle", inherit = ProbingStrategy,
  public = list(
    #' @field name strategy name.
    name = "oracle",
    #' @field needs_truth requires `state$truth`.
    needs_truth = TRUE,
    #' @field uses_posterior no estimator needed.
    uses_posterior = FALSE,
    #' @description Use a controller on the truth model.
    #' @param model_truth truth model.
    setup = function(model_truth) {
      if (is.null(model_truth)) stop("the oracle needs the truth model")
      self$controller <- Controller$new(model_truth, cfg = self$cfg)
    },
    #' @description Act.
    #' @param state,posterior,forecast,time see [ProbingStrategy].
    act = function(state, posterior, forecast, time) {
      pl <- self$controller$plan(x0 = state$truth$x, parms = state$truth$parms, u_prev = state$u_prev,
                                 forecast_dt = forecast, t0 = time, epoch = state$epoch,
                                 last_event_epoch = state$last_event_epoch,
                                 seed = derive_seed(state$seed, "plan"))
      self$result(pl$u_first, events = pl$events_first, info = list(J = pl$J))
    }
  )
)

register_builtin_strategies <- function() {
  register_strategy("calendar", StrategyCalendar)
  register_strategy("ce", StrategyCE)
  register_strategy("none", StrategyCE)
  register_strategy("dither", StrategyDither)
  register_strategy("infompc", StrategyInfoMPC)
  register_strategy("thompson", StrategyThompson)
  register_strategy("evsi", StrategyEVSI)
  register_strategy("hillclimb", StrategyHillClimb)
  register_strategy("oracle", StrategyOracle)
  invisible(NULL)
}
