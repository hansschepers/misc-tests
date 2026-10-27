# =============================================================================
# plant.R
# The simulated truth: a `truth` variant of the model (possibly with extra
# states = deliberate structural mismatch) whose parameters are hidden.
# Units are drawn from a hierarchical population:
#   phi_pop   = prior_mean + truth_pop_sd * prior_sd * z          (shared)
#   phi_unit  = phi_pop + truth_unit_sd * z_unit                   (random effects)
# in the transformed parameter space. Weather, population and observation
# noise depend only on the seed, so different methods see identical plants.
# =============================================================================

#' Draw a hierarchical population of true parameters
#'
#' @param model a `dc_model` (truth or fitted; uses the estimated-parameter table).
#' @param cfg scenario config (`cfg$plant$truth_spread_mult`,
#'   `cfg$plant$clamp_to_range`).
#' @param n_units number of units.
#' @param seed population seed.
#' @return list with `pop` (named vector) and `unit` (matrix n_units x n_p),
#'   transformed space.
#' @export
draw_population <- function(model, cfg, n_units, seed) {
  pn <- model$phi_names
  pt <- model$params[match(pn, model$params$name), ]
  mult <- cfg$plant$truth_spread_mult %||% 1
  rng <- phi_range(model)
  with_seed(seed, {
    z_pop <- stats::rnorm(model$n_p)
    Z <- matrix(stats::rnorm(n_units * model$n_p), nrow = n_units, ncol = model$n_p)
    pop <- model$phi_prior_mean + mult * pt$truth_pop_sd * model$phi_prior_sd * z_pop
    unit <- matrix(pop, nrow = n_units, ncol = model$n_p, byrow = TRUE) +
      sweep(Z, 2, pt$truth_unit_sd, "*")
    if (isTRUE(cfg$plant$clamp_to_range)) {
      pop <- clamp(pop, rng["lo", ], rng["hi", ])
      j <- 1                                          # debugging line
      for (j in seq_len(model$n_p)) unit[, j] <- clamp(unit[, j], rng["lo", j], rng["hi", j])
    }
    names(pop) <- pn
    colnames(unit) <- pn
    list(pop = pop, unit = unit)
  })
}

#' Simulated plant
#'
#' @description R6 class holding the true states, hidden parameters and
#'   realized disturbances of all units in one cohort.
#' @export
Plant <- R6::R6Class("Plant",
  public = list(
    #' @field model truth model.
    model = NULL,
    #' @field cfg scenario config.
    cfg = NULL,
    #' @field seed base seed.
    seed = NULL,
    #' @field n_units number of units.
    n_units = NULL,
    #' @field pop_phi true population parameters (transformed).
    pop_phi = NULL,
    #' @field unit_phi true unit parameters (matrix, transformed).
    unit_phi = NULL,
    #' @field unit_parms list of natural parameter lists per unit.
    unit_parms = NULL,
    #' @field x0 initial true states per unit.
    x0 = NULL,
    #' @field x current true states per unit.
    x = NULL,
    #' @field dist realized disturbance table.
    dist = NULL,
    #' @field dfun interpolating function of `dist`.
    dfun = NULL,
    #' @field noise_mult observation noise multiplier.
    noise_mult = 1,

    #' @description Create the plant.
    #' @param model_truth truth `dc_model`.
    #' @param cfg scenario config.
    #' @param seed base seed.
    #' @param n_units number of units.
    initialize = function(model_truth, cfg, seed, n_units) {
      self$model <- model_truth
      self$cfg <- cfg
      self$seed <- seed
      self$n_units <- n_units
      E <- cfg_num(cfg$season$epoch_days, "season$epoch_days")
      t_end <- cfg_num(cfg$season$length_days, "season$length_days") +
        (cfg$controller$horizon_epochs + 2) * E
      times <- seq(0, t_end, by = cfg_num(cfg$disturbances$dt, "disturbances$dt"))
      self$dist <- get_disturbances(model_truth, cfg = cfg, times = times,
                                    seed = derive_seed(seed, "weather"))
      self$dfun <- make_interp_fun(self$dist, cols = model_truth$dist_names)
      pop <- draw_population(model_truth, cfg = cfg, n_units = n_units,
                             seed = derive_seed(seed, "population"))
      self$pop_phi <- pop$pop
      self$unit_phi <- pop$unit
      base <- model_truth$parms_nominal
      ov <- cfg$plant$parm_overrides %||% list()
      ov <- ov[names(ov) %in% names(base)]
      base[names(ov)] <- lapply(ov, as.numeric)
      self$unit_parms <- lapply(seq_len(n_units), function(i) {
        phi_to_parms(model_truth, phi = self$unit_phi[i, ], base_parms = base)
      })
      st <- model_truth$states
      self$x0 <- with_seed(derive_seed(seed, "initial_state"), lapply(seq_len(n_units), function(i) {
        clamp_states(st$init + st$init_sd * stats::rnorm(nrow(st)), model_truth)
      }))
      self$x <- self$x0
      self$noise_mult <- (cfg$sensors$noise_mult %||% 1) * (cfg$plant$extra_noise_mult %||% 1)
    },

    #' @description Advance one unit with constant controls.
    #' @param unit unit index.
    #' @param t0,t1 window.
    #' @param u named control vector.
    #' @param out_times times at which true outputs are returned.
    #' @return result of [simulate_unit()].
    advance = function(unit, t0, t1, u, out_times = numeric(0)) {
      u_plan <- make_u_plan(times = t0, U = u, control_names = self$model$control_names)
      res <- simulate_unit(self$model, x0 = self$x[[unit]], parms = self$unit_parms[[unit]],
                           u_plan = u_plan, dfun = self$dfun, t0 = t0, t1 = t1,
                           dt = self$cfg$solver$dt_plant, out_times = out_times,
                           engine = self$cfg$solver$plant_engine)
      self$x[[unit]] <- res$x_end
      res
    },

    #' @description Apply a discrete event (e.g. cleaning) to one unit.
    #' @param unit unit index.
    #' @param event event name.
    #' @param time event time.
    #' @param u controls at that time.
    #' @return lump event cost.
    apply_event = function(unit, event, time, u) {
      ev <- self$model$events[[event]]
      parms <- self$unit_parms[[unit]]
      cost <- ev$cost(self$x[[unit]], parms, u, self$dfun(time))
      self$x[[unit]] <- clamp_states(ev$apply(self$x[[unit]], parms), self$model)
      cost
    },

    #' @description Noisy measurement of one unit (common random numbers:
    #'   the noise depends on seed, unit, time and sensor set only).
    #' @param unit unit index.
    #' @param time sample time.
    #' @param sensor_set sensor set name.
    #' @param y_true named vector of true outputs.
    observe = function(unit, time, sensor_set, y_true) {
      ss <- self$model$sensor_sets[[sensor_set]]
      sd <- ss$sd[ss$outputs] * self$noise_mult
      z <- with_seed(derive_seed(self$seed, "obs", unit, round(time * 1000), sensor_set),
                     stats::rnorm(length(sd)))
      stats::setNames(as.numeric(y_true[ss$outputs]) + sd * z, ss$outputs)
    },

    #' @description True transformed parameters of a unit (fitted-model names).
    #' @param unit unit index.
    true_phi = function(unit) self$unit_phi[unit, ]
  )
)
