# =============================================================================
# model_spec.R
# The ModelSpec: everything the engine needs to know about one model.
#
#   dX/dt = rhs(time, x, u, d, parms)          states x, controls u,
#   Y     = outputs(time, x, u, d, parms)      disturbances d, parameters parms
#
# `outputs` returns ONE named vector holding the observables, the cost-rate
# components [currency/time] and the constraint margins g (violated if g > 0).
# The spec lists which output names play which role. Parameters carry a
# transform (log/logit/identity), a prior (mean in natural units, sd in the
# transformed space), a plausible range and a truth distribution for the
# simulated plant. Numbers come from the scenario YAML; equations from the
# model plug-in file.
# =============================================================================

# build a data.table from a named list of lists, filling defaults
spec_table <- function(lst, fields) {
  if (length(lst) == 0) {
    cols <- c(list(name = character(0)), lapply(fields, function(f) f[0]))
    return(data.table::as.data.table(cols))
  }
  nm <- names(lst)
  if (is.null(nm) || any(!nzchar(nm))) stop("spec entries must be a named list")
  rows <- lapply(nm, function(n) {
    e <- lst[[n]]
    vals <- lapply(names(fields), function(f) {
      v <- e[[f]]
      if (is.null(v)) v <- fields[[f]]
      if (length(v) != 1) stop("field '", f, "' of '", n, "' must be a scalar")
      if (is.numeric(fields[[f]])) v <- as.numeric(v)
      if (is.character(fields[[f]])) v <- as.character(v)
      if (is.logical(fields[[f]])) v <- as.logical(v)
      v
    })
    names(vals) <- names(fields)
    data.table::as.data.table(c(list(name = n), vals))
  })
  data.table::rbindlist(rows, use.names = TRUE)
}

# pre-split vector fields ("range: [lo, hi]", "bounds: [lo, hi]") into scalars
split_pair <- function(lst, field, lo_name, hi_name) {
  lapply(lst, function(e) {
    if (!is.null(e[[field]])) {
      e[[lo_name]] <- e[[field]][1]
      e[[hi_name]] <- e[[field]][2]
      e[[field]] <- NULL
    }
    e
  })
}

#' Create a model specification
#'
#' Usually called from a model plug-in file's `build_model(cfg, variant)`.
#' All numeric entries normally come straight from the scenario YAML.
#'
#' @param name model name.
#' @param states named list; each entry may hold `unit`, `init`, `lower`,
#'   `upper`, `init_sd` (sd of the initial state) and `process_sd` (state
#'   process noise sd per sqrt(time unit) used by the estimator).
#' @param params named list; each entry may hold `unit`, `value` (prior mean,
#'   natural units), `transform` (`log`, `logit`, `identity`), `bounds` (for
#'   logit), `prior_sd` (transformed space), `range` (plausible natural range),
#'   `estimate` (logical), `unit_sd` (prior sd of unit random effects in the
#'   estimator, transformed space, 0 = no random effect), `truth_pop_sd`
#'   (population truth is drawn with sd `truth_pop_sd * prior_sd`) and
#'   `truth_unit_sd` (sd of true unit deviations, transformed space).
#' @param constants named list of known fixed constants (geometry, prices).
#' @param controls named list; entries hold `unit`, `lower`, `upper`, `rate`
#'   (max change per decision epoch), `default`, `step` (advisor and probe
#'   step) and `change_cost` (implementation cost of one change).
#' @param events named list of discrete events; each holds `apply`
#'   (function(x, parms) returning the new state), `cost`
#'   (function(x, parms, u, d) returning a lump cost), `change_cost` and
#'   `min_interval_epochs`.
#' @param disturbances list with `names`, `units` and `generate`
#'   (function(times, cfg, seed) returning a data.table with `time`, one column
#'   per disturbance and optionally `<name>_clim` climatology columns).
#' @param rhs function(time, x, u, d, parms) returning dx/dt (state order).
#' @param outputs function(time, x, u, d, parms) returning a named vector.
#' @param sensor_sets named list; each entry holds `interval` (time between
#'   samples) and `outputs` (named list: output name -> noise sd).
#' @param cost_outputs names of outputs that are cost rates \[currency/time\].
#' @param constraint_outputs names of outputs that are constraint margins.
#' @param variant `fitted` or `truth`.
#' @param description free text.
#' @return object of class `dc_model`.
#' @export
model_spec <- function(name, states, params, constants = list(), controls,
                       events = list(), disturbances, rhs, outputs, sensor_sets,
                       cost_outputs, constraint_outputs = character(0),
                       variant = "fitted", description = "") {
  st <- spec_table(states, fields = list(unit = "-", init = 0, lower = -Inf, upper = Inf,
                                         init_sd = 0, process_sd = 0))
  params <- split_pair(params, field = "range", lo_name = "range_lo", hi_name = "range_hi")
  params <- split_pair(params, field = "bounds", lo_name = "bound_lo", hi_name = "bound_hi")
  pt <- spec_table(params, fields = list(unit = "-", value = NA_real_, transform = "identity",
                                         bound_lo = 0, bound_hi = 1, prior_sd = 1,
                                         range_lo = -Inf, range_hi = Inf, estimate = TRUE,
                                         unit_sd = 0, truth_pop_sd = 1, truth_unit_sd = 0))
  if (any(is.na(pt$value))) stop("every parameter needs a 'value' (prior mean, natural units)")
  bad_tr <- setdiff(unique(pt$transform), c("log", "logit", "identity"))
  if (length(bad_tr) > 0) stop("unknown parameter transform(s): ", paste(bad_tr, collapse = ", "))
  ct <- spec_table(controls, fields = list(unit = "-", lower = -Inf, upper = Inf, rate = Inf,
                                           default = NA_real_, step = NA_real_, change_cost = 1))
  if (any(is.na(ct$default))) stop("every control needs a 'default'")
  ct[is.na(step), step := (upper - lower) / 10]

  # events: functions from the model file, numbers from the config
  ev_names <- names(events)
  ev <- lapply(ev_names, function(n) {
    e <- events[[n]]
    if (!is.function(e$apply) || !is.function(e$cost)) stop("event '", n, "' needs functions 'apply' and 'cost'")
    e$change_cost         <- as.numeric(e$change_cost %||% 1)
    e$min_interval_epochs <- as.numeric(e$min_interval_epochs %||% 0)
    e
  })
  names(ev) <- ev_names

  ss <- lapply(sensor_sets, function(s) {
    sd <- unlist(s$outputs)
    list(interval = as.numeric(s$interval), outputs = names(sd), sd = sd)
  })

  dist_names <- disturbances$names %||% character(0)
  all_names <- c(st$name, pt$name, names(constants), ct$name, dist_names)
  dup <- unique(all_names[duplicated(all_names)])
  if (length(dup) > 0) stop("names must be unique across states, parameters, constants, controls and disturbances: ",
                            paste(dup, collapse = ", "))

  est <- pt$estimate
  phi_names <- pt$name[est]
  phi_mean <- transform_vec(pt$value[est], transform = pt$transform[est],
                            lo = pt$bound_lo[est], hi = pt$bound_hi[est])
  names(phi_mean) <- phi_names
  phi_sd <- stats::setNames(pt$prior_sd[est], phi_names)

  parms_nominal <- c(constants, stats::setNames(as.list(pt$value), pt$name))

  structure(list(
    name = name, variant = variant, description = description,
    states = st, state_names = st$name, n_x = nrow(st),
    params = pt, param_names = pt$name,
    phi_names = phi_names, n_p = length(phi_names),
    phi_prior_mean = phi_mean, phi_prior_sd = phi_sd,
    phi_transform = pt$transform[est], phi_lo = pt$bound_lo[est], phi_hi = pt$bound_hi[est],
    constants = constants, parms_nominal = parms_nominal,
    controls = ct, control_names = ct$name, n_u = nrow(ct),
    control_default = stats::setNames(ct$default, ct$name),
    events = ev, event_names = ev_names,
    disturbances = disturbances, dist_names = dist_names,
    rhs = rhs, outputs = outputs, sensor_sets = ss,
    cost_outputs = cost_outputs, constraint_outputs = constraint_outputs,
    output_names = NULL, n_out = NA_integer_
  ), class = "dc_model")
}

#' @export
print.dc_model <- function(x, ...) {
  cat("dualctl model '", x$name, "' (", x$variant, ")\n", sep = "")
  cat("  states      :", paste(x$state_names, collapse = ", "), "\n")
  cat("  estimated   :", paste(x$phi_names, collapse = ", "), "\n")
  cat("  controls    :", paste(x$control_names, collapse = ", "), "\n")
  cat("  events      :", paste(x$event_names, collapse = ", "), "\n")
  cat("  disturbances:", paste(x$dist_names, collapse = ", "), "\n")
  cat("  sensor sets :", paste(names(x$sensor_sets), collapse = ", "), "\n")
  invisible(x)
}

#' Convert transformed estimated parameters to a natural `parms` list
#'
#' @param model a `dc_model`.
#' @param phi transformed values of the estimated parameters (model order).
#' @param base_parms list with all other parameters and constants.
#' @return `parms` list in natural units.
#' @export
phi_to_parms <- function(model, phi, base_parms = model$parms_nominal) {
  theta <- inverse_transform_vec(as.numeric(phi), transform = model$phi_transform,
                                 lo = model$phi_lo, hi = model$phi_hi)
  base_parms[model$phi_names] <- as.list(theta)
  base_parms
}

#' Extract transformed estimated parameters from a natural `parms` list
#' @inheritParams phi_to_parms
#' @param parms `parms` list.
#' @export
parms_to_phi <- function(model, parms) {
  theta <- vapply(model$phi_names, function(n) as.numeric(parms[[n]]), numeric(1))
  out <- transform_vec(theta, transform = model$phi_transform, lo = model$phi_lo, hi = model$phi_hi)
  stats::setNames(out, model$phi_names)
}

# transformed plausible range of the estimated parameters (rows: lo, hi)
phi_range <- function(model) {
  pt <- model$params[model$params$estimate, ]
  tr <- function(v) {
    out <- rep(NA_real_, length(v))
    ok <- is.finite(v)
    out[ok] <- transform_vec(v[ok], transform = pt$transform[ok], lo = pt$bound_lo[ok], hi = pt$bound_hi[ok])
    out
  }
  lo <- tr(pt$range_lo)
  hi <- tr(pt$range_hi)
  lo[!is.finite(lo)] <- -Inf; hi[!is.finite(hi)] <- Inf
  rbind(lo = stats::setNames(lo, model$phi_names), hi = stats::setNames(hi, model$phi_names))
}

clamp_states <- function(x, model) {
  if (is.matrix(x)) {
    j <- 1                                            # debugging line
    for (j in seq_len(ncol(x))) x[, j] <- clamp(x[, j], model$states$lower[j], model$states$upper[j])
    return(x)
  }
  out <- clamp(as.numeric(x), model$states$lower, model$states$upper)
  stats::setNames(out, model$state_names)
}

enforce_control_limits <- function(u_new, u_prev, model) {
  ct <- model$controls
  u_new  <- as.numeric(u_new[ct$name])
  u_prev <- as.numeric(u_prev[ct$name])
  lo <- pmax(ct$lower, u_prev - ct$rate)
  hi <- pmin(ct$upper, u_prev + ct$rate)
  bad <- lo > hi
  lo[bad] <- hi[bad] <- clamp(u_prev[bad], ct$lower[bad], ct$upper[bad])
  stats::setNames(clamp(u_new, lo, hi), ct$name)
}

#' Load a model plug-in file and build a `dc_model`
#'
#' The file named in `cfg$model$file` is sourced into a fresh environment
#' (whose parent is the dualctl namespace, so engine helpers are visible) and
#' must define `build_model(cfg, variant)`. The result is checked by
#' [check_model()].
#'
#' @param cfg scenario config from [load_scenario()].
#' @param variant `fitted` (what the estimator and controller believe) or
#'   `truth` (the simulated plant, may contain extra states).
#' @export
load_model <- function(cfg, variant = c("fitted", "truth")) {
  variant <- match.arg(variant)
  path <- resolve_path(file = cfg$model$file, base_dir = cfg$.dir, subdir = "models")
  env <- new.env(parent = asNamespace("dualctl"))
  sys.source(path, envir = env)
  if (!exists("build_model", envir = env, inherits = FALSE)) {
    stop("model file ", path, " must define build_model(cfg, variant)")
  }
  model <- env$build_model(cfg, variant)
  if (!inherits(model, "dc_model")) stop("build_model() must return model_spec(...)")
  check_model(model, cfg)
}

#' Check a model against its config and record its output names
#'
#' Evaluates `rhs`, `outputs` and the events once at the nominal point and
#' fails with a readable message if lengths or names do not match.
#'
#' @param model a `dc_model`.
#' @param cfg scenario config.
#' @return the model with `output_names` and `n_out` filled in.
#' @export
check_model <- function(model, cfg) {
  dist_dt <- get_disturbances(model, cfg = cfg, times = c(0, 1, 2), seed = 1L)
  dfun <- make_interp_fun(dist_dt, cols = model$dist_names)
  x <- stats::setNames(model$states$init, model$state_names)
  u <- model$control_default
  d <- dfun(0)
  parms <- model$parms_nominal
  dx <- model$rhs(0, x, u, d, parms)
  if (length(dx) != model$n_x) stop("rhs() returned ", length(dx), " derivatives, expected ", model$n_x)
  if (any(!is.finite(dx))) stop("rhs() returned non-finite derivatives at the nominal point")
  out <- model$outputs(0, x, u, d, parms)
  if (is.null(names(out)) || any(!nzchar(names(out)))) stop("outputs() must return a fully named vector")
  need <- unique(c(unlist(lapply(model$sensor_sets, `[[`, "outputs")), model$cost_outputs,
                   model$constraint_outputs))
  miss <- setdiff(need, names(out))
  if (length(miss) > 0) stop("outputs() lacks: ", paste(miss, collapse = ", "))
  clash <- intersect(names(out), c("time", model$state_names, model$control_names))
  if (length(clash) > 0) stop("output names must differ from 'time', state and control names: ",
                              paste(clash, collapse = ", "))
  en <- model$event_names[1]                          # debugging line
  for (en in model$event_names) {
    xn <- model$events[[en]]$apply(x, parms)
    if (length(xn) != model$n_x) stop("event '", en, "' apply() must return ", model$n_x, " states")
    ce <- model$events[[en]]$cost(x, parms, u, d)
    if (length(ce) != 1 || !is.finite(ce)) stop("event '", en, "' cost() must return one finite number")
  }
  model$output_names <- names(out)
  model$n_out <- length(out)
  model
}
