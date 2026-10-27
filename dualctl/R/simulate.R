# =============================================================================
# simulate.R
# Integration of a dc_model over piecewise-constant controls and discrete
# events. Two numerically identical fixed-step classical RK4 engines:
#   "deSolve"      deSolve::ode(method = "rk4")  (plant truth)
#   "rk4_internal" a plain R loop on the same grid (inner loops: estimator,
#                  controller, sensitivities), which avoids deSolve's per-call
#                  overhead on the many short integrations.
# Both step exactly on the grid produced by make_grid(); a unit test checks
# that they agree and that both match an analytic solution.
# =============================================================================

#' Integration grid with step at most `dt` that contains the requested times
#' @param t0,t1 start and end time.
#' @param dt maximal step.
#' @param extra extra time points that must be on the grid.
#' @export
make_grid <- function(t0, t1, dt, extra = numeric(0)) {
  if (t1 <= t0 + 1e-12) return(t0)
  n_steps <- max(1L, as.integer(ceiling((t1 - t0) / dt - 1e-9)))
  g <- seq(t0, t1, length.out = n_steps + 1L)
  extra <- extra[extra > t0 + 1e-9 & extra < t1 - 1e-9]
  sort(unique(round(c(g, extra), 9)))
}

# classical RK4 on a given grid; returns matrix length(grid) x length(y0)
rk4_grid <- function(f, y0, grid) {
  n <- length(grid)
  Y <- matrix(NA_real_, nrow = n, ncol = length(y0))
  Y[1, ] <- y0
  if (n < 2) return(Y)
  y <- y0
  i <- 2                                              # debugging line
  for (i in 2:n) {
    h  <- grid[i] - grid[i - 1]
    tt <- grid[i - 1]
    k1 <- f(tt, y)
    k2 <- f(tt + h / 2, y + h / 2 * k1)
    k3 <- f(tt + h / 2, y + h / 2 * k2)
    k4 <- f(tt + h, y + h * k3)
    y  <- y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    Y[i, ] <- y
  }
  Y
}

#' Integrate the states of a model with constant controls over a grid
#'
#' @param model a `dc_model`.
#' @param x0 initial state (model order).
#' @param parms natural parameter list.
#' @param u named control vector (constant over the grid).
#' @param dfun disturbance function of time.
#' @param grid integration grid (steps = grid differences).
#' @param engine `deSolve` or `rk4_internal`.
#' @return matrix length(grid) x n_x.
#' @export
integrate_states <- function(model, x0, parms, u, dfun, grid, engine = "rk4_internal") {
  y0 <- unname(as.numeric(x0))
  if (length(grid) < 2L) return(matrix(y0, nrow = 1L))
  snames <- model$state_names
  f <- function(time, y) {
    names(y) <- snames
    as.numeric(model$rhs(time, y, u, dfun(time), parms))
  }
  if (identical(engine, "deSolve")) {
    func <- function(time, y, p) list(f(time, y))
    out <- deSolve::ode(y = y0, times = grid, func = func, parms = NULL, method = "rk4")
    Y <- unname(as.matrix(out[, -1, drop = FALSE]))
  } else if (identical(engine, "rk4_internal")) {
    Y <- rk4_grid(f = f, y0 = y0, grid = grid)
  } else {
    stop("unknown solver engine: ", engine)
  }
  Y
}

# evaluate the model outputs at every grid node; returns matrix n_grid x n_out
eval_outputs <- function(model, grid, Y, u, dfun, parms) {
  n_out <- model$n_out
  O <- vapply(seq_along(grid), function(r) {
    x <- stats::setNames(Y[r, ], model$state_names)
    as.numeric(model$outputs(grid[r], x, u, dfun(grid[r]), parms))
  }, numeric(n_out))
  O <- matrix(O, nrow = n_out)
  O <- t(O)
  colnames(O) <- model$output_names
  O
}

# control vector valid at time `time` from a plan table (column `time` = start)
plan_controls <- function(u_plan, time, control_names) {
  row <- max(which(u_plan$time <= time + 1e-9))
  stats::setNames(unlist(u_plan[row, control_names, with = FALSE], use.names = FALSE), control_names)
}

#' Simulate one unit over \[t0, t1\] with a control plan and discrete events
#'
#' Costs are integrated with the trapezoidal rule on the integration grid
#' (step `dt`), constraint violations as the time integral of the positive
#' part of each margin. Event costs are lumped. An optional terminal cost
#' `terminal_days` x (cost rate at t1) stands in for the cost-to-go beyond the
#' horizon.
#'
#' @param model a `dc_model`.
#' @param x0 initial state.
#' @param parms natural parameter list.
#' @param u_plan data.table with `time` (segment start) and control columns;
#'   the first row must start at or before `t0`.
#' @param dfun disturbance function.
#' @param t0,t1 simulation window.
#' @param dt integration step.
#' @param events_plan NULL or data.table(time, event); events act at the start
#'   of the segment that begins at `time`.
#' @param out_times times at which states and outputs are returned.
#' @param engine solver engine.
#' @param terminal_days multiplier of the terminal cost rate \[time\].
#' @param keep_traj keep the full grid trajectory.
#' @return list with `cost` (total), `cost_running`, `cost_events`,
#'   `cost_terminal`, `components` (named running cost by component),
#'   `viol` (named violation integrals), `viol_days`, `x_end`, `y_out`
#'   (data.table at `out_times`) and optionally `traj`.
#' @export
simulate_unit <- function(model, x0, parms, u_plan, dfun, t0, t1, dt,
                          events_plan = NULL, out_times = numeric(0),
                          engine = "rk4_internal", terminal_days = 0, keep_traj = FALSE) {
  cn <- model$control_names
  breaks <- c(t0, u_plan$time[u_plan$time > t0 + 1e-9 & u_plan$time < t1 - 1e-9])
  if (!is.null(events_plan) && nrow(events_plan) > 0) {
    et <- events_plan$time
    breaks <- c(breaks, et[et > t0 + 1e-9 & et < t1 - 1e-9])
  }
  breaks <- sort(unique(round(c(breaks, t1), 9)))
  out_times <- out_times[out_times >= t0 - 1e-9 & out_times <= t1 + 1e-9]

  x <- stats::setNames(as.numeric(x0), model$state_names)
  comp <- stats::setNames(numeric(length(model$cost_outputs)), model$cost_outputs)
  viol <- stats::setNames(numeric(length(model$constraint_outputs)), model$constraint_outputs)
  viol_days <- 0
  cost_events <- 0
  y_rows <- list()
  traj_rows <- list()
  O_last <- NULL
  u <- plan_controls(u_plan, time = t0, control_names = cn)

  s <- 1                                              # debugging line
  for (s in seq_len(length(breaks) - 1L)) {
    ta <- breaks[s]
    tb <- breaks[s + 1L]
    u  <- plan_controls(u_plan, time = ta, control_names = cn)
    if (!is.null(events_plan) && nrow(events_plan) > 0) {
      hit <- which(abs(events_plan$time - ta) < 1e-9)
      k <- hit[1]                                     # debugging line
      for (k in hit) {
        ev <- model$events[[events_plan$event[k]]]
        cost_events <- cost_events + ev$cost(x, parms, u, dfun(ta))
        x <- clamp_states(ev$apply(x, parms), model)
      }
    }
    grid <- make_grid(ta, tb, dt, extra = out_times)
    Y <- integrate_states(model, x0 = x, parms = parms, u = u, dfun = dfun, grid = grid, engine = engine)
    Y <- clamp_states(Y, model)
    O <- eval_outputs(model, grid = grid, Y = Y, u = u, dfun = dfun, parms = parms)
    j <- 1                                            # debugging line
    for (j in seq_along(comp)) comp[j] <- comp[j] + trapz(grid, O[, model$cost_outputs[j]])
    j <- 1                                            # debugging line
    for (j in seq_along(viol)) viol[j] <- viol[j] + trapz(grid, pmax(O[, model$constraint_outputs[j]], 0))
    if (length(viol) > 0) {
      any_v <- apply(O[, model$constraint_outputs, drop = FALSE] > 0, 1, any)
      viol_days <- viol_days + trapz(grid, as.numeric(any_v))
    }
    # outputs requested in (ta, tb]; t0 itself only in the first segment
    sel <- which(vapply(grid, function(g) any(abs(out_times - g) < 1e-9), logical(1)) &
                   (grid > ta + 1e-9 | (s == 1L & grid <= t0 + 1e-9)))
    if (length(sel) > 0) {
      M <- cbind(grid[sel], Y[sel, , drop = FALSE], O[sel, , drop = FALSE])
      colnames(M) <- c("time", model$state_names, model$output_names)
      y_rows[[length(y_rows) + 1L]] <- data.table::as.data.table(M)
    }
    if (keep_traj) {
      keep <- if (s == 1L) seq_along(grid) else seq_along(grid)[-1]
      M <- cbind(grid[keep], Y[keep, , drop = FALSE], O[keep, , drop = FALSE],
                 matrix(u, nrow = length(keep), ncol = length(u), byrow = TRUE))
      colnames(M) <- c("time", model$state_names, model$output_names, cn)
      traj_rows[[length(traj_rows) + 1L]] <- data.table::as.data.table(M)
    }
    x <- stats::setNames(Y[nrow(Y), ], model$state_names)
    O_last <- O[nrow(O), ]
  }

  cost_running  <- sum(comp)
  cost_terminal <- if (terminal_days > 0 && !is.null(O_last)) terminal_days * sum(O_last[model$cost_outputs]) else 0
  list(cost = cost_running + cost_events + cost_terminal,
       cost_running = cost_running, cost_events = cost_events, cost_terminal = cost_terminal,
       components = comp, viol = viol, viol_days = viol_days, x_end = x,
       y_out = if (length(y_rows) > 0) data.table::rbindlist(y_rows) else NULL,
       traj = if (keep_traj) data.table::rbindlist(traj_rows) else NULL)
}

# build a one-row (or multi-row) control plan table
make_u_plan <- function(times, U, control_names) {
  U <- matrix(as.numeric(U), ncol = length(control_names))
  M <- cbind(times, U)
  colnames(M) <- c("time", control_names)
  data.table::as.data.table(M)
}
