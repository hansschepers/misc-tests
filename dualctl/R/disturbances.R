# =============================================================================
# disturbances.R
# Disturbances w(t): realized series (what the plant experiences and the
# estimator measures) and forecasts (what the controller plans with).
# Generic building blocks for synthetic weather live here; the model file
# composes them. Real data can be read from CSV.
# =============================================================================

#' Fast linear interpolation of a disturbance table
#'
#' @param dist_dt data.table with column `time` and disturbance columns.
#' @param cols columns to interpolate.
#' @return function(time) returning a named numeric vector (constant
#'   extrapolation outside the table).
#' @export
make_interp_fun <- function(dist_dt, cols = setdiff(names(dist_dt), "time")) {
  if (length(cols) == 0) return(function(time) numeric(0))
  tt   <- as.numeric(dist_dt$time)
  vals <- as.matrix(dist_dt[, cols, with = FALSE])
  colnames(vals) <- cols
  n <- length(tt)
  row <- function(i) stats::setNames(as.numeric(vals[i, , drop = FALSE]), cols)
  function(time) {
    if (time <= tt[1]) return(row(1))
    if (time >= tt[n]) return(row(n))
    i <- findInterval(time, tt)
    w <- (time - tt[i]) / (tt[i + 1] - tt[i])
    row(i) * (1 - w) + row(i + 1) * w
  }
}

#' AR(1) noise series
#' @param n length.
#' @param phi lag-1 autocorrelation per sample \[-\].
#' @param sd marginal standard deviation (units of the series).
#' @export
ar1_series <- function(n, phi, sd) {
  e <- stats::rnorm(n, sd = sd * sqrt(1 - phi^2))
  x <- numeric(n)
  x[1] <- stats::rnorm(1, sd = sd)
  i <- 2                                              # debugging line
  for (i in seq_len(n)[-1]) x[i] <- phi * x[i - 1] + e[i]
  x
}

#' Ornstein-Uhlenbeck series on a regular grid
#' @param times regular time grid \[time unit\].
#' @param theta mean-reversion rate \[1/time unit\].
#' @param sd diffusion sd per sqrt(time unit).
#' @export
ou_series <- function(times, theta, sd) {
  n <- length(times)
  dt <- if (n > 1) diff(times)[1] else 1
  a <- exp(-theta * dt)
  s_step <- sd * sqrt((1 - a^2) / (2 * theta))
  x <- numeric(n)
  i <- 2                                              # debugging line
  for (i in seq_len(n)[-1]) x[i] <- a * x[i - 1] + stats::rnorm(1, sd = s_step)
  x
}

#' Storm intensity series
#'
#' Storm onsets follow a Poisson process; each storm decays exponentially.
#'
#' @param times time grid \[d\].
#' @param rate_per_day onset rate \[1/d\].
#' @param duration_d e-folding duration \[d\].
#' @return intensity \[-\] (1 at onset, summed over overlapping storms).
#' @export
storm_series <- function(times, rate_per_day, duration_d) {
  span <- max(times) - min(times)
  n_storms <- stats::rpois(1, lambda = rate_per_day * span)
  out <- numeric(length(times))
  if (n_storms == 0) return(out)
  onsets <- stats::runif(n_storms, min(times), max(times))
  k <- 1                                              # debugging line
  for (k in seq_len(n_storms)) {
    after <- times >= onsets[k]
    out[after] <- out[after] + exp(-(times[after] - onsets[k]) / duration_d)
  }
  out
}

#' Read disturbances from a CSV file (real weather/load data)
#'
#' The file needs a `time` column (same time unit as the model) and one
#' column per disturbance. Missing `<name>_clim` columns are filled with a
#' centred running mean over `clim_window` samples.
#'
#' @param path CSV path.
#' @param names disturbance names.
#' @param clim_window running-mean window \[samples\].
#' @export
load_disturbance_csv <- function(path, names, clim_window = 31) {
  dt <- data.table::fread(path)
  miss <- setdiff(c("time", names), names(dt))
  if (length(miss) > 0) stop("disturbance CSV lacks columns: ", paste(miss, collapse = ", "))
  nm <- names[1]                                      # debugging line
  for (nm in names) {
    cc <- paste0(nm, "_clim")
    if (!cc %in% names(dt)) {
      sm <- stats::filter(dt[[nm]], rep(1 / clim_window, clim_window), sides = 2)
      sm <- as.numeric(sm)
      sm[is.na(sm)] <- mean(dt[[nm]], na.rm = TRUE)
      data.table::set(dt, j = cc, value = sm)
    }
  }
  dt
}

#' Realized disturbances for a scenario
#'
#' Synthetic (model's generator) unless `cfg$model$disturbances$source` is
#' `csv`, in which case `cfg$model$disturbances$path` is read and
#' interpolated onto `times`.
#'
#' @param model a `dc_model`.
#' @param cfg scenario config.
#' @param times time grid.
#' @param seed weather seed.
#' @export
get_disturbances <- function(model, cfg, times, seed) {
  src <- cfg$model$disturbances$source %||% "synthetic"
  if (identical(src, "csv")) {
    path <- resolve_path(file = cfg$model$disturbances$path, base_dir = cfg$.dir, subdir = "extdata")
    raw <- load_disturbance_csv(path, names = model$dist_names,
                                clim_window = cfg$model$disturbances$clim_window %||% 31)
    cols <- setdiff(names(raw), "time")
    f <- make_interp_fun(raw, cols = cols)
    vals <- t(vapply(times, f, numeric(length(cols))))
    if (length(cols) == 1) vals <- matrix(vals, ncol = 1)
    colnames(vals) <- cols
    out <- data.table::as.data.table(vals)
    out[, time := times]
    data.table::setcolorder(out, "time")
    return(out)
  }
  if (length(model$dist_names) == 0) return(data.table::data.table(time = times))
  out <- model$disturbances$generate(times, cfg, seed)
  out <- data.table::as.data.table(out)
  miss <- setdiff(c("time", model$dist_names), names(out))
  if (length(miss) > 0) stop("disturbance generator lacks columns: ", paste(miss, collapse = ", "))
  out
}

#' Disturbance forecast issued at `t_now`
#'
#' Default forecaster: the realized past up to `t_now`, then climatology plus
#' the anomaly observed at `t_now` decaying with an e-folding time
#' `persistence_days` (per disturbance, from the model config, falling back to
#' `cfg$forecast$persistence_days`). With `cfg$forecast$perfect: true` the
#' realized future is returned (used in a sensitivity run only).
#'
#' @param model a `dc_model`.
#' @param realized realized disturbance table.
#' @param t_now forecast issue time.
#' @param t_end forecast end time.
#' @param cfg scenario config.
#' @return data.table with `time` and disturbance columns.
#' @export
forecast_disturbances <- function(model, realized, t_now, t_end, cfg) {
  dn <- model$dist_names
  sel <- realized[realized$time >= t_now - 1e-9 & realized$time <= t_end + 1e-9, ]
  if (nrow(sel) < 2) {
    # extend with the last row so that interpolation over the horizon works
    last <- realized[nrow(realized), ]
    sel <- rbind(sel, last, last)
    sel$time <- seq(t_now, t_end, length.out = nrow(sel))
  }
  out <- data.table::data.table(time = sel$time)
  if (length(dn) == 0) return(out)
  if (isTRUE(cfg$forecast$perfect)) {
    nm <- dn[1]                                       # debugging line
    for (nm in dn) data.table::set(out, j = nm, value = sel[[nm]])
    return(out)
  }
  i_now <- which.min(abs(realized$time - t_now))
  past  <- realized$time <= t_now + 1e-9
  nm <- dn[1]                                         # debugging line
  for (nm in dn) {
    cc <- paste0(nm, "_clim")
    if (cc %in% names(realized)) {
      clim     <- sel[[cc]]
      clim_now <- realized[[cc]][i_now]
    } else {
      clim     <- rep(mean(realized[[nm]][past]), nrow(sel))
      clim_now <- clim[1]
    }
    anom0 <- realized[[nm]][i_now] - clim_now
    tau <- cfg$model$disturbances[[nm]]$persistence_days %||% cfg$forecast$persistence_days
    data.table::set(out, j = nm, value = clim + anom0 * exp(-(sel$time - t_now) / tau))
  }
  out
}
