# =============================================================================
# schedules.R
# Open-loop schedules shared by the calendar baseline, the identifiability
# scenarios, the dither strategy and the attribution baseline.
# =============================================================================

#' Calendar schedule from the config
#'
#' Fixed recipe (`cfg$strategies$calendar$controls`, missing controls at the
#' model default) and periodic events
#' (`cfg$strategies$calendar$events$<event>$every_epochs`, `offset_epochs`).
#'
#' @param model a `dc_model`.
#' @param cfg scenario config.
#' @param n_epochs number of epochs in the season.
#' @return list(controls = named vector, events = data.table(epoch, event)),
#'   epochs counted from 0.
#' @export
calendar_schedule <- function(model, cfg, n_epochs) {
  cc <- cfg$strategies$calendar
  u <- model$control_default
  ov <- unlist(cc$controls)
  if (length(ov) > 0) {
    ov <- ov[names(ov) %in% names(u)]
    if (length(ov) > 0) u[names(ov)] <- as.numeric(ov)
  }
  ev_rows <- list()
  en <- names(cc$events)[1]                           # debugging line
  for (en in names(cc$events)) {
    if (!en %in% model$event_names) next
    e <- cc$events[[en]]
    first <- as.integer(e$offset_epochs %||% 0)
    if (first > n_epochs - 1) next
    ep <- seq(first, n_epochs - 1, by = as.integer(e$every_epochs))
    ev_rows[[en]] <- data.table::data.table(epoch = ep, event = en)
  }
  ev <- if (length(ev_rows) > 0) data.table::rbindlist(ev_rows) else
    data.table::data.table(epoch = integer(0), event = character(0))
  list(controls = u, events = ev)
}

#' Pseudo-random binary sequence of +-1 with a minimum hold
#' @param n length (epochs).
#' @param hold hold length (epochs).
#' @param seed seed.
#' @export
prbs_signal <- function(n, hold = 1, seed = 1L) {
  n_blocks <- ceiling(n / hold)
  s <- with_seed(seed, sample(c(-1, 1), n_blocks, replace = TRUE))
  rep(s, each = hold)[seq_len(n)]
}

#' Single-tone multisine value
#' @param k epoch index.
#' @param period period in epochs.
#' @param phase phase \[rad\].
#' @export
multisine_signal <- function(k, period, phase = 0) sin(2 * pi * k / period + phase)

# open-loop dithered control table around a base recipe (one row per epoch)
dither_plan <- function(model, base_u, n_epochs, epoch_days, amplitude, hold, seed) {
  U <- matrix(base_u, nrow = n_epochs, ncol = model$n_u, byrow = TRUE)
  j <- 1                                              # debugging line
  for (j in seq_len(model$n_u)) {
    s <- prbs_signal(n_epochs, hold = hold, seed = derive_seed(seed, "dither_plan", j))
    U[, j] <- clamp(base_u[j] + amplitude * model$controls$step[j] * s,
                    model$controls$lower[j], model$controls$upper[j])
  }
  make_u_plan(times = (seq_len(n_epochs) - 1) * epoch_days, U = U, control_names = model$control_names)
}
