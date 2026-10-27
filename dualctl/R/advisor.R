# =============================================================================
# advisor.R
# Each epoch: which control changes are worth making, by how much, in which
# order? Under the current posterior (common random draws), every candidate
# single change (a step of +-k * step on one continuous control, within the
# rate limit, or triggering an event now) is compared with holding the
# current recipe over the controller horizon. Ranking key: expected saving
# per unit implementation cost. The sequence is built greedily: adopt the
# top change, re-evaluate all remaining changes given it, re-rank (the
# limiting factor moves once a constraint is relieved). Because the advisor
# runs after every refit, rankings change as the posterior sharpens.
# =============================================================================

#' Rank control changes by marginal value per unit cost
#'
#' @param model fitted `dc_model`.
#' @param cfg scenario config (`cfg$advisor`).
#' @param posterior posterior list of the unit (mean, cov, sample).
#' @param x_mean state estimate.
#' @param u_prev current recipe.
#' @param forecast_dt forecast table.
#' @param t0 decision time.
#' @param epoch epoch index.
#' @param last_event_epoch named vector of last event epochs.
#' @param seed seed of the posterior draws.
#' @return list(ranking = first-round table with `rank`, sequence = greedy
#'   sequence with `order`). Savings in currency over the horizon (incl.
#'   terminal cost); `saving_lo`/`saving_hi` are 5%/95% posterior quantiles.
#' @export
advise <- function(model, cfg, posterior, x_mean, u_prev, forecast_dt, t0, epoch = 0,
                   last_event_epoch = NULL, seed = 1L) {
  ac <- cfg$advisor
  ctrl <- Controller$new(model, cfg = cfg)
  H <- ctrl$H
  ct <- model$controls
  cn <- model$control_names
  n_u <- model$n_u
  u_prev <- stats::setNames(as.numeric(u_prev[cn]), cn)
  thetas <- with_seed(seed, posterior$sample(ac$n_samples))
  parms_list <- lapply(seq_len(nrow(thetas)), function(m) phi_to_parms(model, phi = thetas[m, ]))
  dfun <- make_interp_fun(forecast_dt, cols = model$dist_names)
  J_vec <- function(U, ev) {
    vapply(parms_list, function(p) ctrl$evaluate_U(U, ev = ev, x0 = x_mean, parms = p, dfun = dfun, t0 = t0)$J,
           numeric(1))
  }
  ev_now <- Filter(function(o) o$tau == 0, ctrl$event_options(epoch, last_event_epoch = last_event_epoch))
  ev_now <- vapply(ev_now, function(o) o$name, character(1))
  summarise <- function(control, kind, from, to, sav, change_cost) {
    data.table::data.table(control = control, kind = kind, from = from, to = to, step = to - from,
                           saving_mean = mean(sav),
                           saving_lo = stats::quantile(sav, 0.05, names = FALSE),
                           saving_hi = stats::quantile(sav, 0.95, names = FALSE),
                           prob_positive = mean(sav > 0), change_cost = change_cost)
  }
  base_u <- u_prev
  base_ev <- NULL
  ranking <- NULL
  seq_rows <- list()
  r_idx <- 1                                          # debugging line
  for (r_idx in seq_len(ac$max_changes)) {
    U_base <- matrix(base_u, nrow = H, ncol = n_u, byrow = TRUE, dimnames = list(NULL, cn))
    J0 <- J_vec(U_base, ev = base_ev)
    rows <- list()
    j <- 1                                            # debugging line
    for (j in seq_len(n_u)) {
      lo <- max(ct$lower[j], u_prev[j] - ct$rate[j])
      hi <- min(ct$upper[j], u_prev[j] + ct$rate[j])
      k <- 1                                          # debugging line
      for (k in seq_len(ac$n_steps)) {
        sg <- -1                                      # debugging line
        for (sg in c(-1, 1)) {
          new <- clamp(base_u[j] + sg * k * ct$step[j], lo, hi)
          if (abs(new - base_u[j]) < 1e-12) next
          U <- U_base
          U[, j] <- new
          rows[[length(rows) + 1L]] <- summarise(cn[j], kind = "continuous", from = base_u[j], to = new,
                                                 sav = J0 - J_vec(U, ev = base_ev),
                                                 change_cost = ct$change_cost[j])
        }
      }
    }
    if (is.null(base_ev)) {
      en <- ev_now[1]                                 # debugging line
      for (en in ev_now) {
        rows[[length(rows) + 1L]] <- summarise(en, kind = "event", from = 0, to = 1,
                                               sav = J0 - J_vec(U_base, ev = list(name = en, tau = 0)),
                                               change_cost = model$events[[en]]$change_cost)
      }
    }
    if (length(rows) == 0) break
    cand <- data.table::rbindlist(rows)
    cand[, score := saving_mean / change_cost]
    best <- cand[order(-saving_mean)][, .SD[1], by = control]
    best <- best[order(-score)]
    best[, rank := seq_len(.N)]
    if (r_idx == 1) ranking <- data.table::copy(best)[, epoch := epoch]
    top <- best[1]
    if (top$saving_mean <= ac$min_saving) break
    seq_rows[[length(seq_rows) + 1L]] <- data.table::copy(top)[, `:=`(order = r_idx, epoch = epoch)]
    if (top$kind == "continuous") {
      base_u[top$control] <- top$to
    } else {
      base_ev <- list(name = top$control, tau = 0)
    }
  }
  list(ranking = ranking,
       sequence = if (length(seq_rows) > 0) data.table::rbindlist(seq_rows) else NULL)
}
