# =============================================================================
# experiment.R
# One season for one cohort of units under one method.
# A method (cfg$methods$<name>) assigns a strategy to each arm:
#   production arm: runs the champion recipe (usually certainty-equivalent
#                   MPC on the shared posterior), routine sensors
#   probe arm     : the probing strategy plus intensive sensors (extra
#                   measured quantities, lower noise)
# Units are randomised to arms within cohorts (seeded, so identical across
# methods with the same probe-arm size); optional crossover swaps the arms.
# `learn: false` freezes the parameters at the prior (states still tracked).
# Epoch loop: forecast -> act (per unit) -> events -> plant -> observations
# -> estimator predict/update in time order -> records.
# =============================================================================

#' Randomise units to production and probe arms
#'
#' @param n_units number of units.
#' @param n_probe number of probe units.
#' @param seed seed.
#' @param cohorts optional cohort label per unit; probe slots are spread over
#'   cohorts proportionally (largest remainder) and randomised within each.
#' @return character vector of arms.
#' @export
assign_arms <- function(n_units, n_probe, seed, cohorts = NULL) {
  arms <- rep("production", n_units)
  n_probe <- min(max(0, n_probe), n_units)
  if (n_probe == 0) return(arms)
  if (is.null(cohorts)) cohorts <- rep(1, n_units)
  cs <- split(seq_len(n_units), cohorts)
  share <- n_probe * lengths(cs) / n_units
  k <- floor(share)
  rem <- n_probe - sum(k)
  if (rem > 0) {
    top <- order(-(share - k))[seq_len(rem)]
    k[top] <- k[top] + 1
  }
  with_seed(seed, {
    g <- 1                                            # debugging line
    for (g in seq_along(cs)) {
      idx <- cs[[g]]
      if (k[g] > 0) {
        pick <- idx[sample.int(length(idx), k[g])]
        arms[pick] <- "probe"
      }
    }
  })
  arms
}

# posterior-like list from the prior (for methods without an estimator)
prior_posterior <- function(model) {
  S <- diag(model$phi_prior_sd^2, nrow = model$n_p)
  dimnames(S) <- list(model$phi_names, model$phi_names)
  mu <- model$phi_prior_mean
  list(mean = mu, cov = S, sd = model$phi_prior_sd,
       x_mean = stats::setNames(model$states$init, model$state_names),
       x_cov = diag(pmax(model$states$init_sd, 1e-12)^2, nrow = model$n_x),
       sample = function(n) rmvn(n, mean = mu, cov = S))
}

#' Run one season
#'
#' @param cfg scenario config.
#' @param method method name (key of `cfg$methods`).
#' @param seed seed (weather, population, noise and assignment derive from it).
#' @param models optional list(fit, truth) of loaded models.
#' @param advisor run the advisor on the first unit of `cfg$experiment$advisor_arm`.
#' @param keep_obs keep true and noisy observations.
#' @return list(epochs, posterior, advisor, observations, arms, plant, dist,
#'   method, seed).
#' @export
run_season <- function(cfg, method, seed, models = NULL, advisor = FALSE, keep_obs = FALSE) {
  if (is.null(models)) models <- list(fit = load_model(cfg, variant = "fitted"),
                                      truth = load_model(cfg, variant = "truth"))
  fit <- models$fit
  tru <- models$truth
  mc <- cfg$methods[[method]]
  if (is.null(mc)) stop("method not configured: ", method)
  E <- cfg_num(cfg$season$epoch_days, "season$epoch_days")
  K <- as.integer(floor(cfg$season$length_days / E))
  H <- as.integer(cfg$controller$horizon_epochs)
  n_units <- as.integer(cfg$experiment$n_units)
  n_probe <- as.integer(mc$n_probe %||% cfg$experiment$n_probe)
  level <- cfg$evaluation$coverage_level %||% 0.9
  arms0 <- assign_arms(n_units, n_probe = n_probe, seed = derive_seed(seed, "assign"),
                       cohorts = unlist(cfg$experiment$cohorts))
  cross_k <- if (isTRUE(cfg$experiment$crossover) && n_probe > 0) floor(K * cfg$experiment$crossover_at) else Inf
  sets <- cfg$experiment$sensor_sets

  plant <- Plant$new(tru, cfg = cfg, seed = seed, n_units = n_units)
  strat <- lapply(seq_len(n_units), function(i) {
    list(production = make_strategy(mc$production, model = fit, cfg = cfg, unit = i,
                                    seed = derive_seed(seed, method, "production", i), model_truth = tru),
         probe = make_strategy(mc$probe, model = fit, cfg = cfg, unit = i,
                               seed = derive_seed(seed, method, "probe", i), model_truth = tru))
  })
  used <- unique(c("production", if (n_probe > 0) "probe"))
  needs_est <- any(vapply(strat, function(s) any(vapply(used, function(a) s[[a]]$uses_posterior, logical(1))),
                          logical(1)))
  est <- if (needs_est) make_estimator(fit, cfg = cfg, n_units = n_units, seed = derive_seed(seed, "estimator"),
                                       learn_params = isTRUE(mc$learn)) else NULL
  prior_post <- prior_posterior(fit)

  u_prev <- rep(list(fit$control_default), n_units)
  last_ev <- rep(list(stats::setNames(rep(-Inf, length(fit$event_names)), fit$event_names)), n_units)
  last_cost_rate <- rep(NA_real_, n_units)
  last_obs <- vector("list", n_units)
  adv_unit <- NA_integer_
  if (advisor && isTRUE(cfg$advisor$enabled) && !is.null(est)) {
    adv_unit <- which(arms0 == (cfg$experiment$advisor_arm %||% "production"))[1]
    if (is.na(adv_unit)) adv_unit <- 1L
  }

  rec_ep <- list(); rec_post <- list(); rec_rank <- list(); rec_seq <- list(); rec_obs <- list()
  post_rows <- function(arms_now, ep_idx, time_now) {
    pt <- posterior_table(est, level = level, truth_unit = plant$unit_phi, truth_pop = plant$pop_phi)
    pt[, `:=`(seed = seed, method = method, epoch = ep_idx, time = time_now,
              arm = c("shared", arms_now)[pt$unit + 1L])]
    pt
  }
  if (!is.null(est)) rec_post[[1]] <- post_rows(arms0, ep_idx = 0L, time_now = 0)

  k <- 1                                              # debugging line
  for (k in seq_len(K)) {
    ep <- k - 1L
    t_k <- ep * E
    t_k1 <- t_k + E
    arms <- if (ep >= cross_k) ifelse(arms0 == "probe", "production", "probe") else arms0
    set_of <- vapply(arms, function(a) sets[[a]], character(1))
    fc <- forecast_disturbances(fit, realized = plant$dist, t_now = t_k, t_end = t_k + H * E, cfg = cfg)
    u_now <- vector("list", n_units)
    ev_now <- vector("list", n_units)
    ev_cost <- numeric(n_units)
    info_now <- vector("list", n_units)

    i <- 1                                            # debugging line
    for (i in seq_len(n_units)) {
      s <- strat[[i]][[arms[i]]]
      post <- if (!is.null(est)) est$posterior(i) else prior_post
      st <- list(unit = i, epoch = ep, n_epochs = K, season_frac = t_k / (K * E),
                 x_mean = post$x_mean, x_cov = post$x_cov, u_prev = u_prev[[i]],
                 last_event_epoch = last_ev[[i]], last_cost_rate = last_cost_rate[i],
                 last_obs = last_obs[[i]], sensor_set = set_of[i], arm = arms[i],
                 seed = derive_seed(seed, method, i, ep))
      if (isTRUE(s$needs_truth)) st$truth <- list(x = plant$x[[i]], parms = plant$unit_parms[[i]])
      if (!is.na(adv_unit) && i == adv_unit) {
        adv <- advise(fit, cfg = cfg, posterior = post, x_mean = post$x_mean, u_prev = u_prev[[i]],
                      forecast_dt = fc, t0 = t_k, epoch = ep, last_event_epoch = last_ev[[i]],
                      seed = derive_seed(seed, "advisor", ep))
        if (!is.null(adv$ranking)) rec_rank[[length(rec_rank) + 1L]] <- adv$ranking[, unit := i]
        if (!is.null(adv$sequence)) rec_seq[[length(rec_seq) + 1L]] <- adv$sequence[, unit := i]
      }
      act <- s$act(st, post, fc, t_k)
      u_i <- enforce_control_limits(act$u, u_prev = u_prev[[i]], model = fit)
      evs <- unique(act$events[act$events %in% fit$event_names])
      evs <- evs[vapply(evs, function(en) ep - last_ev[[i]][[en]] >= fit$events[[en]]$min_interval_epochs,
                        logical(1))]
      en <- evs[1]                                    # debugging line
      for (en in evs) {
        ev_cost[i] <- ev_cost[i] + plant$apply_event(i, event = en, time = t_k, u = u_i)
        if (!is.null(est)) est$apply_event(i, event = en, time = t_k, u = u_i, dfun = plant$dfun)
        last_ev[[i]][[en]] <- ep
      }
      u_now[[i]] <- u_i
      ev_now[[i]] <- evs
      inf <- act$info
      keep <- vapply(inf, function(v) is.numeric(v) && length(v) == 1, logical(1))
      info_now[[i]] <- if (length(inf) > 0 && any(keep)) {
        stats::setNames(as.list(as.numeric(unlist(inf[keep]))), paste0("info_", names(inf)[keep]))
      } else list()
    }

    # plant over the epoch, with true outputs at each unit's sampling times
    obs_t <- lapply(seq_len(n_units), function(i) {
      iv <- fit$sensor_sets[[set_of[i]]]$interval
      tt <- seq(iv, K * E, by = iv)
      tt[tt > t_k + 1e-9 & tt <= t_k1 + 1e-9]
    })
    res <- lapply(seq_len(n_units), function(i) plant$advance(i, t0 = t_k, t1 = t_k1, u = u_now[[i]],
                                                              out_times = obs_t[[i]]))
    y_obs <- lapply(seq_len(n_units), function(i) {
      yo <- res[[i]]$y_out
      if (is.null(yo) || nrow(yo) == 0) return(NULL)
      outs <- fit$sensor_sets[[set_of[i]]]$outputs
      Y <- t(vapply(seq_len(nrow(yo)), function(r) {
        plant$observe(i, time = yo$time[r], sensor_set = set_of[i],
                      y_true = unlist(yo[r, outs, with = FALSE]))
      }, numeric(length(outs))))
      Y <- matrix(Y, nrow = nrow(yo), ncol = length(outs), dimnames = list(NULL, outs))
      list(time = yo$time, Y = Y)
    })

    if (!is.null(est)) {
      tg <- sort(unique(unlist(obs_t)))
      tau_prev <- t_k
      tau <- tg[1]                                    # debugging line
      for (tau in tg) {
        est$predict(tau_prev, tau, u_list = u_now, dfun = plant$dfun)
        y_list <- lapply(seq_len(n_units), function(i) {
          yi <- y_obs[[i]]
          if (is.null(yi)) return(NULL)
          r <- which(abs(yi$time - tau) < 1e-9)
          if (length(r) == 0) return(NULL)
          stats::setNames(as.numeric(yi$Y[r[1], , drop = FALSE]), colnames(yi$Y))
        })
        est$update(tau, y_list = y_list, sensor_sets = set_of, u_list = u_now, dfun = plant$dfun)
        tau_prev <- tau
      }
      if (tau_prev < t_k1 - 1e-9) est$predict(tau_prev, t_k1, u_list = u_now, dfun = plant$dfun)
    }

    i <- 1                                            # debugging line
    for (i in seq_len(n_units)) {
      r <- res[[i]]
      row <- c(list(seed = seed, method = method, unit = i, arm = arms[i], epoch = ep, time = t_k,
                    time_end = t_k1, events = paste(ev_now[[i]], collapse = "+"),
                    n_events = length(ev_now[[i]]), cost_running = r$cost_running,
                    cost_events = ev_cost[i], cost_total = r$cost_running + ev_cost[i],
                    viol_days = r$viol_days),
               stats::setNames(as.list(u_now[[i]]), paste0("u_", names(u_now[[i]]))),
               stats::setNames(as.list(r$components), paste0("comp_", names(r$components))),
               stats::setNames(as.list(r$viol), paste0("viol_", names(r$viol))),
               stats::setNames(as.list(r$x_end), paste0("x_", names(r$x_end))),
               info_now[[i]])
      rec_ep[[length(rec_ep) + 1L]] <- data.table::as.data.table(row)
      last_cost_rate[i] <- r$cost_running / E
      if (!is.null(y_obs[[i]])) {
        yi <- y_obs[[i]]
        last_obs[[i]] <- stats::setNames(as.numeric(yi$Y[nrow(yi$Y), , drop = FALSE]), colnames(yi$Y))
        if (keep_obs) {
          noisy <- data.table::as.data.table(yi$Y)
          data.table::setnames(noisy, paste0("obs_", colnames(yi$Y)))
          ob <- cbind(data.table::copy(r$y_out), noisy)
          arm_i <- arms[i]
          ob[, `:=`(seed = seed, method = method, unit = i, arm = arm_i)]
          rec_obs[[length(rec_obs) + 1L]] <- ob
        }
      }
    }
    u_prev <- u_now
    if (!is.null(est)) rec_post[[length(rec_post) + 1L]] <- post_rows(arms, ep_idx = ep + 1L, time_now = t_k1)
  }

  list(epochs = data.table::rbindlist(rec_ep, fill = TRUE),
       posterior = if (length(rec_post) > 0) data.table::rbindlist(rec_post, fill = TRUE) else NULL,
       advisor = list(ranking = data.table::rbindlist(rec_rank, fill = TRUE),
                      sequence = data.table::rbindlist(rec_seq, fill = TRUE)),
       observations = if (length(rec_obs) > 0) data.table::rbindlist(rec_obs, fill = TRUE) else NULL,
       arms = arms0,
       plant = list(x0 = plant$x0, unit_parms = plant$unit_parms, unit_phi = plant$unit_phi,
                    pop_phi = plant$pop_phi),
       dist = plant$dist, method = method, seed = seed)
}
