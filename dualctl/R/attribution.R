# =============================================================================
# attribution.R
# Which adopted changes produced the savings? The season's controls are
# replayed open-loop on the true plant (same weather, same initial state)
# with each adopted change switched on or off (0/1 gates). Savings of a gate
# set S: v(S) = J(baseline) - J(baseline with the changes in S). Shapley
# values split v(all) fairly over the changes (exact enumeration for few
# gates, permutation Monte Carlo otherwise). With space = "log" the game is
# played on log J (multiplicative effects become additive) and the shares are
# mapped back to currency. Stretch: Aumann-Shapley path attribution for the
# continuous controls, integrating finite-difference sensitivities of J
# along the straight path from baseline to adopted trajectories.
# Caveat: open-loop replay ignores feedback (with a change switched off, the
# controller would have acted differently).
# =============================================================================

#' Shapley values of a cooperative game
#'
#' @param value_fun function of a logical vector (gates on) returning v(S).
#' @param players player names.
#' @param n_perm permutations for the Monte Carlo estimate.
#' @param seed seed of the permutations.
#' @param exact_max_players exact enumeration up to this many players.
#' @return data.table(player, phi, se, method).
#' @export
shapley_values <- function(value_fun, players, n_perm = 200, seed = 1L, exact_max_players = 8) {
  K <- length(players)
  cache <- new.env(parent = emptyenv())
  v <- function(S) {
    key <- paste0("k", paste(as.integer(S), collapse = ""))
    if (exists(key, envir = cache, inherits = FALSE)) return(get(key, envir = cache))
    val <- value_fun(S)
    assign(key, val, envir = cache)
    val
  }
  if (K == 0) return(data.table::data.table(player = character(0), phi = numeric(0), se = numeric(0),
                                            method = character(0)))
  if (K <= exact_max_players) {
    phi <- numeric(K)
    j <- 1                                            # debugging line
    for (j in seq_len(K)) {
      others <- setdiff(seq_len(K), j)
      m <- 0                                          # debugging line
      for (m in 0:(2^(K - 1) - 1)) {
        S <- logical(K)
        if (length(others) > 0) {
          S[others] <- bitwAnd(as.integer(m), as.integer(2^(seq_along(others) - 1))) > 0
        }
        s <- sum(S)
        w <- factorial(s) * factorial(K - s - 1) / factorial(K)
        S1 <- S
        S1[j] <- TRUE
        phi[j] <- phi[j] + w * (v(S1) - v(S))
      }
    }
    return(data.table::data.table(player = players, phi = phi, se = 0, method = "exact"))
  }
  perms <- with_seed(seed, lapply(seq_len(n_perm), function(r) sample.int(K)))
  contrib <- matrix(0, nrow = n_perm, ncol = K)
  r <- 1                                              # debugging line
  for (r in seq_len(n_perm)) {
    S <- logical(K)
    v_prev <- v(S)
    j <- perms[[r]][1]                                # debugging line
    for (j in perms[[r]]) {
      S[j] <- TRUE
      v_new <- v(S)
      contrib[r, j] <- v_new - v_prev
      v_prev <- v_new
    }
  }
  data.table::data.table(player = players, phi = colMeans(contrib),
                         se = apply(contrib, 2, stats::sd) / sqrt(n_perm), method = "permutation")
}

#' Aumann-Shapley path attribution
#'
#' AS_j = integral over s in (0, 1) of dJ/dalpha_j at alpha = (s, ..., s),
#' midpoint rule with central finite differences in the gate alpha_j.
#' sum_j AS_j = J(1) - J(0) up to quadrature error.
#'
#' @param J_alpha function of a gate vector alpha in \[0, 1\]^n.
#' @param n_players number of continuous gates.
#' @param n_quad quadrature points.
#' @param h finite-difference step in alpha.
#' @export
aumann_shapley <- function(J_alpha, n_players, n_quad = 8, h = 0.02) {
  s <- (seq_len(n_quad) - 0.5) / n_quad
  G <- matrix(0, nrow = n_quad, ncol = n_players)
  q <- 1                                              # debugging line
  for (q in seq_len(n_quad)) {
    j <- 1                                            # debugging line
    for (j in seq_len(n_players)) {
      ap <- rep(s[q], n_players)
      am <- ap
      ap[j] <- min(1, s[q] + h)
      am[j] <- max(0, s[q] - h)
      G[q, j] <- (J_alpha(ap) - J_alpha(am)) / (ap[j] - am[j])
    }
  }
  colMeans(G)
}

# events of an epoch table as data.table(time, event)
epoch_events <- function(ep, E) {
  rows <- list()
  r <- 1                                              # debugging line
  for (r in seq_len(nrow(ep))) {
    if (ep$n_events[r] > 0) {
      evs <- strsplit(ep$events[r], "+", fixed = TRUE)[[1]]
      rows[[length(rows) + 1L]] <- data.table::data.table(time = ep$time[r], event = evs)
    }
  }
  if (length(rows) == 0) return(data.table::data.table(time = numeric(0), event = character(0)))
  data.table::rbindlist(rows)
}

#' Attribute a season's savings to the adopted changes
#'
#' Baseline: the calendar schedule (`cfg$strategies$calendar`). Adopted: the
#' controls and events a unit actually ran in `res`.
#'
#' @param res result of [run_season()].
#' @param unit_idx unit to analyse.
#' @param cfg scenario config.
#' @param models list(fit, truth); loaded if NULL.
#' @param gate_by `control` (one gate per control and event type) or
#'   `change` (one gate per adopted step change of each continuous control,
#'   one per event type).
#' @param space `linear` or `log`.
#' @param seed seed for permutation sampling.
#' @return list(shapley, aumann, J_base, J_adopted, total_saving, gate_by, space).
#' @export
season_attribution <- function(res, unit_idx, cfg, models = NULL, gate_by = NULL, space = NULL,
                               seed = 1L) {
  if (is.null(models)) models <- list(fit = load_model(cfg, variant = "fitted"),
                                      truth = load_model(cfg, variant = "truth"))
  ac <- cfg$attribution
  gate_by <- gate_by %||% ac$gate_by
  space <- space %||% ac$space
  tru <- models$truth
  E <- cfg$season$epoch_days
  ep <- res$epochs[res$epochs[["unit"]] == unit_idx, ]
  ep <- ep[order(ep$epoch), ]
  K <- nrow(ep)
  cn <- tru$control_names
  n_u <- length(cn)
  times <- ep$time
  U_ad <- as.matrix(ep[, paste0("u_", cn), with = FALSE])
  colnames(U_ad) <- cn
  cal <- calendar_schedule(tru, cfg = cfg, n_epochs = K)
  U_base <- matrix(cal$controls[cn], nrow = K, ncol = n_u, byrow = TRUE, dimnames = list(NULL, cn))
  ev_ad <- epoch_events(ep, E = E)
  ev_base <- data.table::data.table(time = cal$events$epoch * E, event = cal$events$event)
  ev_types <- sort(unique(c(ev_ad$event, ev_base$event)))
  dfun <- make_interp_fun(res$dist, cols = tru$dist_names)
  x0 <- res$plant$x0[[unit_idx]]
  parms <- res$plant$unit_parms[[unit_idx]]
  dt <- ac$dt %||% cfg$solver$dt_inner
  replay <- function(U, ev) {
    simulate_unit(tru, x0 = x0, parms = parms, u_plan = make_u_plan(times = times, U = U, control_names = cn),
                  dfun = dfun, t0 = 0, t1 = K * E, dt = dt, events_plan = ev,
                  engine = cfg$solver$inner_engine)$cost
  }
  mix_events <- function(on_types) {
    rbind(ev_base[!ev_base$event %in% on_types, ], ev_ad[ev_ad$event %in% on_types, ])
  }

  if (identical(gate_by, "change")) {
    # one player per adopted step change of each continuous control
    ch <- list()
    j <- 1                                            # debugging line
    for (j in seq_len(n_u)) {
      prev <- c(U_base[1, j], U_ad[-K, j])
      dlt <- U_ad[, j] - prev
      idx <- which(abs(dlt) > 1e-9)
      if (length(idx) > 0) ch[[length(ch) + 1L]] <- data.table::data.table(control = cn[j], epoch_row = idx, delta = dlt[idx])
    }
    ch <- if (length(ch) > 0) data.table::rbindlist(ch) else
      data.table::data.table(control = character(0), epoch_row = integer(0), delta = numeric(0))
    players <- c(sprintf("%s@%d", ch$control, ep$epoch[ch$epoch_row]), ev_types)
    n_ch <- nrow(ch)
    build <- function(S) {
      U <- U_base
      j <- 1                                          # debugging line
      for (j in seq_len(n_u)) {
        sel <- which(ch$control == cn[j] & S[seq_len(n_ch)])
        steps <- numeric(K)
        steps[ch$epoch_row[sel]] <- ch$delta[sel]
        U[, j] <- clamp(U_base[1, j] + cumsum(steps), tru$controls$lower[j], tru$controls$upper[j])
      }
      list(U = U, ev = mix_events(ev_types[S[n_ch + seq_along(ev_types)]]))
    }
  } else {
    players <- c(cn, ev_types)
    build <- function(S) {
      U <- U_base
      on_c <- S[seq_len(n_u)]
      U[, on_c] <- U_ad[, on_c]
      list(U = U, ev = mix_events(ev_types[S[n_u + seq_along(ev_types)]]))
    }
  }
  J_of <- function(S) {
    b <- build(S)
    replay(b$U, ev = b$ev)
  }
  n_pl <- length(players)
  J_base <- J_of(rep(FALSE, n_pl))
  J_adopted <- J_of(rep(TRUE, n_pl))
  value_fun <- if (identical(space, "log")) {
    function(S) log(J_base) - log(J_of(S))
  } else {
    function(S) J_base - J_of(S)
  }
  sh <- shapley_values(value_fun, players = players, n_perm = ac$n_perm, seed = seed,
                       exact_max_players = ac$exact_max_players)
  total <- J_base - J_adopted
  if (identical(space, "log")) {
    tot_log <- sum(sh$phi)
    sh[, saving := if (abs(tot_log) > 1e-12) phi / tot_log * total else NA_real_]
  } else {
    sh[, saving := phi]
  }
  sh[, share := saving / total]

  # Aumann-Shapley on the continuous controls (events fixed at the adopted schedule)
  J_alpha <- function(a) {
    U <- U_base + sweep(U_ad - U_base, 2, a, "*")
    replay(U, ev = ev_ad)
  }
  as_vals <- aumann_shapley(J_alpha, n_players = n_u, n_quad = ac$n_quad, h = ac$fd_step)
  aumann <- data.table::data.table(player = cn, saving = -as_vals,
                                   note = "continuous controls, events fixed at the adopted schedule")
  list(shapley = sh, aumann = aumann, J_base = J_base, J_adopted = J_adopted, total_saving = total,
       gate_by = gate_by, space = space, players = players)
}
