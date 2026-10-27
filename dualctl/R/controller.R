# =============================================================================
# controller.R
# Receding-horizon optimisation over decision epochs.
#   * continuous controls: move blocking (cfg$controller$move_blocks), each
#     block constant, block changes limited by the per-epoch rate limits;
#     decision variables live in [0, 1] and are mapped into the rate-limited
#     box, so every candidate is feasible by construction.
#   * discrete events (e.g. cleaning): enumeration over their timing within
#     the horizon (none, epoch 0, ..., H-1), respecting minimum intervals.
#   * optimiser (derivative-free): random shooting with common candidates,
#     Nelder-Mead refinement (stats::optim) or BOBYQA (nloptr, optional).
# Objective: integrated stage cost + event costs + constraint penalties +
# terminal cost, simulated with the forecast disturbances.
# =============================================================================

#' Receding-horizon controller
#'
#' @description Plans controls and event timing over `horizon_epochs` for one
#'   unit given a state, parameters and a disturbance forecast.
#' @export
Controller <- R6::R6Class("Controller",
  public = list(
    #' @field model `dc_model` used for prediction.
    model = NULL,
    #' @field cfg scenario config.
    cfg = NULL,
    #' @field H horizon in epochs.
    H = NULL,
    #' @field E epoch length.
    E = NULL,
    #' @field blocks move-block lengths in epochs.
    blocks = NULL,
    #' @field penalty constraint penalty weights.
    penalty = NULL,

    #' @description Create a controller.
    #' @param model `dc_model`.
    #' @param cfg scenario config.
    initialize = function(model, cfg) {
      if (model$n_u < 1) stop("the controller needs at least one continuous control")
      self$model <- model
      self$cfg <- cfg
      cc <- cfg$controller
      self$H <- as.integer(cc$horizon_epochs)
      self$E <- cfg_num(cfg$season$epoch_days, "season$epoch_days")
      b <- as.integer(unlist(cc$move_blocks))
      if (length(b) == 0) b <- self$H
      if (sum(b) < self$H) {
        b[length(b)] <- b[length(b)] + (self$H - sum(b))
      } else if (sum(b) > self$H) {
        keep <- which(cumsum(b) < self$H)
        b <- c(b[keep], self$H - sum(b[keep]))
      }
      self$blocks <- b[b > 0]
      self$penalty <- vapply(model$constraint_outputs, function(n) {
        as.numeric(cc$constraint_penalty[[n]] %||% cc$constraint_penalty_default)
      }, numeric(1))
    },

    #' @description Rate-limited box for a control change from `u_last`.
    #' @param u_last previous control values.
    box = function(u_last) {
      ct <- self$model$controls
      lo <- pmax(ct$lower, u_last - ct$rate)
      hi <- pmin(ct$upper, u_last + ct$rate)
      bad <- lo > hi
      lo[bad] <- hi[bad] <- clamp(u_last[bad], ct$lower[bad], ct$upper[bad])
      list(lo = lo, hi = hi)
    },

    #' @description Map decision variables in \[0, 1\] to an H x n_u control matrix.
    #' @param dec decision vector (block-major).
    #' @param u_prev controls applied in the previous epoch.
    decode = function(dec, u_prev) {
      n_u <- self$model$n_u
      nb <- length(self$blocks)
      D <- matrix(dec, nrow = nb, ncol = n_u, byrow = TRUE)
      U <- matrix(NA_real_, nrow = self$H, ncol = n_u,
                  dimnames = list(NULL, self$model$control_names))
      u_last <- as.numeric(u_prev)
      row0 <- 0L
      b <- 1                                          # debugging line
      for (b in seq_len(nb)) {
        bx <- self$box(u_last)
        ub <- bx$lo + D[b, ] * (bx$hi - bx$lo)
        rows <- row0 + seq_len(self$blocks[b])
        U[rows, ] <- matrix(ub, nrow = length(rows), ncol = n_u, byrow = TRUE)
        u_last <- ub
        row0 <- row0 + self$blocks[b]
      }
      U
    },

    #' @description Decision vector that holds the previous controls.
    #' @param u_prev previous controls.
    hold_dec = function(u_prev) {
      bx <- self$box(as.numeric(u_prev))
      w <- bx$hi - bx$lo
      d <- ifelse(w > 0, (as.numeric(u_prev) - bx$lo) / pmax(w, 1e-300), 0.5)
      rep(clamp(d, 0, 1), times = length(self$blocks))
    },

    #' @description Project an H x n_u control matrix onto bounds and rate limits.
    #' @param U control matrix.
    #' @param u_prev previous controls.
    project_U = function(U, u_prev) {
      u_last <- as.numeric(u_prev)
      r <- 1                                          # debugging line
      for (r in seq_len(nrow(U))) {
        bx <- self$box(u_last)
        U[r, ] <- clamp(U[r, ], bx$lo, bx$hi)
        u_last <- U[r, ]
      }
      U
    },

    #' @description Feasible event options within the horizon.
    #' @param epoch current epoch index.
    #' @param last_event_epoch named vector of the epoch of each event's last occurrence.
    event_options = function(epoch, last_event_epoch = NULL) {
      opts <- list()
      en <- self$model$event_names[1]                 # debugging line
      for (en in self$model$event_names) {
        last <- if (!is.null(last_event_epoch) && en %in% names(last_event_epoch)) last_event_epoch[[en]] else -Inf
        mi <- self$model$events[[en]]$min_interval_epochs
        tau <- 0                                      # debugging line
        for (tau in seq_len(self$H) - 1L) {
          if (epoch + tau - last >= mi) opts[[length(opts) + 1L]] <- list(name = en, tau = tau)
        }
      }
      opts
    },

    #' @description Objective of an explicit control matrix and event option.
    #' @param U H x n_u control matrix.
    #' @param ev NULL or list(name, tau).
    #' @param x0 initial state.
    #' @param parms natural parameters.
    #' @param dfun disturbance (forecast) function.
    #' @param t0 plan start.
    #' @return list(J, res).
    evaluate_U = function(U, ev, x0, parms, dfun, t0) {
      times <- t0 + (seq_len(self$H) - 1) * self$E
      u_plan <- make_u_plan(times = times, U = U, control_names = self$model$control_names)
      ev_plan <- if (is.null(ev)) NULL else data.table::data.table(time = t0 + ev$tau * self$E, event = ev$name)
      res <- simulate_unit(self$model, x0 = x0, parms = parms, u_plan = u_plan, dfun = dfun,
                           t0 = t0, t1 = t0 + self$H * self$E, dt = self$cfg$solver$dt_inner,
                           events_plan = ev_plan, engine = self$cfg$solver$inner_engine,
                           terminal_days = self$cfg$controller$terminal_days)
      pen <- if (length(self$penalty) > 0) sum(self$penalty * res$viol[names(self$penalty)]) else 0
      list(J = res$cost + pen, res = res)
    },

    #' @description Objective of a decision vector.
    #' @param dec decision vector.
    #' @param ev event option or NULL.
    #' @param x0,parms,u_prev,dfun,t0 see `evaluate_U`.
    evaluate = function(dec, ev, x0, parms, u_prev, dfun, t0) {
      self$evaluate_U(self$decode(dec, u_prev), ev = ev, x0 = x0, parms = parms, dfun = dfun, t0 = t0)
    },

    #' @description Local derivative-free refinement in the decision space.
    #' @param dec0 start.
    #' @param f objective of a decision vector.
    #' @param maxit iterations / evaluations.
    #' @param method optimiser name.
    refine = function(dec0, f, maxit, method = "hybrid") {
      n <- length(dec0)
      if (maxit <= 0 || n == 0) return(list(dec = dec0, J = Inf))
      if (identical(method, "nloptr") && n >= 2 && requireNamespace("nloptr", quietly = TRUE)) {
        opt <- nloptr::bobyqa(dec0, f, lower = rep(0, n), upper = rep(1, n),
                              control = list(maxeval = maxit))
        return(list(dec = opt$par, J = opt$value))
      }
      obj <- function(p) f(stats::plogis(p))
      if (n == 1) {
        opt <- stats::optimize(obj, interval = c(-8, 8))
        return(list(dec = stats::plogis(opt$minimum), J = opt$objective))
      }
      p0 <- stats::qlogis(clamp(dec0, 0.02, 0.98))
      opt <- stats::optim(p0, obj, method = "Nelder-Mead", control = list(maxit = maxit, reltol = 1e-6))
      list(dec = stats::plogis(opt$par), J = opt$value)
    },

    #' @description Plan controls and event timing over the horizon.
    #' @param x0 named initial state.
    #' @param parms natural parameters.
    #' @param u_prev controls applied in the previous epoch.
    #' @param forecast_dt disturbance forecast table.
    #' @param t0 plan start.
    #' @param epoch current epoch index.
    #' @param last_event_epoch named vector of last event epochs.
    #' @param seed seed for the random candidates.
    #' @param allow_events consider events.
    #' @param n_candidates,maxit optional effort overrides.
    #' @return list(U, event, J, u_first, events_first, n_eval, dec).
    plan = function(x0, parms, u_prev, forecast_dt, t0, epoch = 0, last_event_epoch = NULL,
                    seed = 1L, allow_events = TRUE, n_candidates = NULL, maxit = NULL) {
      oc <- self$cfg$controller$optimizer
      n_cand <- as.integer(n_candidates %||% oc$n_candidates)
      maxit <- as.integer(maxit %||% oc$maxit)
      cn <- self$model$control_names
      u_prev <- stats::setNames(as.numeric(u_prev[cn]), cn)
      dfun <- make_interp_fun(forecast_dt, cols = self$model$dist_names)
      n_dec <- self$model$n_u * length(self$blocks)
      ev_opts <- if (allow_events) self$event_options(epoch, last_event_epoch = last_event_epoch) else list()
      counter <- new.env()
      counter$n <- 0L
      fn <- function(dec, ev) {
        counter$n <- counter$n + 1L
        self$evaluate(dec, ev = ev, x0 = x0, parms = parms, u_prev = u_prev, dfun = dfun, t0 = t0)$J
      }
      cands <- matrix(self$hold_dec(u_prev), nrow = 1)
      if (n_cand > 0) {
        cands <- rbind(cands, with_seed(seed, matrix(stats::runif(n_cand * n_dec), nrow = n_cand, ncol = n_dec)))
      }
      method <- oc$method %||% "hybrid"
      if (identical(method, "random_shooting")) {
        best <- list(J = Inf, dec = cands[1, ], ev = NULL)
        ev <- NULL                                    # debugging line
        for (ev in c(list(NULL), ev_opts)) {
          r <- 1                                      # debugging line
          for (r in seq_len(nrow(cands))) {
            J <- fn(cands[r, ], ev)
            if (J < best$J) best <- list(J = J, dec = cands[r, ], ev = ev)
          }
        }
      } else {
        Jc <- apply(cands, 1, function(dd) fn(dd, NULL))
        dec <- cands[which.min(Jc), ]
        J0 <- min(Jc)
        ref <- self$refine(dec, f = function(d) fn(d, NULL), maxit = maxit, method = method)
        if (ref$J < J0) {
          dec <- ref$dec
          J0 <- ref$J
        }
        best <- list(J = J0, dec = dec, ev = NULL)
        ev <- NULL                                    # debugging line
        for (ev in ev_opts) {
          J <- fn(dec, ev)
          if (J < best$J) best <- list(J = J, dec = dec, ev = ev)
        }
        if (!is.null(best$ev) && isTRUE(oc$event_reopt) && maxit > 0) {
          ev_best <- best$ev
          ref <- self$refine(best$dec, f = function(d) fn(d, ev_best),
                             maxit = as.integer(ceiling(maxit / 2)), method = method)
          if (ref$J < best$J) {
            best$J <- ref$J
            best$dec <- ref$dec
          }
        }
      }
      U <- self$decode(best$dec, u_prev = u_prev)
      ev_first <- if (!is.null(best$ev) && best$ev$tau == 0) best$ev$name else character(0)
      list(U = U, event = best$ev, J = best$J, u_first = stats::setNames(U[1, ], cn),
           events_first = ev_first, n_eval = counter$n, dec = best$dec)
    }
  )
)
