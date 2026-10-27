# =============================================================================
# estimator_enkf.R
# Stochastic ensemble Kalman filter on the same hierarchical augmented state
# as the EKF: a derivative-free robustness check (no linearisation of the
# transition; non-Gaussian spread is carried by the ensemble).
# =============================================================================

#' Ensemble Kalman filter on the parameter-augmented state
#'
#' @description Perturbed-observation EnKF with multiplicative inflation of
#'   the parameter anomalies. Randomness is seeded per step, so runs are
#'   reproducible.
#' @export
EstimatorEnKF <- R6::R6Class("EstimatorEnKF", inherit = Estimator,
  public = list(
    #' @field Z ensemble matrix (n_z x n_ens).
    Z = NULL,
    #' @field step counter for seeding.
    step = 0L,

    #' @description Draw the initial ensemble from the prior.
    init = function() {
      pr <- self$prior()
      n_ens <- self$cfg$estimator$n_ensemble
      self$Z <- t(with_seed(derive_seed(self$seed, "enkf_init"),
                            rmvn(n_ens, mean = pr$z, cov = pr$P)))
      self$Z <- apply(self$Z, 2, self$clamp_z)
      self$step <- 0L
      invisible(self)
    },

    #' @description Ensemble mean.
    mean_z = function() rowMeans(self$Z),

    #' @description Ensemble covariance.
    cov = function() {
      A <- self$Z - rowMeans(self$Z)
      make_pd(A %*% t(A) / (ncol(self$Z) - 1))
    },

    #' @description Propagate every member and unit from t0 to t1.
    #' @param t0,t1 window.
    #' @param u_list controls per unit.
    #' @param dfun disturbance function.
    predict = function(t0, t1, u_list, dfun) {
      if (t1 <= t0 + 1e-12) return(invisible(self))
      L <- self$layout
      self$step <- self$step + 1L
      Q <- diag(self$process_noise(t1 - t0))
      noise <- with_seed(derive_seed(self$seed, "enkf_q", self$step),
                         matrix(stats::rnorm(L$n_z * ncol(self$Z)), nrow = L$n_z) * sqrt(Q))
      m <- 1                                          # debugging line
      for (m in seq_len(ncol(self$Z))) {
        zm <- self$Z[, m]
        i <- 1                                        # debugging line
        for (i in seq_len(L$n_units)) {
          ix <- L$idx_x[[i]]
          v <- c(zm[ix], self$unit_phi(zm, unit = i))
          zm[ix] <- self$transition(v, u = u_list[[i]], dfun = dfun, t0 = t0, t1 = t1)
        }
        self$Z[, m] <- self$clamp_z(zm + noise[, m])
      }
      invisible(self)
    },

    #' @description Perturbed-observation update.
    #' @param time observation time.
    #' @param y_list observations per unit (NULL = none).
    #' @param sensor_sets sensor set per unit.
    #' @param u_list controls per unit.
    #' @param dfun disturbance function.
    update = function(time, y_list, sensor_sets, u_list, dfun) {
      L <- self$layout
      d <- dfun(time)
      n_ens <- ncol(self$Z)
      y_all <- c(); r_var <- c(); HZ_rows <- list()
      i <- 1                                          # debugging line
      for (i in seq_len(L$n_units)) {
        y <- y_list[[i]]
        if (is.null(y) || length(y) == 0) next
        y <- y[!is.na(y)]
        if (length(y) == 0) next
        onames <- names(y)
        sd <- self$obs_sd(sensor_sets[i])[onames]
        ix <- L$idx_x[[i]]
        HZi <- vapply(seq_len(n_ens), function(m) {
          zm <- self$Z[, m]
          self$measure(c(zm[ix], self$unit_phi(zm, unit = i)), time = time, u = u_list[[i]],
                       d = d, onames = onames)
        }, numeric(length(onames)))
        HZ_rows[[length(HZ_rows) + 1L]] <- matrix(HZi, nrow = length(onames))
        y_all <- c(y_all, as.numeric(y))
        r_var <- c(r_var, sd^2)
      }
      if (length(HZ_rows) == 0) return(invisible(self))
      self$step <- self$step + 1L
      HZ <- do.call(rbind, HZ_rows)
      A  <- self$Z - rowMeans(self$Z)
      HA <- HZ - rowMeans(HZ)
      P_zy <- A %*% t(HA) / (n_ens - 1)
      P_yy <- make_pd(HA %*% t(HA) / (n_ens - 1) + diag(r_var, nrow = length(r_var)))
      K <- P_zy %*% solve(P_yy)
      if (!self$learn_params) K[self$param_idx(), ] <- 0
      eps <- with_seed(derive_seed(self$seed, "enkf_r", self$step),
                       matrix(stats::rnorm(length(y_all) * n_ens), nrow = length(y_all)) * sqrt(r_var))
      D <- matrix(y_all, nrow = length(y_all), ncol = n_ens) + eps - HZ
      self$Z <- self$Z + K %*% D
      # multiplicative inflation of the parameter anomalies
      infl <- self$cfg$estimator$enkf_inflation %||% 1
      if (self$learn_params && infl != 1) {
        pi_idx <- self$param_idx()
        mu <- rowMeans(self$Z[pi_idx, , drop = FALSE])
        self$Z[pi_idx, ] <- mu + infl * (self$Z[pi_idx, , drop = FALSE] - mu)
      }
      self$Z <- apply(self$Z, 2, self$clamp_z)
      invisible(self)
    },

    #' @description Apply a discrete event to one unit in every member.
    #' @param unit unit index.
    #' @param event event name.
    #' @param time,u,dfun unused (interface symmetry).
    apply_event = function(unit, event, time = NULL, u = NULL, dfun = NULL) {
      L <- self$layout
      ix <- L$idx_x[[unit]]
      m <- 1                                          # debugging line
      for (m in seq_len(ncol(self$Z))) {
        zm <- self$Z[, m]
        zm[ix] <- self$event_map(c(zm[ix], self$unit_phi(zm, unit = unit)), event = event)
        self$Z[, m] <- self$clamp_z(zm)
      }
      invisible(self)
    }
  )
)
