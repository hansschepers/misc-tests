# =============================================================================
# utils.R
# Small helpers used across the engine: seeding, transforms, linear algebra,
# configuration loading. No model-specific numbers live here; every tunable
# number is read from the scenario config (see inst/config/engine_defaults.yaml).
# =============================================================================

#' @import data.table
#' @importFrom R6 R6Class
NULL

`%||%` <- function(a, b) if (is.null(a)) b else a

# ---------------------------------------------------------------------------
# Random numbers
# ---------------------------------------------------------------------------

#' Evaluate an expression with a local RNG seed
#'
#' The global RNG state is restored afterwards, so seeded blocks do not
#' disturb each other. This is what makes paired Monte Carlo comparisons
#' possible: weather, population draws and observation noise depend only on
#' the seed, never on the method that is run.
#'
#' @param seed integer seed.
#' @param expr expression to evaluate (lazily, after the seed is set).
#' @return the value of `expr`.
#' @export
with_seed <- function(seed, expr) {
  genv <- globalenv()
  had_seed <- exists(".Random.seed", envir = genv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = genv, inherits = FALSE)
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = genv)
    } else if (exists(".Random.seed", envir = genv, inherits = FALSE)) {
      rm(".Random.seed", envir = genv)
    }
  }, add = TRUE)
  set.seed(seed)
  expr
}

#' Derive a reproducible integer seed from any number of keys
#'
#' djb2 string hash of the pasted keys, reduced modulo 2^31 - 1.
#'
#' @param ... keys (numbers or strings).
#' @return integer seed.
#' @export
derive_seed <- function(...) {
  key   <- paste(..., sep = "|")
  codes <- utf8ToInt(key)
  h     <- 5381
  cc    <- codes[1]                                   # debugging line
  for (cc in codes) h <- (h * 33 + cc) %% 2147483647
  as.integer(h)
}

#' Draw from a multivariate normal distribution
#'
#' @param n number of draws.
#' @param mean mean vector.
#' @param cov covariance matrix.
#' @return matrix with `n` rows.
#' @export
rmvn <- function(n, mean, cov) {
  k <- length(mean)
  if (k == 0) return(matrix(numeric(0), nrow = n, ncol = 0))
  R <- chol(make_pd(cov))
  z <- matrix(stats::rnorm(n * k), nrow = n, ncol = k)
  out <- z %*% R + matrix(mean, nrow = n, ncol = k, byrow = TRUE)
  colnames(out) <- names(mean)
  out
}

# ---------------------------------------------------------------------------
# Parameter transforms
# ---------------------------------------------------------------------------

#' Map natural parameter values to the unconstrained estimation space
#'
#' `log`: phi = log(theta); `logit`: phi = logit((theta - lo) / (hi - lo));
#' `identity`: phi = theta.
#'
#' @param theta natural values.
#' @param transform character vector of transforms.
#' @param lo,hi bounds used by the logit transform.
#' @return transformed values.
#' @export
transform_vec <- function(theta, transform, lo = 0, hi = 1) {
  n <- length(theta)
  transform <- rep_len(transform, n); lo <- rep_len(lo, n); hi <- rep_len(hi, n)
  out <- theta
  i_log <- transform == "log"
  out[i_log] <- log(theta[i_log])
  i_lgt <- transform == "logit"
  out[i_lgt] <- stats::qlogis((theta[i_lgt] - lo[i_lgt]) / (hi[i_lgt] - lo[i_lgt]))
  out
}

#' Map transformed parameter values back to natural units
#'
#' @inheritParams transform_vec
#' @param phi transformed values.
#' @return natural values.
#' @export
inverse_transform_vec <- function(phi, transform, lo = 0, hi = 1) {
  n <- length(phi)
  transform <- rep_len(transform, n); lo <- rep_len(lo, n); hi <- rep_len(hi, n)
  out <- phi
  i_log <- transform == "log"
  out[i_log] <- exp(phi[i_log])
  i_lgt <- transform == "logit"
  out[i_lgt] <- lo[i_lgt] + (hi[i_lgt] - lo[i_lgt]) * stats::plogis(phi[i_lgt])
  out
}

# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

#' Numerically stable softplus, a smooth version of max(x, 0)
#'
#' softplus(x; s) = s * log(1 + exp(x / s)); tends to max(x, 0) for s -> 0.
#'
#' @param x numeric.
#' @param s smoothing scale (same units as x).
#' @export
softplus <- function(x, s) {
  z <- x / s
  s * (pmax(z, 0) + log1p(exp(-abs(z))))
}

# symmetrise and floor the eigenvalues of a covariance matrix
make_pd <- function(m, floor_rel = 1e-10, floor_abs = 1e-14) {
  m <- (m + t(m)) / 2
  if (length(m) == 0) return(m)
  ok <- tryCatch({ chol(m); TRUE }, error = function(e) FALSE)
  if (ok) return(m)
  ev <- eigen(m, symmetric = TRUE)
  lam_floor <- max(floor_abs, floor_rel * max(abs(ev$values)))
  lam <- pmax(ev$values, lam_floor)
  out <- ev$vectors %*% (lam * t(ev$vectors))
  dimnames(out) <- dimnames(m)
  (out + t(out)) / 2
}

#' Log-determinant of a symmetric positive definite matrix
#' @param m matrix.
#' @export
logdet <- function(m) {
  if (length(m) == 0) return(0)
  R <- chol(make_pd(m))
  2 * sum(log(diag(R)))
}

# trapezoidal integral of y over x
trapz <- function(x, y) {
  if (length(x) < 2) return(0)
  sum(diff(x) * (utils::head(y, -1) + utils::tail(y, -1)) / 2)
}

# forward or central finite-difference Jacobian of fun at v0
# steps: vector of step sizes (one per element of v0)
fd_jacobian <- function(fun, v0, steps, f0 = NULL, central = FALSE) {
  if (is.null(f0)) f0 <- fun(v0)
  n <- length(v0)
  J <- matrix(0, nrow = length(f0), ncol = n)
  j <- 1                                              # debugging line
  for (j in seq_len(n)) {
    h  <- steps[j]
    vp <- v0; vp[j] <- vp[j] + h
    if (central) {
      vm <- v0; vm[j] <- vm[j] - h
      J[, j] <- (fun(vp) - fun(vm)) / (2 * h)
    } else {
      J[, j] <- (fun(vp) - f0) / h
    }
  }
  list(f0 = f0, J = J)
}

clamp <- function(x, lo, hi) pmin(pmax(x, lo), hi)

# ---------------------------------------------------------------------------
# Annealing schedules (used for lambda_t and dither amplitude)
# ---------------------------------------------------------------------------

#' Annealing schedule multiplier
#'
#' @param frac season fraction elapsed, in \[0, 1\].
#' @param schedule one of `constant`, `linear`, `exponential`, `step`, `none`.
#' @param rate decay rate for `exponential` \[-\].
#' @param step_at season fraction where `step` drops to zero \[-\].
#' @return multiplier in \[0, 1\].
#' @export
anneal_schedule <- function(frac, schedule = "linear", rate = 3, step_at = 0.5) {
  switch(schedule,
         constant    = 1,
         linear      = max(0, 1 - frac),
         exponential = exp(-rate * frac),
         step        = as.numeric(frac < step_at),
         none        = 0,
         stop("unknown annealing schedule: ", schedule))
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#' Load a scenario configuration
#'
#' Reads the engine defaults shipped with the package, overlays the scenario
#' YAML file and finally `overrides` (a nested list, e.g. from an ablation).
#'
#' @param path path to a scenario YAML file. A bare file name is also looked
#'   up in the package's `config` directory.
#' @param overrides nested list merged last.
#' @return nested list; `cfg$.dir` holds the directory of the scenario file
#'   (used to resolve relative model and data paths).
#' @export
load_scenario <- function(path, overrides = list()) {
  path <- resolve_path(file = path, base_dir = getwd(), subdir = "config")
  defaults_path <- system.file("config", "engine_defaults.yaml", package = "dualctl")
  if (!nzchar(defaults_path)) stop("engine_defaults.yaml not found; is dualctl installed or loaded?")
  defaults <- yaml::read_yaml(defaults_path)
  scen <- yaml::read_yaml(path)
  cfg <- utils::modifyList(defaults, scen)
  cfg <- utils::modifyList(cfg, overrides)
  cfg$.dir  <- dirname(path)
  cfg$.file <- path
  cfg
}

# locate a file: as given, relative to base_dir, or inside the package's inst/<subdir>
resolve_path <- function(file, base_dir = NULL, subdir = NULL) {
  if (file.exists(file)) return(normalizePath(file))
  if (!is.null(base_dir)) {
    cand <- file.path(base_dir, file)
    if (file.exists(cand)) return(normalizePath(cand))
  }
  if (!is.null(subdir)) {
    cand <- system.file(subdir, basename(file), package = "dualctl")
    if (nzchar(cand)) return(cand)
  }
  stop("cannot find file: ", file)
}

# numeric config value with a clear error if missing
cfg_num <- function(x, what) {
  if (is.null(x)) stop("missing numeric config entry: ", what)
  as.numeric(x)
}
