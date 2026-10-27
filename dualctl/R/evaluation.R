# =============================================================================
# evaluation.R
# Monte Carlo over paired seeds (each seed fixes weather, plant population,
# initial states, observation noise and arm assignment, so all methods face
# the same plant), regret against the oracle with confidence intervals,
# parameter error, posterior log-det, interval coverage, information per
# cost, constraint violations, probe-arm cost versus the value of what it
# taught, ablations, and a plain-language summary that states where dual
# control did NOT beat simpler baselines.
# =============================================================================

# mean and t-interval
ci_summary <- function(x, level = 0.95) {
  x <- x[is.finite(x)]
  n <- length(x)
  m <- if (n > 0) mean(x) else NA_real_
  s <- if (n > 1) stats::sd(x) else NA_real_
  half <- if (n > 1) stats::qt(1 - (1 - level) / 2, df = n - 1) * s / sqrt(n) else NA_real_
  list(mean = m, sd = s, n = n, lo = m - half, hi = m + half)
}

#' Monte Carlo over paired seeds and methods
#'
#' Jobs run with `future.apply` when `parallel = TRUE` and the package is
#' available (set a `future::plan()` first; parallel workers need dualctl
#' installed). Failed jobs are recorded in `errors`, not fatal.
#'
#' @param cfg scenario config.
#' @param methods method names (default `cfg$evaluation$methods`).
#' @param seeds seeds (default master_seed + 0:(n_seeds - 1)).
#' @param parallel use future.apply.
#' @return list(epochs, posterior, errors, seeds, methods).
#' @export
run_monte_carlo <- function(cfg, methods = NULL, seeds = NULL, parallel = NULL) {
  ev <- cfg$evaluation
  methods <- methods %||% unlist(ev$methods) %||% names(cfg$methods)
  seeds <- seeds %||% (ev$master_seed + seq_len(ev$n_seeds) - 1L)
  parallel <- parallel %||% isTRUE(ev$parallel)
  jobs <- data.table::data.table(seed = rep(seeds, each = length(methods)),
                                 method = rep(methods, times = length(seeds)))
  job_fun <- function(r) {
    tryCatch({
      models <- list(fit = load_model(cfg, variant = "fitted"), truth = load_model(cfg, variant = "truth"))
      res <- run_season(cfg, method = jobs$method[r], seed = jobs$seed[r], models = models)
      list(epochs = res$epochs, posterior = res$posterior, error = NULL)
    }, error = function(e) {
      list(epochs = NULL, posterior = NULL,
           error = data.table::data.table(seed = jobs$seed[r], method = jobs$method[r],
                                          message = conditionMessage(e)))
    })
  }
  use_future <- isTRUE(parallel) && requireNamespace("future.apply", quietly = TRUE)
  out <- if (use_future) {
    future.apply::future_lapply(seq_len(nrow(jobs)), job_fun, future.seed = TRUE)
  } else {
    lapply(seq_len(nrow(jobs)), job_fun)
  }
  list(epochs = data.table::rbindlist(lapply(out, `[[`, "epochs"), fill = TRUE),
       posterior = data.table::rbindlist(lapply(out, `[[`, "posterior"), fill = TRUE),
       errors = data.table::rbindlist(lapply(out, `[[`, "error"), fill = TRUE),
       seeds = seeds, methods = methods)
}

#' Cumulative cost and regret against the oracle
#'
#' @param epochs epoch table from [run_monte_carlo()].
#' @param oracle name of the oracle method; if absent, regret is taken
#'   against the best method per seed and epoch.
#' @param level CI level.
#' @return list(per_seed, summary, final).
#' @export
compute_regret <- function(epochs, oracle = "oracle", level = 0.95) {
  per <- epochs[, .(cost = sum(cost_total)), by = .(seed, method, epoch, time_end)]
  data.table::setorder(per, seed, method, epoch)
  per[, cum_cost := cumsum(cost), by = .(seed, method)]
  ref <- if (oracle %in% per$method) {
    per[method == oracle, .(seed, epoch, cum_cost_ref = cum_cost)]
  } else {
    per[, .(cum_cost_ref = min(cum_cost)), by = .(seed, epoch)]
  }
  per <- merge(per, ref, by = c("seed", "epoch"))
  per[, cum_regret := cum_cost - cum_cost_ref]
  summ <- per[, {
    s <- ci_summary(cum_regret, level = level)
    list(mean = s$mean, lo = s$lo, hi = s$hi, n = s$n)
  }, by = .(method, epoch, time_end)]
  last <- max(per$epoch)
  final <- summ[epoch == last][order(mean)]
  list(per_seed = per, summary = summ, final = final)
}

# paired difference of total season cost: method - baseline (per seed)
paired_difference <- function(totals, method_a, method_b, level = 0.95) {
  a <- totals[method == method_a, .(seed, cost_a = cost)]
  b <- totals[method == method_b, .(seed, cost_b = cost)]
  ab <- merge(a, b, by = "seed")
  if (nrow(ab) == 0) return(NULL)
  s <- ci_summary(ab$cost_a - ab$cost_b, level = level)
  verdict <- if (is.na(s$lo) || is.na(s$hi)) "not enough seeds" else
    if (s$hi < 0) "better (lower cost)" else if (s$lo > 0) "WORSE (higher cost)" else "no significant difference"
  data.table::data.table(method = method_a, baseline = method_b, diff = s$mean, ci_lo = s$lo,
                         ci_hi = s$hi, n = s$n, rel_diff = s$mean / mean(ab$cost_b), verdict = verdict)
}

#' Summary metrics of a Monte Carlo run
#'
#' @param mc result of [run_monte_carlo()].
#' @param cfg scenario config.
#' @return list(regret, costs, comparisons, param_error, coverage, logdet,
#'   info_cost, violations, probe_value).
#' @export
summarize_metrics <- function(mc, cfg) {
  ev <- cfg$evaluation
  level <- ev$ci_level %||% 0.95
  oracle <- ev$oracle_method %||% "oracle"
  ep <- mc$epochs
  reg <- compute_regret(ep, oracle = oracle, level = level)
  totals <- ep[, .(cost = sum(cost_total)), by = .(seed, method)]
  costs <- totals[, {
    s <- ci_summary(cost, level = level)
    list(mean = s$mean, lo = s$lo, hi = s$hi, n = s$n)
  }, by = method][order(mean)]

  # paired comparisons of every dual method against every simpler baseline
  meths <- unique(totals$method)
  dual <- grep("^dual_", meths, value = TRUE)
  simple <- intersect(c("learn_sensors", "learn_noprobe", "ce_prior", "calendar", "hillclimb"), meths)
  comp <- list()
  a <- dual[1]                                        # debugging line
  for (a in dual) {
    b <- simple[1]                                    # debugging line
    for (b in simple) comp[[length(comp) + 1L]] <- paired_difference(totals, method_a = a, method_b = b, level = level)
  }
  comparisons <- data.table::rbindlist(comp)

  # parameter error, coverage and log-det (unit-level effective parameters)
  po <- mc$posterior
  param_error <- coverage <- logdet_dt <- NULL
  if (!is.null(po) && nrow(po) > 0) {
    pu <- po[unit > 0]
    pu[, `:=`(err2 = (mean - truth)^2, covered = truth >= lo & truth <= hi)]
    param_error <- pu[, .(rmse = sqrt(mean(err2))), by = .(seed, method, epoch, time, param)][
      , .(rmse = mean(rmse)), by = .(method, epoch, time, param)]
    coverage <- pu[, .(coverage = mean(covered)), by = .(method, epoch, time, param)]
    logdet_dt <- po[unit == 0, .(logdet_shared = mean(logdet)), by = .(method, epoch, time)]
  }

  # information gained per unit of probing cost and probe-arm value
  ref <- ev$reference_method %||% "learn_sensors"
  info_cost <- probe_value <- NULL
  if (!is.null(po) && nrow(po) > 0 && ref %in% meths) {
    sh <- unique(po[unit == 0, .(seed, method, epoch, logdet)])
    first <- sh[, .SD[which.min(epoch)], by = .(seed, method)]
    last <- sh[, .SD[which.max(epoch)], by = .(seed, method)]
    info <- merge(first[, .(seed, method, logdet0 = logdet)], last[, .(seed, method, logdet)],
                  by = c("seed", "method"))
    info[, info_nats := 0.5 * (logdet0 - logdet)]
    arm_cost <- ep[, .(cost = sum(cost_total)), by = .(seed, method, unit, arm)]
    rows <- list()
    a <- dual[1]                                      # debugging line
    for (a in dual) {
      x <- arm_cost[method == a]
      y <- arm_cost[method == ref, .(seed, unit, cost_ref = cost)]
      xy <- merge(x, y, by = c("seed", "unit"))
      per_seed <- xy[, .(probe_cost = sum((cost - cost_ref)[arm == "probe"]),
                         probe_value = sum((cost_ref - cost)[arm == "production"])), by = seed]
      ia <- info[method == a, .(seed, info_nats)]
      ir <- info[method == ref, .(seed, info_ref = info_nats)]
      per_seed <- merge(merge(per_seed, ia, by = "seed"), ir, by = "seed")
      pc <- ci_summary(per_seed$probe_cost, level = level)
      pv <- ci_summary(per_seed$probe_value, level = level)
      extra_info <- mean(per_seed$info_nats - per_seed$info_ref)
      rows[[length(rows) + 1L]] <- data.table::data.table(
        method = a, reference = ref,
        probe_cost = pc$mean, probe_cost_lo = pc$lo, probe_cost_hi = pc$hi,
        probe_value = pv$mean, probe_value_lo = pv$lo, probe_value_hi = pv$hi,
        net_value = pv$mean - pc$mean, info_nats = mean(per_seed$info_nats),
        extra_info_nats = extra_info,
        extra_info_per_1000_cost = if (pc$mean > 0) 1000 * extra_info / pc$mean else NA_real_)
    }
    probe_value <- data.table::rbindlist(rows)
    info_cost <- info[, .(info_nats = mean(info_nats)), by = method]
  }

  violations <- ep[, .(viol_days = sum(viol_days)), by = .(seed, method)][
    , .(viol_days = mean(viol_days)), by = method]

  list(regret = reg, costs = costs, comparisons = comparisons, param_error = param_error,
       coverage = coverage, logdet = logdet_dt, info_cost = info_cost,
       violations = violations, probe_value = probe_value)
}

#' Run the configured ablations
#'
#' Each ablation in `cfg$ablations` lists `methods` and named `values`
#' (config overrides). Returns final regret and paired comparisons per value.
#'
#' @param cfg scenario config.
#' @param seeds seeds (default master_seed + 0:(ablation_seeds - 1)).
#' @param which ablation names (default all).
#' @param parallel use future.apply.
#' @export
run_ablations <- function(cfg, seeds = NULL, which = NULL, parallel = NULL) {
  ev <- cfg$evaluation
  seeds <- seeds %||% (ev$master_seed + seq_len(ev$ablation_seeds) - 1L)
  which <- which %||% names(cfg$ablations)
  regret_rows <- list(); comp_rows <- list(); err_rows <- list()
  ab <- which[1]                                      # debugging line
  for (ab in which) {
    spec <- cfg$ablations[[ab]]
    val <- names(spec$values)[1]                      # debugging line
    for (val in names(spec$values)) {
      cfg_v <- utils::modifyList(cfg, spec$values[[val]])
      mc <- run_monte_carlo(cfg_v, methods = unlist(spec$methods), seeds = seeds, parallel = parallel)
      if (nrow(mc$errors) > 0) err_rows[[length(err_rows) + 1L]] <- mc$errors[, `:=`(ablation = ab, value = val)]
      if (nrow(mc$epochs) == 0) next
      met <- summarize_metrics(mc, cfg = cfg_v)
      regret_rows[[length(regret_rows) + 1L]] <- met$regret$final[, `:=`(ablation = ab, value = val)]
      if (!is.null(met$comparisons) && nrow(met$comparisons) > 0) {
        comp_rows[[length(comp_rows) + 1L]] <- met$comparisons[, `:=`(ablation = ab, value = val)]
      }
    }
  }
  list(regret = data.table::rbindlist(regret_rows, fill = TRUE),
       comparisons = data.table::rbindlist(comp_rows, fill = TRUE),
       errors = data.table::rbindlist(err_rows, fill = TRUE))
}

# markdown table from a data.table (numbers rounded)
md_table <- function(dt, digits = 3) {
  if (is.null(dt) || nrow(dt) == 0) return("(no rows)\n")
  d <- data.table::copy(dt)
  num <- names(d)[vapply(d, is.numeric, logical(1))]
  nm <- num[1]                                        # debugging line
  for (nm in num) data.table::set(d, j = nm, value = signif(d[[nm]], digits))
  hdr <- paste0("| ", paste(names(d), collapse = " | "), " |")
  sep <- paste0("|", paste(rep("---", ncol(d)), collapse = "|"), "|")
  body <- apply(as.matrix(d), 1, function(r) paste0("| ", paste(r, collapse = " | "), " |"))
  paste(c(hdr, sep, body, ""), collapse = "\n")
}

#' Write a plain-language results summary (markdown)
#'
#' States, from the numbers, where each dual-control method beat, matched or
#' lost to the simpler baselines, and reminds the reader that the plant is
#' synthetic.
#'
#' @param metrics result of [summarize_metrics()].
#' @param path output file.
#' @param ablations optional result of [run_ablations()].
#' @param ident optional result of [identifiability_report()].
#' @param cfg scenario config.
#' @export
write_summary <- function(metrics, path, cfg, ablations = NULL, ident = NULL) {
  L <- c("# dualctl results summary", "",
         paste0("Scenario: `", basename(cfg$.file %||% "?"), "`; seeds: ", cfg$evaluation$n_seeds,
                "; units per cohort: ", cfg$experiment$n_units, "."), "",
         "**All results come from a synthetic plant with illustrative, unvalidated parameter values.**",
         "They show how the methods behave on this simulated problem, not what they would save on a real exchanger.",
         "", "## Total season cost per method (all units)", "", md_table(metrics$costs),
         "## Final cumulative regret vs oracle", "", md_table(metrics$regret$final[, .(method, mean, lo, hi, n)]),
         "## Dual control vs simpler baselines (paired by seed; negative diff = dual cheaper)", "",
         md_table(metrics$comparisons))
  comp <- metrics$comparisons
  if (!is.null(comp) && nrow(comp) > 0) {
    not_better <- comp[verdict != "better (lower cost)"]
    L <- c(L, "### Where dual control did not win", "")
    if (nrow(not_better) == 0) {
      L <- c(L, "Every dual method was significantly cheaper than every simpler baseline in this run.", "")
    } else {
      r <- 1                                          # debugging line
      for (r in seq_len(nrow(not_better))) {
        L <- c(L, sprintf("- `%s` vs `%s`: %s (diff %.0f, 95%% CI %.0f to %.0f).",
                          not_better$method[r], not_better$baseline[r], not_better$verdict[r],
                          not_better$diff[r], not_better$ci_lo[r], not_better$ci_hi[r]))
      }
      L <- c(L, "")
    }
  }
  if (!is.null(metrics$probe_value)) {
    L <- c(L, "## Probe arm: cost versus value of what it taught", "",
           "probe_cost: extra cost of the probe units vs the reference method; probe_value: savings on the production units.", "",
           md_table(metrics$probe_value))
  }
  L <- c(L, "## Constraint violations (days with any violation, per cohort-season)", "", md_table(metrics$violations))
  if (!is.null(metrics$coverage)) {
    last <- metrics$coverage[, .SD[epoch == max(epoch)], by = method]
    L <- c(L, "## 90% interval coverage of the unit parameters at season end", "",
           "Coverage well below 0.9 signals over-confidence (expected under structural mismatch).", "",
           md_table(last[, .(method, param, coverage)]))
  }
  if (!is.null(ident)) {
    L <- c(L, "## Identifiability (one season, prior mean)", "",
           md_table(ident$classification))
  }
  if (!is.null(ablations) && nrow(ablations$regret) > 0) {
    L <- c(L, "## Ablations: final regret", "", md_table(ablations$regret[, .(ablation, value, method, mean, lo, hi)]))
    if (nrow(ablations$comparisons) > 0) {
      L <- c(L, "## Ablations: dual vs baselines", "",
             md_table(ablations$comparisons[, .(ablation, value, method, baseline, diff, ci_lo, ci_hi, verdict)]))
    }
  }
  writeLines(L, path)
  invisible(path)
}
