# =============================================================================
# zzz.R  --  package hooks and NSE column names
# =============================================================================

.onLoad <- function(libname, pkgname) {
  register_builtin_strategies()
  invisible(NULL)
}

# column names used with data.table non-standard evaluation
utils::globalVariables(c(
  ".", ".N", ".SD", "time", "seed", "method", "unit", "arm", "epoch", "param", "mean", "sd",
  "lo", "hi", "truth", "logdet", "cost", "cost_total", "cost_running", "cost_events",
  "cum_cost", "cum_cost_ref", "cum_regret", "regret", "time_end", "scenario", "step",
  "upper", "lower", "score", "saving_mean", "change_cost", "rank", "control", "kind",
  "covered", "err2", "n", "se", "estimate", "value", "player", "phi", "share",
  "identified_by_disturbances", "identified_with_probing", "disturbances", "probing",
  "verdict", "eigenvalue", "k", "ratio", "bcrb_sd", "prior_sd", "viol_days",
  "ablation", "level", "diff", "ci_lo", "ci_hi", "info_nats", "probe_cost", "probe_value",
  "logdet0", "n_events", "events", "y", "mean_val", "lo_val", "hi_val", "order",
  "delta_nll", "saving", "contribution", "label", "base", "cost_ref", "is_probe",
  "rmse", "coverage", "logdet_shared", ".data", "top", "what", "sign", "probe_cost_lo",
  "probe_cost_hi", "probe_value_lo", "probe_value_hi", "cost_a", "cost_b", "baseline",
  "info_ref", "x", "epoch_row", "delta", "note"))
