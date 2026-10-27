# =============================================================================
# demo.R
# End-to-end demo: identifiability report, one season with the advisor,
# Shapley attribution, Monte Carlo over paired seeds, ablations, figures,
# tables and a plain-language summary. Everything derives from one seed.
# =============================================================================

#' Run the full demo and write all figures
#'
#' @param config scenario file (path or file name in the package `config` dir).
#' @param seed master seed (overrides `cfg$evaluation$master_seed`).
#' @param out_dir output directory (figures, tables, `summary.md`, `results.rds`).
#' @param n_seeds optional override of the number of Monte Carlo seeds.
#' @param ablation_seeds optional override of the seeds per ablation value.
#' @param do_mc run the Monte Carlo comparison.
#' @param do_ablations run the ablations.
#' @param parallel use future.apply (set `future::plan()` before).
#' @param save write files to `out_dir`.
#' @return list(pList, results).
#' @export
run_demo <- function(config = "fouling_default.yaml", seed = NULL, out_dir = "demo_output",
                     n_seeds = NULL, ablation_seeds = NULL, do_mc = TRUE, do_ablations = TRUE,
                     parallel = NULL, save = TRUE) {
  cfg <- load_scenario(config)
  if (!is.null(seed)) cfg$evaluation$master_seed <- seed
  if (!is.null(n_seeds)) cfg$evaluation$n_seeds <- n_seeds
  if (!is.null(ablation_seeds)) cfg$evaluation$ablation_seeds <- ablation_seeds
  seed <- cfg$evaluation$master_seed
  dm <- cfg$demo %||% list()
  pList <- list()
  results <- list(cfg = cfg)
  models <- list(fit = load_model(cfg, variant = "fitted"), truth = load_model(cfg, variant = "truth"))

  # 1. identifiability: which parameters do weather/load identify, which need probing?
  message("[dualctl] identifiability report")
  ident <- identifiability_report(cfg, model = models$fit, seed = seed)
  results$identifiability <- ident
  p_id <- plot_identifiability(ident)
  pList$identifiability_spectrum <- p_id$p_spectrum
  pList$identifiability_bounds <- p_id$p_bounds

  # 2. one season with the advisor
  demo_method <- dm$method %||% "dual_infompc"
  message("[dualctl] demo season: ", demo_method)
  season <- run_season(cfg, method = demo_method, seed = seed, models = models, advisor = TRUE, keep_obs = TRUE)
  results$season <- season
  pList$param_convergence_shared <- plot_param_convergence(season$posterior, unit_idx = 0)
  adv_unit <- if (nrow(season$advisor$ranking) > 0) season$advisor$ranking$unit[1] else
    which(season$arms == "production")[1]
  if (is.na(adv_unit)) adv_unit <- 1L
  pList$param_convergence_unit <- plot_param_convergence(season$posterior, unit_idx = adv_unit)
  if (nrow(season$advisor$ranking) > 0) pList$advisor_sequence <- plot_advisor_sequence(season$advisor$ranking)
  ps <- plot_season(season, state_col = dm$state_col %||% models$truth$state_names[1], obs_col = dm$obs_col)
  pn <- names(ps)[1]                                  # debugging line
  for (pn in names(ps)) pList[[paste0("season_", pn)]] <- ps[[pn]]

  # 3. Shapley attribution of the advisor unit's savings vs the calendar baseline
  message("[dualctl] attribution")
  attrib <- season_attribution(season, unit_idx = adv_unit, cfg = cfg, models = models, seed = seed)
  results$attribution <- attrib
  pList$shapley <- plot_shapley(attrib)

  # 4. Monte Carlo over paired seeds
  met <- NULL
  if (do_mc) {
    message("[dualctl] Monte Carlo: ", cfg$evaluation$n_seeds, " seeds")
    mc <- run_monte_carlo(cfg, parallel = parallel)
    results$mc <- mc
    if (nrow(mc$errors) > 0) warning(nrow(mc$errors), " Monte Carlo jobs failed; see results$mc$errors")
    met <- summarize_metrics(mc, cfg = cfg)
    results$metrics <- met
    pr <- plot_regret(met$regret)
    pList$regret_time <- pr$p_time
    pList$regret_final <- pr$p_final
    if (!is.null(met$param_error)) pList$param_rmse <- plot_param_rmse(met$param_error)
    if (!is.null(met$logdet)) pList$posterior_logdet <- plot_logdet(met$logdet)
    if (!is.null(met$coverage)) pList$coverage <- plot_coverage(met$coverage, level = cfg$evaluation$coverage_level)
    if (!is.null(met$probe_value) && nrow(met$probe_value) > 0) pList$probe_value <- plot_probe_value(met$probe_value)
  }

  # 5. ablations
  abl <- NULL
  if (do_ablations && length(cfg$ablations) > 0) {
    message("[dualctl] ablations")
    abl <- run_ablations(cfg, parallel = parallel)
    results$ablations <- abl
    if (nrow(abl$regret) > 0) pList$ablations <- plot_ablations(abl)
  }

  if (save) {
    dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
    pn <- names(pList)[1]                             # debugging line
    for (pn in names(pList)) {
      ggplot2::ggsave(file.path(out_dir, paste0(pn, ".png")), plot = pList[[pn]],
                      width = 8, height = 5, dpi = 150, bg = dc_colors$surface)
    }
    data.table::fwrite(ident$classification, file.path(out_dir, "identifiability.csv"))
    data.table::fwrite(season$epochs, file.path(out_dir, "season_epochs.csv"))
    if (nrow(season$advisor$ranking) > 0) data.table::fwrite(season$advisor$ranking, file.path(out_dir, "advisor_ranking.csv"))
    if (nrow(season$advisor$sequence) > 0) data.table::fwrite(season$advisor$sequence, file.path(out_dir, "advisor_sequence.csv"))
    data.table::fwrite(attrib$shapley, file.path(out_dir, "shapley.csv"))
    if (!is.null(met)) {
      data.table::fwrite(met$costs, file.path(out_dir, "costs.csv"))
      data.table::fwrite(met$regret$final, file.path(out_dir, "regret_final.csv"))
      data.table::fwrite(met$comparisons, file.path(out_dir, "comparisons.csv"))
      write_summary(met, path = file.path(out_dir, "summary.md"), cfg = cfg, ablations = abl, ident = ident)
    }
    saveRDS(results, file.path(out_dir, "results.rds"))
    message("[dualctl] wrote ", length(pList), " figures to ", normalizePath(out_dir))
  }
  invisible(list(pList = pList, results = results))
}
