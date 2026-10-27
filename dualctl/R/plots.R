# =============================================================================
# plots.R
# ggplot2 figures. Simulation and visualisation are kept apart: every
# function takes result tables and returns a ggplot object.
# Colour: categorical slots in fixed order, mapped to the method NAME (colour
# follows the entity, not its rank); single-hue blue for magnitude; blue/red
# for polarity; recessive grid; one y-axis per panel.
# =============================================================================

dc_colors <- list(
  series = c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
  surface = "#fcfcfb", ink = "#0b0b0b", ink2 = "#52514e", muted = "#898781",
  grid = "#e1e0d9", axis = "#c3c2b7", seq_lo = "#cde2fb", seq_hi = "#104281",
  pos = "#2a78d6", neg = "#e34948", mid = "#f0efec"
)

# canonical method order: fixes each method's colour across all figures
dc_method_order <- c("calendar", "ce_prior", "learn_noprobe", "learn_sensors", "hillclimb",
                     "dual_dither", "dual_infompc", "dual_thompson", "dual_evsi", "oracle")

#' Colour per method (fixed by name)
#' @param methods method names.
#' @export
method_palette <- function(methods) {
  m <- unique(c(intersect(dc_method_order, methods), setdiff(methods, dc_method_order)))
  m <- setdiff(m, "oracle")
  if (length(m) > length(dc_colors$series)) {
    warning("more than ", length(dc_colors$series), " methods: split the figure (colours are not recycled)")
    m <- m[seq_along(dc_colors$series)]
  }
  pal <- stats::setNames(dc_colors$series[seq_along(m)], m)
  c(pal, oracle = dc_colors$ink2)
}

#' ggplot2 theme of the package figures
#' @param base_size base font size \[pt\].
#' @export
dc_theme <- function(base_size = 11) {
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = dc_colors$surface, colour = NA),
      panel.background = ggplot2::element_rect(fill = dc_colors$surface, colour = NA),
      panel.grid.major = ggplot2::element_line(colour = dc_colors$grid, linewidth = 0.3),
      panel.grid.minor = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(colour = dc_colors$axis, linewidth = 0.4),
      axis.text = ggplot2::element_text(colour = dc_colors$ink2),
      axis.title = ggplot2::element_text(colour = dc_colors$ink2),
      plot.title = ggplot2::element_text(colour = dc_colors$ink, face = "bold"),
      plot.subtitle = ggplot2::element_text(colour = dc_colors$ink2),
      plot.caption = ggplot2::element_text(colour = dc_colors$muted, hjust = 0),
      strip.text = ggplot2::element_text(colour = dc_colors$ink, face = "bold"),
      legend.text = ggplot2::element_text(colour = dc_colors$ink2),
      legend.title = ggplot2::element_text(colour = dc_colors$ink2),
      legend.position = "bottom"
    )
}

synthetic_caption <- "Synthetic plant, illustrative parameter values (not validated)."

#' Cumulative regret against the oracle
#' @param regret result of [compute_regret()].
#' @return list(p_time, p_final).
#' @export
plot_regret <- function(regret) {
  s <- regret$summary[method != "oracle"]
  pal <- method_palette(unique(s$method))
  last <- s[epoch == max(epoch)]
  p_time <- ggplot2::ggplot(s, ggplot2::aes(x = time_end, y = mean / 1000, colour = method, fill = method)) +
    ggplot2::geom_hline(yintercept = 0, colour = dc_colors$axis, linewidth = 0.4) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = lo / 1000, ymax = hi / 1000), alpha = 0.12, colour = NA) +
    ggplot2::geom_line(linewidth = 0.7) +
    ggplot2::scale_colour_manual(values = pal) +
    ggplot2::scale_fill_manual(values = pal, guide = "none") +
    ggplot2::labs(x = "time [d]", y = "cumulative regret vs oracle [kEUR]", colour = NULL,
                  title = "Cumulative regret by method",
                  subtitle = "Mean over paired seeds, 95% CI band; 0 = oracle with true parameters",
                  caption = synthetic_caption) +
    dc_theme()
  last[, method := factor(method, levels = last[order(mean)]$method)]
  p_final <- ggplot2::ggplot(last, ggplot2::aes(x = mean / 1000, y = method, colour = method)) +
    ggplot2::geom_vline(xintercept = 0, colour = dc_colors$axis, linewidth = 0.4) +
    ggplot2::geom_errorbar(ggplot2::aes(xmin = lo / 1000, xmax = hi / 1000), width = 0, linewidth = 0.7,
                           orientation = "y") +
    ggplot2::geom_point(size = 2.6) +
    ggplot2::scale_colour_manual(values = pal, guide = "none") +
    ggplot2::labs(x = "season regret vs oracle [kEUR] (mean, 95% CI)", y = NULL,
                  title = "Season-end regret", caption = synthetic_caption) +
    dc_theme()
  list(p_time = p_time, p_final = p_final)
}

#' Parameter convergence of one unit in one season
#' @param posterior posterior table of [run_season()].
#' @param unit_idx unit (0 = shared parameters).
#' @export
plot_param_convergence <- function(posterior, unit_idx = 1) {
  d <- posterior[posterior[["unit"]] == unit_idx, ]
  ggplot2::ggplot(d, ggplot2::aes(x = time)) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = lo, ymax = hi), fill = dc_colors$series[1], alpha = 0.18) +
    ggplot2::geom_line(ggplot2::aes(y = mean), colour = dc_colors$series[1], linewidth = 0.7) +
    ggplot2::geom_line(ggplot2::aes(y = truth), colour = dc_colors$ink2, linewidth = 0.5, linetype = "dashed") +
    ggplot2::facet_wrap(~param, scales = "free_y") +
    ggplot2::labs(x = "time [d]", y = "parameter (transformed scale)",
                  title = sprintf("Parameter convergence, %s",
                                  if (unit_idx == 0) "shared (population) parameters" else paste("unit", unit_idx)),
                  subtitle = "Posterior mean (line) with 90% interval; dashed = true value",
                  caption = synthetic_caption) +
    dc_theme()
}

#' Parameter RMSE over time by method
#' @param param_error `param_error` of [summarize_metrics()].
#' @export
plot_param_rmse <- function(param_error) {
  pal <- method_palette(unique(param_error$method))
  ggplot2::ggplot(param_error, ggplot2::aes(x = time, y = rmse, colour = method)) +
    ggplot2::geom_line(linewidth = 0.7) +
    ggplot2::facet_wrap(~param, scales = "free_y") +
    ggplot2::scale_colour_manual(values = pal) +
    ggplot2::labs(x = "time [d]", y = "RMSE of the unit parameters (transformed scale)", colour = NULL,
                  title = "Parameter error over the season", caption = synthetic_caption) +
    dc_theme()
}

#' Posterior log-det of the shared parameters over time
#' @param logdet `logdet` of [summarize_metrics()].
#' @export
plot_logdet <- function(logdet) {
  pal <- method_palette(unique(logdet$method))
  ggplot2::ggplot(logdet, ggplot2::aes(x = time, y = logdet_shared, colour = method)) +
    ggplot2::geom_line(linewidth = 0.7) +
    ggplot2::scale_colour_manual(values = pal) +
    ggplot2::labs(x = "time [d]", y = "log det posterior covariance (shared parameters)", colour = NULL,
                  title = "Uncertainty reduction", subtitle = "Lower = more learned",
                  caption = synthetic_caption) +
    dc_theme()
}

#' 90% interval coverage over time
#' @param coverage `coverage` of [summarize_metrics()].
#' @param level nominal level.
#' @export
plot_coverage <- function(coverage, level = 0.9) {
  pal <- method_palette(unique(coverage$method))
  d <- coverage[, .(coverage = mean(coverage)), by = .(method, time)]
  ggplot2::ggplot(d, ggplot2::aes(x = time, y = coverage, colour = method)) +
    ggplot2::geom_hline(yintercept = level, colour = dc_colors$ink2, linetype = "dashed", linewidth = 0.4) +
    ggplot2::geom_line(linewidth = 0.7) +
    ggplot2::scale_colour_manual(values = pal) +
    ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::labs(x = "time [d]", y = "fraction of true values inside the 90% interval", colour = NULL,
                  title = "Interval coverage (all parameters and units)",
                  subtitle = "Dashed = nominal 0.9", caption = synthetic_caption) +
    dc_theme()
}

#' Identifiability: Fisher-information spectrum and bounds
#' @param ident result of [identifiability_report()].
#' @return list(p_spectrum, p_bounds).
#' @export
plot_identifiability <- function(ident) {
  sc_pal <- stats::setNames(dc_colors$series[1:3], c("disturbances", "probing_routine", "probing"))
  sp <- unique(ident$spectrum[, .(scenario, k, eigenvalue)])
  sp[, scenario := factor(scenario, levels = names(sc_pal))]
  p_spectrum <- ggplot2::ggplot(sp, ggplot2::aes(x = k, y = pmax(eigenvalue, 1e-12), colour = scenario)) +
    ggplot2::geom_line(linewidth = 0.7) +
    ggplot2::geom_point(size = 2.6) +
    ggplot2::scale_y_log10() +
    ggplot2::scale_colour_manual(values = sc_pal) +
    ggplot2::labs(x = "eigenvalue index", y = "Fisher information eigenvalue (log scale)", colour = NULL,
                  title = "Identifiability spectrum over one season",
                  subtitle = "Small eigenvalues = parameter directions the data barely inform",
                  caption = synthetic_caption) +
    dc_theme()
  b <- data.table::copy(ident$bounds)
  b[, scenario := factor(scenario, levels = names(sc_pal))]
  p_bounds <- ggplot2::ggplot(b, ggplot2::aes(x = param, y = ratio, fill = scenario)) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.8), width = 0.7) +
    ggplot2::geom_hline(yintercept = 1, colour = dc_colors$axis, linewidth = 0.4) +
    ggplot2::scale_fill_manual(values = sc_pal) +
    ggplot2::labs(x = NULL, y = "posterior-sd bound / prior sd", fill = NULL,
                  title = "Which parameters the disturbances identify",
                  subtitle = "1 = nothing learned; low = well identified",
                  caption = synthetic_caption) +
    dc_theme()
  list(p_spectrum = p_spectrum, p_bounds = p_bounds)
}

#' Advisor ranking over a season
#' @param ranking advisor ranking table (all epochs) of [run_season()].
#' @export
plot_advisor_sequence <- function(ranking) {
  d <- data.table::copy(ranking)
  d[, label := ifelse(kind == "event", "now", sprintf("%+.2g", step))]
  d[, top := rank == 1]
  ggplot2::ggplot(d, ggplot2::aes(x = epoch, y = control, fill = saving_mean)) +
    ggplot2::geom_tile(colour = dc_colors$surface, linewidth = 0.8) +
    ggplot2::geom_tile(data = d[top == TRUE], fill = NA, colour = dc_colors$ink, linewidth = 0.6) +
    ggplot2::geom_text(ggplot2::aes(label = label), colour = dc_colors$ink, size = 2.6) +
    ggplot2::scale_fill_gradient(low = dc_colors$seq_lo, high = dc_colors$seq_hi) +
    ggplot2::labs(x = "decision epoch", y = NULL, fill = "expected saving [EUR]",
                  title = "Advisor: recommended change per control and epoch",
                  subtitle = "Label = recommended step; outlined = ranked first (best saving per unit cost)",
                  caption = synthetic_caption) +
    dc_theme() +
    ggplot2::theme(legend.key.width = ggplot2::unit(1.5, "cm"))
}

#' Shapley attribution of season savings
#' @param attribution result of [season_attribution()].
#' @export
plot_shapley <- function(attribution) {
  d <- data.table::copy(attribution$shapley)
  d[, sign := ifelse(saving >= 0, "saving", "loss")]
  d[, player := factor(player, levels = d[order(saving)]$player)]
  ggplot2::ggplot(d, ggplot2::aes(x = saving / 1000, y = player, fill = sign)) +
    ggplot2::geom_col(width = 0.6) +
    ggplot2::geom_vline(xintercept = 0, colour = dc_colors$axis, linewidth = 0.4) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.1f", saving / 1000),
                                    hjust = ifelse(saving >= 0, -0.15, 1.15)),
                       colour = dc_colors$ink2, size = 3) +
    ggplot2::scale_fill_manual(values = c(saving = dc_colors$pos, loss = dc_colors$neg), guide = "none") +
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = 0.2)) +
    ggplot2::labs(x = "Shapley share of season savings vs calendar baseline [kEUR]", y = NULL,
                  title = "Which adopted changes produced the savings",
                  subtitle = sprintf("Total %.1f kEUR; %s gates, %s space; open-loop replay on the true plant",
                                     attribution$total_saving / 1000, attribution$gate_by, attribution$space),
                  caption = synthetic_caption) +
    dc_theme()
}

#' Controls, events and fouling state of one season
#' @param res result of [run_season()] with `keep_obs = TRUE`.
#' @param state_col true state column to show.
#' @param obs_col noisy observation column to overlay (optional).
#' @return list of plots, one per control plus the state.
#' @export
plot_season <- function(res, state_col, obs_col = NULL) {
  ep <- res$epochs
  ucols <- grep("^u_", names(ep), value = TRUE)
  units <- sort(unique(ep$unit))
  upal <- stats::setNames(dc_colors$series[seq_along(units)], as.character(units))
  out <- list()
  uc <- ucols[1]                                      # debugging line
  for (uc in ucols) {
    d <- ep[, c("unit", "arm", "time", "time_end", uc), with = FALSE]
    data.table::setnames(d, uc, "value")
    d[, unit := as.character(unit)]
    out[[uc]] <- ggplot2::ggplot(d, ggplot2::aes(colour = unit)) +
      ggplot2::geom_segment(ggplot2::aes(x = time, xend = time_end, y = value, yend = value), linewidth = 0.7) +
      ggplot2::scale_colour_manual(values = upal) +
      ggplot2::labs(x = "time [d]", y = sub("^u_", "", uc), colour = "unit",
                    title = sprintf("Control '%s' per unit (%s)", sub("^u_", "", uc), res$method),
                    caption = synthetic_caption) +
      dc_theme()
  }
  if (!is.null(res$observations) && state_col %in% names(res$observations)) {
    ob <- data.table::copy(res$observations)
    ob[, unit := as.character(unit)]
    ev <- ep[n_events > 0, .(unit = as.character(unit), time)]
    p <- ggplot2::ggplot(ob, ggplot2::aes(x = time, colour = unit)) +
      ggplot2::geom_line(ggplot2::aes(y = .data[[state_col]]), linewidth = 0.7)
    if (!is.null(obs_col) && obs_col %in% names(ob)) {
      p <- p + ggplot2::geom_point(ggplot2::aes(y = .data[[obs_col]]), size = 0.8, alpha = 0.5)
    }
    if (nrow(ev) > 0) {
      p <- p + ggplot2::geom_vline(data = ev, ggplot2::aes(xintercept = time, colour = unit),
                                   linetype = "dotted", linewidth = 0.4)
    }
    out$state <- p +
      ggplot2::scale_colour_manual(values = upal) +
      ggplot2::labs(x = "time [d]", y = state_col, colour = "unit",
                    title = sprintf("True %s per unit (%s); dotted = events", state_col, res$method),
                    subtitle = if (!is.null(obs_col)) paste("points:", obs_col, "(noisy, where measured)") else NULL,
                    caption = synthetic_caption) +
      dc_theme()
  }
  out
}

#' Ablation results
#' @param ablations result of [run_ablations()].
#' @export
plot_ablations <- function(ablations) {
  d <- ablations$regret[method != "oracle"]
  pal <- method_palette(unique(d$method))
  ggplot2::ggplot(d, ggplot2::aes(x = value, y = mean / 1000, colour = method)) +
    ggplot2::geom_hline(yintercept = 0, colour = dc_colors$axis, linewidth = 0.4) +
    ggplot2::geom_pointrange(ggplot2::aes(ymin = lo / 1000, ymax = hi / 1000),
                             position = ggplot2::position_dodge(width = 0.5), linewidth = 0.6) +
    ggplot2::facet_wrap(~ablation, scales = "free_x") +
    ggplot2::scale_colour_manual(values = pal) +
    ggplot2::labs(x = NULL, y = "season regret vs oracle [kEUR]", colour = NULL,
                  title = "Ablations", subtitle = "Mean and 95% CI over paired seeds",
                  caption = synthetic_caption) +
    dc_theme()
}

#' Probe-arm cost versus the value of what it taught
#' @param probe_value `probe_value` of [summarize_metrics()].
#' @export
plot_probe_value <- function(probe_value) {
  d <- data.table::rbindlist(list(
    probe_value[, .(method, what = "probe-arm extra cost", mean_val = probe_cost, lo_val = probe_cost_lo, hi_val = probe_cost_hi)],
    probe_value[, .(method, what = "production-arm saving", mean_val = probe_value, lo_val = probe_value_lo, hi_val = probe_value_hi)]))
  ggplot2::ggplot(d, ggplot2::aes(x = mean_val / 1000, y = method, colour = what)) +
    ggplot2::geom_vline(xintercept = 0, colour = dc_colors$axis, linewidth = 0.4) +
    ggplot2::geom_pointrange(ggplot2::aes(xmin = lo_val / 1000, xmax = hi_val / 1000),
                             position = ggplot2::position_dodge(width = 0.5), linewidth = 0.6) +
    ggplot2::scale_colour_manual(values = c("probe-arm extra cost" = dc_colors$neg,
                                            "production-arm saving" = dc_colors$pos)) +
    ggplot2::labs(x = "kEUR per cohort-season (mean, 95% CI) vs reference method", y = NULL, colour = NULL,
                  title = "Probe arm: cost versus value of what it taught",
                  caption = synthetic_caption) +
    dc_theme()
}
