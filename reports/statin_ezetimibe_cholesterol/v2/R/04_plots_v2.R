# =============================================================================
# 04_plots_v2.R  -  visualisation only
# =============================================================================

theme_model <- function() {
  ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank(),
                   legend.position  = "bottom")
}

PATTERN_LABELS <- c(three_meals = "3 meals (8, 13, 19 h)",
                    tre_16_8    = "16:8 TRE (12, 19 h)",
                    omad        = "one meal (18 h)")

# readable, ordered factor of meal patterns
pattern_factor <- function(pattern) {
  factor(PATTERN_LABELS[pattern], levels = PATTERN_LABELS)
}

# A1: LDL change per meal pattern and treatment
plot_pattern_effects <- function(summary_dt) {
  dt <- summary_dt[experiment == "A_meal_pattern"]
  addon <- merge(dt[treatment == "rsv_eze", .(pattern, LDL_both = LDL)],
                 dt[treatment == "rsv",     .(pattern, LDL_rsv = LDL)], by = "pattern")
  addon[, `:=`(treatment = "eze on top of rsv", pct_LDL = 100 * (LDL_both / LDL_rsv - 1))]
  plot_dt <- rbindlist(list(dt[treatment != "none", .(pattern, treatment, pct_LDL)],
                            addon[, .(pattern, treatment, pct_LDL)]))
  plot_dt[, treatment := factor(treatment, levels = c("eze", "rsv", "rsv_eze", "eze on top of rsv"))]
  base_lab <- dt[treatment == "none", .(pattern, panel = sprintf("%s\nbaseline LDL-C %.0f mg/dL",
                                                                PATTERN_LABELS[pattern], LDL_mgdL))]
  plot_dt <- merge(plot_dt, base_lab, by = "pattern")
  plot_dt[, panel := factor(panel, levels = base_lab$panel[order(match(base_lab$pattern, names(PATTERN_LABELS)))])]
  ggplot(plot_dt, aes(x = treatment, y = pct_LDL, fill = treatment)) +
    geom_col() +
    geom_text(aes(label = sprintf("%.1f", pct_LDL)), vjust = 1.3, size = 3) +
    facet_wrap(~panel) +
    labs(x = NULL, y = "LDL-C change vs drug-free, same meal pattern [%]",
         title = "A. Meal pattern at equal daily intake: drug effects") +
    theme_model() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")
}

# A2: daily-mean mechanism variables per meal pattern (drug-free)
plot_pattern_mechanism <- function(summary_dt) {
  dt <- summary_dt[experiment == "A_meal_pattern" & treatment == "none"]
  vars <- c(J_sec = "BA secretion [mmol/d]", BA_cycles = "BA cycles [1/d]",
            B = "biliary cholesterol [mmol/d]", B_night = "biliary cholesterol 2-6 h [mmol/d]",
            A = "cholesterol absorbed [mmol/d]", f_abs = "fractional absorption [-]",
            J_syn = "BA synthesis CYP7A1 [mmol/d]", litho_max = "max cholesterol/BA in gallbladder [-]")
  plot_dt <- melt(dt, id.vars = "pattern", measure.vars = names(vars),
                  variable.name = "variable", value.name = "value")
  plot_dt[, variable := factor(vars[as.character(variable)], levels = vars)]
  plot_dt[, pattern := pattern_factor(pattern)]
  ggplot(plot_dt, aes(x = pattern, y = value, fill = pattern)) +
    geom_col() +
    facet_wrap(~variable, scales = "free_y", ncol = 4) +
    labs(x = NULL, y = NULL, title = "A. Why meal pattern matters (drug-free, daily means)") +
    theme_model() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")
}

# A3: within-day traces of the ezetimibe runs
plot_pattern_traces <- function(summary_dt, traces_dt, treatment_sel = "eze") {
  keys <- unique(summary_dt[experiment == "A_meal_pattern" & treatment == treatment_sel,
                            .(run_key, pattern)])
  dt <- merge(traces_dt, keys, by = "run_key")
  vars <- c(a_cck = "CCK activity [-]", BA_GB = "BA in gallbladder [mmol]",
            BA_J = "BA in jejunum [mmol]", B = "biliary cholesterol [mmol/d]",
            C_mic = "micellar cholesterol [mmol]", A = "absorption [mmol/d]",
            E_gut = "ezetimibe in gut [mg]", I_E = "NPC1L1 inhibition [-]")
  plot_dt <- melt(dt, id.vars = c("time", "pattern"), measure.vars = names(vars),
                  variable.name = "variable", value.name = "value")
  plot_dt[, variable := factor(vars[as.character(variable)], levels = vars)]
  plot_dt[, pattern := pattern_factor(pattern)]
  ggplot(plot_dt, aes(x = time * 24, y = value, colour = pattern)) +
    geom_line(linewidth = 0.7) +
    facet_wrap(~variable, scales = "free_y", ncol = 2) +
    scale_x_continuous(breaks = seq(0, 48, by = 6)) +
    labs(x = "clock time over two days [h]", y = NULL, colour = NULL,
         title = sprintf("A. Within-day dynamics (%s, periodic steady state)", treatment_sel)) +
    theme_model()
}

# B: pill timing x binding kinetics
plot_pill_timing <- function(summary_dt) {
  dt <- summary_dt[experiment == "B_pill_timing" & treatment == "eze"]
  dt[, binding := ifelse(occupancy_kinetic, "kinetic (k_off 1/d)", "instant equilibrium")]
  plot_dt <- melt(dt, id.vars = c("pattern", "dose_hour", "binding"),
                  measure.vars = c("pct_LDL", "I_E"), variable.name = "variable", value.name = "value")
  plot_dt[, variable := factor(variable, levels = c("pct_LDL", "I_E"),
                               labels = c("LDL-C change [%]", "mean NPC1L1 inhibition [-]"))]
  plot_dt[, pattern := pattern_factor(pattern)]
  ggplot(plot_dt, aes(x = dose_hour, y = value, colour = binding, linetype = pattern)) +
    geom_line(linewidth = 0.8) +
    geom_point() +
    facet_wrap(~variable, scales = "free_y") +
    scale_x_continuous(breaks = c(8, 14, 22)) +
    labs(x = "pill clock time [h]", y = NULL, colour = "NPC1L1 binding", linetype = NULL,
         title = "B. Does it matter when the ezetimibe pill is taken?") +
    theme_model()
}

# C: where cholesterol and fat are eaten
plot_composition <- function(summary_dt) {
  dt <- summary_dt[experiment == "C_composition"]
  wide <- merge(dt[treatment == "none", .(composition, LDL_none = LDL_mgdL, f_abs_none = f_abs)],
                dt[treatment == "eze",  .(composition, pct_eze = pct_LDL)], by = "composition")
  plot_dt <- melt(wide, id.vars = "composition", variable.name = "variable", value.name = "value")
  plot_dt[, variable := factor(variable, levels = c("LDL_none", "f_abs_none", "pct_eze"),
                               labels = c("drug-free LDL-C [mg/dL]", "fractional absorption [-]",
                                          "ezetimibe LDL-C change [%]"))]
  ggplot(plot_dt, aes(x = composition, y = value, fill = composition)) +
    geom_col() +
    geom_text(aes(label = sprintf("%.3g", value)), vjust = -0.3, size = 3) +
    facet_wrap(~variable, scales = "free_y") +
    labs(x = NULL, y = NULL, title = "C. Cholesterol eaten with or without fat, morning or evening") +
    theme_model() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")
}

# D: fibre and plant sterols, alone and with ezetimibe
plot_addons <- function(summary_dt) {
  dt <- copy(summary_dt[experiment == "D_addons"])
  LDL_ref <- dt[treatment == "none" & fib_daily_g == 0 & ps_daily_g == 0, LDL]
  dt[, diet := sprintf("fibre %g g/d, plant sterols %g g/d", fib_daily_g, ps_daily_g)]
  dt[, pct_vs_ref := 100 * (LDL / LDL_ref - 1)]               # vs drug-free reference diet [%]
  ggplot(dt, aes(x = diet, y = pct_vs_ref, fill = treatment)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_text(aes(label = sprintf("%.1f", pct_vs_ref)), position = position_dodge(width = 0.8),
              vjust = 1.3, size = 3) +
    labs(x = NULL, y = "LDL-C change vs drug-free reference diet [%]", fill = NULL,
         title = "D. Fibre and plant sterols with and without ezetimibe",
         subtitle = "plant sterols and ezetimibe act on the same step (micellar uptake): expect sub-additivity") +
    theme_model() +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))
}

# gallbladder bile composition over the day (drug-free)
plot_lithogenicity <- function(summary_dt, traces_dt) {
  keys <- unique(summary_dt[experiment == "A_meal_pattern" & treatment == "none",
                            .(run_key, pattern)])
  dt <- merge(traces_dt, keys, by = "run_key")
  plot_dt <- melt(dt, id.vars = c("time", "pattern"),
                  measure.vars = c("litho_GB", "J_sec", "B"),
                  variable.name = "variable", value.name = "value")
  plot_dt[, variable := factor(variable, levels = c("litho_GB", "J_sec", "B"),
                               labels = c("cholesterol/BA in gallbladder [mol/mol]",
                                          "hepatic BA secretion [mmol/d]",
                                          "biliary cholesterol secretion [mmol/d]"))]
  plot_dt[, pattern := pattern_factor(pattern)]
  ggplot(plot_dt, aes(x = time * 24, y = value, colour = pattern)) +
    geom_line(linewidth = 0.7) +
    facet_wrap(~variable, scales = "free_y", ncol = 1) +
    scale_x_continuous(breaks = seq(0, 48, by = 6)) +
    labs(x = "clock time over two days [h]", y = NULL, colour = NULL,
         title = "The liver keeps secreting while fasting; bile becomes more lithogenic") +
    theme_model()
}
