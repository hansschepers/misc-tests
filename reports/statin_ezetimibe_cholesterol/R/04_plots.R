# =============================================================================
# 04_plots.R  -  visualisation only (no simulation code here)
# =============================================================================

theme_model <- function() {
  ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank(),
                   legend.position  = "bottom")
}

# generic heat map of a scan result; fill = LDL change [%]
plot_heatmap <- function(dt, x_col, y_col, fill_col, x_lab, y_lab, fill_lab, title) {
  ggplot(dt, aes(x = factor(.data[[x_col]]), y = factor(.data[[y_col]]), fill = .data[[fill_col]])) +
    geom_tile(colour = "white") +
    geom_text(aes(label = sprintf("%.0f", .data[[fill_col]])), size = 3.2) +
    scale_fill_viridis_c(option = "magma", direction = 1) +
    labs(x = x_lab, y = y_lab, fill = fill_lab, title = title) +
    theme_model()
}

# ezetimibe add-on effect versus rosuvastatin dose, per synthesizer phenotype
plot_addon_vs_rsv <- function(dt) {
  plot_dt <- melt(dt, id.vars = c("S_max_factor", "dose_rsv_mg"),
                  measure.vars = c("pct_rsv", "pct_eze_addon", "pct_both"),
                  variable.name = "effect", value.name = "pct")
  plot_dt[, effect := factor(effect, levels = c("pct_rsv", "pct_eze_addon", "pct_both"),
                             labels = c("rosuvastatin vs none", "ezetimibe add-on vs rosuvastatin",
                                        "combination vs none"))]
  ggplot(plot_dt, aes(x = dose_rsv_mg, y = pct, colour = factor(S_max_factor))) +
    geom_line(linewidth = 0.8) +
    geom_point() +
    facet_wrap(~effect) +
    labs(x = "rosuvastatin dose [mg/d]", y = "LDL-C change at steady state [%]",
         colour = "HMGCR capacity factor [-]",
         title = "Statin background changes how much ezetimibe adds") +
    theme_model()
}

# one-at-a-time sensitivity (difference from reference patient, percentage points)
plot_tornado <- function(dt) {
  ref <- dt[level == "reference"]
  plot_dt <- melt(dt[level != "reference"], id.vars = c("label", "level"),
                  measure.vars = c("pct_eze_mono", "pct_eze_addon"),
                  variable.name = "outcome", value.name = "pct")
  plot_dt[outcome == "pct_eze_mono",  delta := pct - ref$pct_eze_mono]
  plot_dt[outcome == "pct_eze_addon", delta := pct - ref$pct_eze_addon]
  plot_dt[, outcome := factor(outcome, levels = c("pct_eze_mono", "pct_eze_addon"),
                              labels = c(sprintf("ezetimibe alone (ref %.1f %%)", ref$pct_eze_mono),
                                         sprintf("ezetimibe on rosuvastatin 10 mg (ref %.1f %%)",
                                                 ref$pct_eze_addon)))]
  ggplot(plot_dt, aes(x = delta, y = label, fill = level)) +
    geom_col(position = "identity", alpha = 0.8) +
    geom_vline(xintercept = 0) +
    facet_wrap(~outcome) +
    labs(x = "change of ezetimibe LDL-C effect vs reference [percentage points]\n(negative = stronger lowering)",
         y = NULL, fill = "parameter level",
         title = "Which patient parameters set the ezetimibe response?") +
    theme_model()
}

# compensation of the lost absorption at the new steady state
plot_compensation <- function(dt) {
  plot_dt <- copy(dt)
  plot_dt[, panel := sprintf("HMGCR factor %.2g (LDL %.1f %%)", S_max_factor, pct_LDL)]
  ggplot(plot_dt, aes(x = flux, y = delta, fill = flux)) +
    geom_col() +
    geom_hline(yintercept = 0) +
    facet_wrap(~panel) +
    labs(x = NULL, y = "steady-state change caused by ezetimibe [mmol/d]",
         title = "Lost absorption is compensated by synthesis and lower biliary/bile-acid output",
         subtitle = "balance: dA + dS_hep = dB + dBA ; LDL is a read-out of the hepatic pool") +
    theme_model() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")
}

# LDL-C time course for several patients
plot_ldl_timecourse <- function(dt, start_rsv_d, start_eze_d) {
  ggplot(dt, aes(x = time, y = LDL_mgdL, colour = patient)) +
    geom_line(linewidth = 0.7) +
    geom_vline(xintercept = c(start_rsv_d, start_eze_d), linetype = 2) +
    annotate("text", x = c(start_rsv_d, start_eze_d), y = Inf, vjust = 1.5, hjust = -0.05,
             label = c("rosuvastatin", "+ ezetimibe"), size = 3.2) +
    labs(x = "time [d]", y = "plasma LDL-C [mg/dL]", colour = NULL,
         title = "Dynamic response: statin first, ezetimibe added later") +
    theme_model()
}

# daily-averaged hepatic fluxes for one patient
plot_daily_fluxes <- function(dt) {
  flux_cols <- c("A", "S_hep", "B", "BA", "U_hep", "P_V")
  daily <- dt[, lapply(.SD, mean), by = .(day = floor(time)), .SDcols = flux_cols]
  plot_dt <- melt(daily, id.vars = "day", variable.name = "flux", value.name = "rate")
  ggplot(plot_dt, aes(x = day, y = rate, colour = flux)) +
    geom_line(linewidth = 0.8) +
    labs(x = "time [d]", y = "daily mean flux [mmol/d]", colour = NULL,
         title = "Hepatic cholesterol fluxes (reference patient)") +
    theme_model()
}

# within-day PK/PD over the last days of the simulation
plot_intraday <- function(dt, n_last_days = 3) {
  end_d   <- max(dt$time)
  plot_dt <- melt(dt[time >= end_d - n_last_days],
                  id.vars = "time", measure.vars = c("I_S", "I_E", "D_in", "k_gb"),
                  variable.name = "quantity", value.name = "value")
  plot_dt[, quantity := factor(quantity, levels = c("I_S", "I_E", "D_in", "k_gb"),
                               labels = c("HMGCR inhibition I_S [-]", "NPC1L1 inhibition I_E [-]",
                                          "dietary intake D_in [mmol/d]",
                                          "gallbladder emptying k_gb [1/d]"))]
  ggplot(plot_dt, aes(x = time, y = value)) +
    geom_line(linewidth = 0.6) +
    facet_wrap(~quantity, scales = "free_y", ncol = 1) +
    labs(x = "time [d]", y = NULL,
         title = "Within-day drivers w(t) and drug inhibition (ezetimibe peaks at meals via EHC)") +
    theme_model()
}
