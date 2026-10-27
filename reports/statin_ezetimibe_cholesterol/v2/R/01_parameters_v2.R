# =============================================================================
# 01_parameters_v2.R
# v2 of the cholesterol / rosuvastatin / ezetimibe model:
#   + bile-acid (BA) enterohepatic circulation: gallbladder, jejunum, ileum,
#     ileal reabsorption (ASBT), FXR/FGF19 repression of CYP7A1
#   + biliary cholesterol secretion coupled to BA secretion; stored in the
#     gallbladder together with bile acids and ezetimibe
#   + micelle-gated cholesterol absorption (bile acids + fat digestion products),
#     plant sterols compete for micellar space
#   + meal composition: cholesterol, fat (CCK -> gallbladder, micelles),
#     viscous fibre (binds BA), plant sterols
#   + NPC1L1 occupancy by ezetimibe: kinetic (slow off-rate) or equilibrium
#   + optional circadian modulation of hepatic synthesis
#
# Units: cholesterol and BA amounts [mmol], fluxes [mmol/d]; fat, fibre,
#        plant sterols [g] and [g/d]; drugs [mg], [mg/L]; volumes [L];
#        time [d] (parameters given in hours carry the suffix _h)
#
# dX/dt = f(X, p, u(t), w(t)):
#   u(t): oral intake rates of rosuvastatin and ezetimibe [mg/d], regimen p_u
#   w(t): meal clock and meal composition (D_in, fat_in, fib_in, ps_in)
#
# Calibration: derive_parms_v2() sets rate constants from daily-mean flux
# targets with a mean-field argument (constant inputs). Because the true
# inputs are pulsed and the model is nonlinear, the simulated reference
# baseline lands close to, not exactly on, these targets.
# =============================================================================

make_parms_v2 <- function() {
  parms <- list(
    MW_chol = 386.65,            # molar mass cholesterol [mg/mmol]
    MW_ps   = 414.7,             # molar mass sitosterol (plant sterols) [mg/mmol]

    # ---------- reference patient, daily-mean calibration targets ----------
    C_L0      = 10,              # hepatic regulatory free-cholesterol pool [mmol]
    LDL0      = 3.0,             # plasma LDL-C [mmol/L]
    C_P0      = 100,             # exchangeable peripheral tissue cholesterol [mmol]
    V_p       = 3.0,             # plasma volume [L]
    D_ref     = 300 / 386.65,    # dietary cholesterol at calibration [mmol/d]
    fat_ref_g = 80,              # dietary fat at calibration [g/d]
    B0        = 2.6,             # biliary cholesterol secretion [mmol/d]
    BA0       = 1.2,             # BA synthesis = faecal BA loss (CYP7A1) [mmol/d]
    f_abs0    = 0.44,            # fractional absorption of (diet + biliary) cholesterol [-]
    s_mic0    = 0.6,             # mean micellar solubilised fraction of lumen cholesterol [-]
    S_hep0    = 1.42,            # hepatic de novo synthesis [mmol/d]
    g0        = 0.25,            # SREBP2 activity at baseline [-]
    n_g       = 4,               # Hill coefficient SREBP2 sensor [-]
    m_R       = 1,               # elasticity LDLR to SREBP2 (net of PCSK9) [-]
    q_V       = 0.5,             # elasticity VLDL->LDL production to C_L [-]
    FCR0      = 0.35,            # LDL fractional catabolic rate [1/d]
    frac_hepR = 0.70,            # share of LDL clearance by hepatic LDLR [-]
    tau_R     = 0.75,            # LDLR relaxation time [d]
    rct_quasi_ss = TRUE,         # TRUE: RCT = S_P + U_per (removes the slow C_P mode) [-]

    # ---------- bile-acid circulation ----------
    f_reabs0    = 0.97,          # ileal BA reabsorption per pass (ASBT) [-]
    tau_jej_h   = 2,             # residence time chyme/BA in absorptive (proximal) small intestine [h]
    tau_ileum_h = 1.5,           # residence time BA at ileal reabsorption site [h]
    nu_B        = 0.6,           # elasticity biliary cholesterol to BA secretion rate [-] (<1: lithogenic when fasting)
    K_Z         = 1.5,           # FXR/FGF19 repression constant of CYP7A1 [-]
    n_Z         = 2,             # Hill coefficient FXR/FGF19 repression [-]
    tau_Z_h     = 3,             # time constant of FGF19/FXR signal [h]

    # ---------- gallbladder and CCK ----------
    phi_fast    = 0.7,           # fraction of hepatic bile diverted to gallbladder when fasting [-]
    k_gb_mmc    = 1,             # fasting (MMC-averaged) gallbladder emptying [1/d]
    k_gb_max    = 40,            # gallbladder emptying at full CCK activity [1/d]
    Fat_ref_cck = 10,            # intestinal fat giving CCK stimulus 1 [g]
    K_cck       = 0.5,           # CCK level at half-maximal gallbladder/sphincter response [-]
    tau_cck_h   = 0.75,          # CCK response time [h]

    # ---------- intestinal nutrients and micelles ----------
    k_fat       = 6,             # fat digestion + absorption rate [1/d]
    m0_mic      = 0.3,           # micellar capacity without fat digestion products, relative [-]
    K_fat_mic   = 10,            # intestinal fat at half fat-boost of micellar capacity [g]
    a_ps        = 1.5,           # micellar affinity of plant sterols relative to cholesterol [-]
    b_max_fib   = 0.5,           # max fraction of ileal BA bound by viscous fibre [-]
    K_fib       = 2,             # fibre amount at half-maximal BA binding [g]
    k_tr_fib    = 4,             # fibre transit out of small intestine [1/d]

    # ---------- circadian hepatic synthesis ----------
    a_circ      = 0.3,           # relative amplitude of hepatic synthesis rhythm [-] (0 = off)
    circ_peak_h = 0,             # clock time of maximal synthesis [h]

    # ---------- patient factors ----------
    S_max_factor = 1,            # hepatic HMGCR capacity [-]
    K_C_factor   = 1,            # SREBP2 set point [-]
    k_abs_factor = 1,            # NPC1L1 capacity [-]
    k_B_factor   = 1,            # ABCG5/G8 biliary cholesterol secretion [-]
    f_OATP       = 1,            # OATP1B1 rosuvastatin uptake [-]

    # ---------- drivers w(t): diet (daily totals) and meal schedule ----------
    D_daily     = 300 / 386.65,  # dietary cholesterol [mmol/d]
    fat_daily_g = 80,            # dietary fat (CCK-stimulating fat-equivalent) [g/d]
    fib_daily_g = 0,             # extra viscous fibre, e.g. psyllium [g/d]
    ps_daily_g  = 0,             # plant sterols/stanols [g/d]
    meal_h      = c(8, 13, 19),  # meal clock times [h]
    chol_frac   = c(0.25, 0.35, 0.40), # share of daily cholesterol per meal [-]
    fat_frac    = c(0.25, 0.35, 0.40), # share of daily fat per meal [-]
    fib_frac    = c(0.25, 0.35, 0.40), # share of daily fibre per meal [-]
    ps_frac     = c(0.25, 0.35, 0.40), # share of daily plant sterols per meal [-]
    meal_sd_h   = 1.0,           # width of nutrient delivery pulse (gastric emptying) [h]

    # ---------- control p_u (dosing regimen) ----------
    dose_rsv_mg = 0,             # rosuvastatin daily dose [mg/d]
    dose_eze_mg = 0,             # ezetimibe daily dose [mg/d]
    start_rsv_d = 0,             # first day rosuvastatin [d]
    start_eze_d = 0,             # first day ezetimibe [d]
    dose_hour   = 8,             # clock time of pill intake [h]
    dose_sd_h   = 0.5,           # width of oral input pulse [h]

    # ---------- rosuvastatin PK (as v1) ----------
    F_rsv      = 0.20,           # bioavailability [-]
    ka_rsv     = 12,             # absorption [1/d]
    V_c_rsv    = 100,            # central volume [L]
    V_per_rsv  = 300,            # peripheral volume [L]
    Q_rsv      = 150,            # inter-compartmental clearance [L/d]
    CL_ren_rsv = 330,            # renal clearance [L/d]
    CL_upt_rsv = 1500,           # OATP1B1 uptake clearance at f_OATP = 1 [L/d]
    k_hc_rsv   = 30,             # sinusoidal efflux [1/d]
    k_bil_rsv  = 30,             # biliary excretion [1/d]
    V_h        = 1.5,            # liver volume [L]

    # ---------- ezetimibe PK; biliary part now routed via gallbladder ----------
    ka_eze     = 12,             # uptake gut -> plasma/liver [1/d]
    k_fec_eze  = 6,              # faecal loss from gut [1/d]
    k_bile_eze = 6,              # excretion into hepatic bile [1/d]
    k_ren_eze  = 0.15,           # renal elimination [1/d]

    # ---------- pharmacodynamics ----------
    Imax_S       = 0.90,         # max HMGCR inhibition [-]
    IC50_S       = NA_real_,     # hepatic rosuvastatin at half Imax [mg/L]; calibrate_pd_v2()
    I_S_ref      = 0.70,         # target mean HMGCR inhibition at reference dose [-]
    dose_ref_rsv = 10,           # [mg/d]
    Emax_E       = 0.75,         # max NPC1L1 inhibition (= full occupancy) [-]
    I_E_ref      = 0.60,         # target mean NPC1L1 inhibition at reference regimen [-]
    dose_ref_eze = 10,           # [mg/d]
    occupancy_kinetic = TRUE,    # TRUE: occupancy state O_NPC; FALSE: instantaneous equilibrium [-]
    k_off_NPC    = 1,            # dissociation rate ezetimibe-glucuronide from NPC1L1 [1/d] (assumed, t1/2 ~ 17 h)
    k_on_NPC     = 0,            # association rate [1/(mg d)]; calibrate_pd_v2()
    EC50_E       = NA_real_      # gut ezetimibe at half Emax, equilibrium mode [mg]; calibrate_pd_v2()
  )
  parms
}

# -----------------------------------------------------------------------------
# derive_parms_v2(): targets -> rate constants (mean-field), patient factors.
# Idempotent: always recomputes from base fields.
# -----------------------------------------------------------------------------
derive_parms_v2 <- function(parms) {
  list2env(parms, envir = environment())

  # time constants from hours to days
  k_tr    <- 24 / tau_jej_h                         # proximal SI transit [1/d]
  k_ileum <- 24 / tau_ileum_h                       # ileal exit rate (reabsorbed + to colon) [1/d]
  tau_cck <- tau_cck_h / 24                         # [d]
  tau_Z   <- tau_Z_h / 24                           # [d]

  # bile acids
  J_syn0     <- BA0                                 # BA synthesis [mmol/d]
  J_BA0      <- J_syn0 / (1 - f_reabs0)             # hepatic BA secretion [mmol/d]
  k_asbt     <- f_reabs0 * k_ileum                  # ileal reabsorption [1/d]
  k_col      <- (1 - f_reabs0) * k_ileum            # BA loss to colon [1/d]
  J_asbt_ref <- J_BA0 - J_syn0                      # portal BA return [mmol/d]
  k_syn_BA   <- BA0 / C_L0                          # CYP7A1 rate constant at Z = 1 [1/d]

  # micelle-gated absorption, mean-field: A = k_abs * s_mic * C_lum
  k_abs_eff0 <- f_abs0 / (1 - f_abs0) * k_tr        # effective absorption [1/d]
  k_abs_ref  <- k_abs_eff0 / s_mic0                 # absorption of micellar cholesterol [1/d]
  C_lum0     <- (D_ref + B0) / (k_tr + k_abs_eff0)  # lumen cholesterol [mmol]
  cap0       <- C_lum0 * s_mic0 / (1 - s_mic0)      # micellar capacity [mmol]
  BA_J0      <- J_BA0 / k_tr                        # jejunal BA [mmol]
  Fat_J0     <- fat_ref_g / k_fat                   # intestinal fat [g]
  fat_fac0   <- m0_mic + (1 - m0_mic) * Fat_J0 / (K_fat_mic + Fat_J0)  # [-]
  r_mic      <- cap0 / (BA_J0 * fat_fac0)           # cholesterol capacity per BA [mmol/mmol]

  # whole-body balance (as v1)
  A0        <- f_abs0 * (D_ref + B0)                # [mmol/d]
  S_P       <- B0 + BA0 - A0 - S_hep0               # peripheral synthesis [mmol/d]
  if (S_P < 0) stop("Calibration infeasible: S_P < 0")
  K_C_ref   <- C_L0 / (1 / g0 - 1)^(1 / n_g)        # [mmol]
  S_max_ref <- S_hep0 / g0                          # [mmol/d]
  P_V0      <- FCR0 * LDL0 * V_p                    # [mmol/d]
  k_U       <- frac_hepR * FCR0                     # [1/d]
  k_nR      <- (1 - frac_hepR) * FCR0               # [1/d]
  k_RCT     <- (S_P + k_nR * LDL0 * V_p) / C_P0     # [1/d]

  derived <- list(
    k_tr = k_tr, k_ileum = k_ileum, tau_cck = tau_cck, tau_Z = tau_Z,   # [1/d], [1/d], [d], [d]
    J_BA0 = J_BA0, k_asbt = k_asbt, k_col = k_col,                       # [mmol/d], [1/d], [1/d]
    J_asbt_ref = J_asbt_ref, k_syn_BA = k_syn_BA,                        # [mmol/d], [1/d]
    k_abs = k_abs_ref * k_abs_factor,                                    # [1/d]
    r_mic = r_mic, C_lum0 = C_lum0, BA_J0 = BA_J0,                       # [-], [mmol], [mmol]
    S_P = S_P, K_C = K_C_ref * K_C_factor, S_max = S_max_ref * S_max_factor,  # [mmol/d], [mmol], [mmol/d]
    P_V0 = P_V0, k_U = k_U, k_nR = k_nR, k_RCT = k_RCT                   # [mmol/d], [1/d], [1/d], [1/d]
  )
  modifyList(parms, derived)
}

# state order (deSolve: y and derivative vector in the same order)
GUT_PK_STATES <- c("Fat_J", "CCK", "Fib_I", "PS_J",
                   "A_rg", "A_rc", "A_rp", "A_rh",
                   "E_gut", "E_pl", "E_gb", "O_NPC")
#  Fat_J [g]    fat in small intestine        CCK  [-]  CCK level (1 ~ 10 g fat)
#  Fib_I [g]    viscous fibre in SI            PS_J [mmol] plant sterols in SI lumen
#  A_rg, A_rc, A_rp, A_rh [mg] rosuvastatin gut, central, peripheral, liver
#  E_gut, E_pl, E_gb [mg] ezetimibe(+glucuronide) gut, plasma/liver, gallbladder
#  O_NPC [-]    fraction of NPC1L1 occupied by ezetimibe
BA_CHOL_STATES <- c("BA_GB", "BA_J", "BA_I", "Z_FXR",
                    "C_GB", "C_lum", "C_L", "R_LDLR", "LDL", "C_P")
#  BA_GB, BA_J, BA_I [mmol] bile acids in gallbladder, jejunum, ileum
#  Z_FXR [-]    FXR/FGF19 signal relative to reference
#  C_GB  [mmol] cholesterol in gallbladder bile
#  C_lum [mmol] cholesterol in SI lumen (diet + bile)
#  C_L [mmol], R_LDLR [-], LDL [mmol/L], C_P [mmol] as in v1
ALL_STATES_V2 <- c(GUT_PK_STATES, BA_CHOL_STATES)
