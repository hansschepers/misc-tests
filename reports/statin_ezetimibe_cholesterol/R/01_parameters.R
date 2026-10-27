# =============================================================================
# 01_parameters.R
# Coupled whole-body cholesterol / rosuvastatin / ezetimibe model
#
# Units convention (whole file):
#   cholesterol amounts [mmol], cholesterol fluxes [mmol/d]
#   drug amounts [mg], drug concentrations [mg/L]
#   volumes [L], time [d]
#   MW cholesterol = 386.65 [mg/mmol];  LDL-C [mmol/L] x 38.67 = [mg/dL]
#
# Structure:  dX/dt = f(X, p, u(t), w(t));  Y = g(X, p, u, w)
#   X : C_lum, C_L, R_LDLR, LDL, C_P (cholesterol) + 7 drug PK states
#   u : oral intake rates u_rsv(t), u_eze(t) [mg/d] from the regimen p_u
#       (dose_rsv_mg, dose_eze_mg, start_rsv_d, start_eze_d, dose_hour, dose_sd_h)
#   w : dietary cholesterol D_daily and meal clock (meal_h, meal_frac),
#       meal-driven gallbladder emptying
#
# The reference patient is calibrated to steady state from literature-typical
# whole-body fluxes (no drug); derive_parms() turns flux targets into rate
# constants, so that any change of a target keeps the baseline in balance.
# Values are textbook orders of magnitude, NOT a fit to a data set.
# =============================================================================

make_parms <- function() {
  parms <- list(
    MW_chol   = 386.65,          # molar mass cholesterol [mg/mmol]

    # ---------- reference patient, no drug: calibration targets ----------
    C_L0      = 10,              # hepatic regulatory free-cholesterol pool [mmol] (~3.9 g)
    LDL0      = 3.0,             # plasma LDL-C [mmol/L] (= 116 mg/dL)
    C_P0      = 100,             # exchangeable peripheral (non-liver) tissue cholesterol [mmol]
    V_p       = 3.0,             # plasma volume [L]
    D_ref     = 300 / 386.65,    # dietary cholesterol at calibration [mmol/d] (300 mg/d)
    B0        = 2.6,             # biliary cholesterol secretion (ABCG5/G8) [mmol/d] (~1000 mg/d)
    BA0       = 1.2,             # cholesterol converted to bile acids (CYP7A1) [mmol/d] (~460 mg/d)
    f_abs0    = 0.44,            # fractional absorption of lumen cholesterol (diet + bile) [-]
    tau_lum   = 0.5,             # mean residence time of cholesterol in gut lumen [d]
    S_hep0    = 1.42,            # hepatic de novo synthesis [mmol/d] (~550 mg/d)
    g0        = 0.25,            # SREBP2 activity at baseline, fraction of maximum [-]
    n_g       = 4,               # Hill coefficient of the SREBP2 sterol sensor [-]
    m_R       = 1,               # elasticity of LDLR activity to SREBP2 activity [-] (PCSK9 lowers it)
    q_V       = 0.5,             # elasticity of VLDL->LDL production to hepatic C_L [-]
    FCR0      = 0.35,            # LDL fractional catabolic rate [1/d]
    frac_hepR = 0.70,            # share of LDL clearance by hepatic LDLR at baseline [-]
    tau_R     = 0.75,            # relaxation time of LDLR activity (SREBP2 -> LDLR protein) [d]

    # ---------- patient factors (1 = reference patient) ----------
    S_max_factor = 1,            # hepatic HMGCR capacity multiplier [-]  ("synthesizer" phenotype)
    K_C_factor   = 1,            # SREBP2 sensor set-point multiplier [-]
    k_abs_factor = 1,            # NPC1L1 absorption capacity multiplier [-] ("absorber" phenotype)
    k_B_factor   = 1,            # ABCG5/G8 biliary cholesterol secretion multiplier [-]
    f_OATP       = 1,            # OATP1B1 hepatic uptake activity of rosuvastatin (SLCO1B1) [-]

    # ---------- uncontrollable drivers w(t) ----------
    D_daily    = 300 / 386.65,   # dietary cholesterol intake [mmol/d]
    meal_h     = c(8, 13, 19),   # meal clock times [h]
    meal_frac  = c(0.25, 0.35, 0.40), # share of daily cholesterol intake per meal [-]
    meal_sd_h  = 0.5,            # width (sd) of meal intake pulse [h]
    k_gb_basal = 0.5,            # basal gallbladder emptying rate [1/d]
    k_gb_peak  = 40,             # peak meal-induced gallbladder emptying rate [1/d]
    gb_sd_h    = 0.5,            # width (sd) of meal-induced emptying pulse [h]

    # ---------- control parameters p_u (dosing regimen) ----------
    dose_rsv_mg = 0,             # rosuvastatin daily dose [mg/d]
    dose_eze_mg = 0,             # ezetimibe daily dose [mg/d]
    start_rsv_d = 0,             # first day of rosuvastatin [d]
    start_eze_d = 0,             # first day of ezetimibe [d]
    dose_hour   = 8,             # clock time of daily intake of both drugs [h]
    dose_sd_h   = 0.5,           # width of oral input pulse (dissolution, gastric emptying) [h]

    # ---------- rosuvastatin PK (illustrative; replace by OSP PBPK if needed) ----------
    F_rsv      = 0.20,           # oral bioavailability [-]
    ka_rsv     = 12,             # absorption rate constant [1/d] (tmax ~3-5 h)
    V_c_rsv    = 100,            # central volume [L]
    V_per_rsv  = 300,            # peripheral volume [L]
    Q_rsv      = 150,            # inter-compartmental clearance [L/d]
    CL_ren_rsv = 330,            # renal clearance [L/d] (~28 % of total)
    CL_upt_rsv = 1500,           # OATP1B1-mediated hepatic uptake clearance at f_OATP = 1 [L/d]
    k_hc_rsv   = 30,             # sinusoidal efflux liver -> blood [1/d]
    k_bil_rsv  = 30,             # biliary excretion (BCRP, MRP2) [1/d]
    V_h        = 1.5,            # liver volume [L]

    # ---------- ezetimibe PK with enterohepatic circulation (EHC) ----------
    # E_* lumps ezetimibe + its active glucuronide [mg]
    ka_eze     = 12,             # uptake gut wall -> portal/systemic [1/d]
    k_fec_eze  = 6,              # faecal loss from gut [1/d]
    k_bile_eze = 6,              # hepatic excretion into gallbladder bile [1/d]
    k_ren_eze  = 0.15,           # renal elimination [1/d]

    # ---------- pharmacodynamics (PD) ----------
    Imax_S       = 0.90,         # max fractional inhibition of hepatic HMGCR [-]
    IC50_S       = NA_real_,     # hepatic rosuvastatin conc. at half Imax [mg/L]; set by calibrate_pd()
    I_S_ref      = 0.70,         # target time-averaged HMGCR inhibition at reference dose [-]
    dose_ref_rsv = 10,           # reference rosuvastatin dose [mg/d]
    Emax_E       = 0.65,         # max fractional inhibition of NPC1L1 absorption [-]
    EC50_E       = NA_real_,     # gut ezetimibe amount at half Emax [mg]; set by calibrate_pd()
    I_E_ref      = 0.60,         # target time-averaged NPC1L1 inhibition at reference dose [-]
    dose_ref_eze = 10            # reference ezetimibe dose [mg/d]
  )
  parms
}

# -----------------------------------------------------------------------------
# derive_parms(): flux targets -> rate constants, then apply patient factors.
# Idempotent: recomputes everything from the base fields, so it can be called
# again after any modifyList() of base fields or patient factors.
# -----------------------------------------------------------------------------
derive_parms <- function(parms) {
  list2env(parms, envir = environment())

  k_lum_tot <- 1 / tau_lum                          # total lumen exit rate [1/d]
  k_abs_ref <- f_abs0 * k_lum_tot                   # absorption rate constant [1/d]
  k_fec     <- (1 - f_abs0) * k_lum_tot             # faecal neutral-sterol loss [1/d]
  k_B_ref   <- B0 / C_L0                            # biliary secretion rate constant [1/d]
  k_BA      <- BA0 / C_L0                           # bile-acid synthesis rate constant [1/d]
  A0        <- f_abs0 * (D_ref + B0)                # absorbed cholesterol [mmol/d]
  S_P       <- (k_B_ref + k_BA) * C_L0 - A0 - S_hep0 # peripheral synthesis, closes balance [mmol/d]
  if (S_P < 0) stop("Calibration infeasible: peripheral synthesis S_P < 0; check flux targets.")

  K_C_ref   <- C_L0 / (1 / g0 - 1)^(1 / n_g)        # SREBP2 sensor half-activity pool [mmol]
  S_max_ref <- S_hep0 / g0                          # maximal hepatic synthesis [mmol/d]
  P_V0      <- FCR0 * LDL0 * V_p                    # LDL production (via VLDL) [mmol/d]
  k_U       <- frac_hepR * FCR0                     # hepatic LDLR clearance at R_LDLR = 1 [1/d]
  k_nR      <- (1 - frac_hepR) * FCR0               # non-hepatic LDL clearance [1/d]
  k_RCT     <- (S_P + k_nR * LDL0 * V_p) / C_P0     # reverse cholesterol transport [1/d]

  derived <- list(
    k_fec = k_fec,                                  # [1/d]
    k_abs = k_abs_ref * k_abs_factor,               # [1/d]
    k_B   = k_B_ref * k_B_factor,                   # [1/d]
    k_BA  = k_BA,                                   # [1/d]
    S_P   = S_P,                                    # [mmol/d]
    K_C   = K_C_ref * K_C_factor,                   # [mmol]
    S_max = S_max_ref * S_max_factor,               # [mmol/d]
    P_V0  = P_V0,                                   # [mmol/d]
    k_U   = k_U,                                    # [1/d]
    k_nR  = k_nR,                                   # [1/d]
    k_RCT = k_RCT                                   # [1/d]
  )
  modifyList(parms, derived)
}

# state order used everywhere (deSolve needs identical order in y and derivatives)
CHOL_STATES <- c("C_lum", "C_L", "R_LDLR", "LDL", "C_P")
#  C_lum  [mmol]   cholesterol in gut lumen (diet + bile)
#  C_L    [mmol]   hepatic regulatory free-cholesterol pool
#  R_LDLR [-]      hepatic LDLR activity relative to reference baseline
#  LDL    [mmol/L] plasma LDL-cholesterol
#  C_P    [mmol]   exchangeable peripheral tissue cholesterol (non-liver organ)
PK_STATES <- c("A_rg", "A_rc", "A_rp", "A_rh", "E_gut", "E_pl", "E_gb")
#  A_rg  [mg] rosuvastatin gut depot      A_rc [mg] central   A_rp [mg] peripheral
#  A_rh  [mg] rosuvastatin in liver
#  E_gut [mg] ezetimibe(+glucuronide) in gut lumen/wall (site of NPC1L1 action)
#  E_pl  [mg] ezetimibe in plasma/liver   E_gb [mg] ezetimibe stored in gallbladder bile
