# Rosuvastatin + ezetimibe in a whole-body cholesterol ODE model

Question: clinicians quote "ezetimibe lowers LDL-C by about 15–20 %". How does that number depend on the state of the system (diet, statin background) and on patient differences, in particular liver parameters?

The R code was **not run**; this is by design (no R in this environment). The numbers in the "Expected results" section come from an independent Python re-implementation of the same steady-state and PK equations. Please run `run_all.R` and compare.

## Files

| File | Content |
|---|---|
| `R/01_parameters.R` | `make_parms()` (all parameters with units), `derive_parms()` (flux targets to rate constants, patient factors), state order |
| `R/02_model.R` | drivers w(t), dosing u(t), PK, coupled ODE `chol_drug_ode()`, `simulate_dynamic()` |
| `R/03_steady_state.R` | semi-analytical steady state, scenario scans, tornado grid, compensation table |
| `R/04_plots.R` | ggplot2 functions only |
| `run_all.R` | sources everything; `run_all()` returns `list(pList, results)` |
| `statin_ezetimibe_model.dot` | structure diagram |

To view the diagram:

```r
DiagrammeR::grViz(readLines("statin_ezetimibe_model.dot"))
```

To run:

```r
setwd("reports/statin_ezetimibe_cholesterol"); source("run_all.R")   # res <- run_all() is called at the end
res$pList$p5          # tornado plot
```

## Model structure

The model has the form dX/dt = f(X, p, u(t), w(t)) and Y = g(X, p, u, w).

Units used throughout: [mmol] for cholesterol, [mg] for drugs, [L] for volumes, [d] for time. LDL-C [mmol/L] × 38.67 = [mg/dL].

| Symbol | Meaning | Units |
|---|---|---|
| C_lum | cholesterol in gut lumen (diet + bile) | mmol |
| C_L | hepatic regulatory free-cholesterol pool | mmol |
| R_LDLR | hepatic LDLR activity relative to reference | – |
| LDL | plasma LDL-C | mmol/L |
| C_P | exchangeable peripheral tissue cholesterol (the non-liver organ) | mmol |
| A_rg, A_rc, A_rp, A_rh | rosuvastatin in gut, plasma, peripheral and liver | mg |
| E_gut, E_pl, E_gb | ezetimibe (+ glucuronide) in gut, plasma/liver and gallbladder | mg |
| u_rsv, u_eze | oral intake rates (Gaussian pulse around `dose_hour`) | mg/d |
| D(t) | dietary cholesterol in meal pulses (w) | mmol/d |
| k_gb(t) | meal-driven gallbladder emptying (w) | 1/d |

Key equations (fluxes in mmol/d):

- SREBP2 activity: g = 1 / (1 + (C_L/K_C)^n_g). This is the negative feedback of hepatic cholesterol.
- Hepatic synthesis: S_hep = S_max · g · (1 − I_S), where I_S is HMGCR inhibition by hepatic rosuvastatin (Emax model).
- Absorption: A = k_abs · (1 − I_E) · C_lum, where I_E is NPC1L1 inhibition by gut ezetimibe (Emax model).
- LDLR activity relaxes to (g/g0)^m_R with time constant tau_R. The exponent m_R lumps the LDLR elasticity net of PCSK9 co-induction.
- LDL production: P_V = P_V0 · (C_L/C_L0)^q_V. Clearance: U_hep = k_U · R_LDLR · LDL · V_p (hepatic) and U_per = k_nR · LDL · V_p (peripheral).
- Peripheral tissue: dC_P/dt = S_P + U_per − k_RCT · C_P.
- Biliary cholesterol B = k_B · C_L and bile acids BA = k_BA · C_L.

Feedbacks and time scales:

- Lumen: hours.
- C_L: about 1.5 d.
- LDLR: tau_R = 0.75 d.
- LDL: 1/FCR ≈ 3 d.
- C_P: 1/k_RCT ≈ 40 d.
- Drug PK: hours, with enterohepatic recirculation of ezetimibe tied to meals.

Calibration: `derive_parms()` turns literature-typical baseline fluxes into rate constants, so the reference patient is exactly at steady state. Reference fluxes: diet 300 mg/d, bile cholesterol 1000 mg/d, absorption 44 %, bile acids 460 mg/d, hepatic synthesis 550 mg/d, LDL 3 mmol/L, FCR 0.35/d. `calibrate_pd()` then sets IC50/EC50 so that the time-averaged inhibitions at the reference doses (10 mg) are I_S = 0.70 and I_E = 0.60, which corresponds to absorption falling from 44 % to about 24 %.

## The key piece of theory (and a correction to my earlier chat answer)

At steady state every LDL particle produced is cleared, and what the periphery takes up comes back by reverse cholesterol transport (RCT). The LDL and RCT routes therefore **cancel in the hepatic mass balance**, and the whole system collapses to one scalar equation in C_L:

f_abs · (D + k_B·C_L) + S_max·(1 − I_S)·g(C_L) + S_P = (k_B + k_BA)·C_L

LDL is then a **read-out** of C_L, with elasticity

ε_LDL = d ln LDL / d ln C_L = q_V + φ_R · m_R · n_g · (1 − g),

where φ_R is the share of LDL cleared by LDLR.

Linearising, the ezetimibe effect is approximately

ΔLDL/LDL ≈ −ε_LDL · Δf_abs·(D + B) / [S_hep·n_g·(1 − g) + (k_B(1 − f_abs) + k_BA)·C_L]

- **Numerator:** the absorbed cholesterol that ezetimibe removes. It scales with diet plus bile, not with diet alone.
- **Denominator:** the buffers. One is extra hepatic synthesis (the "synthesizer" loop). The other is lower biliary and bile-acid output.
- **ε_LDL:** how strongly the hepatic pool is translated into LDLR activity and LDL production.

In my previous chat answer I described LDLR uptake as a parallel channel that compensates the lost absorption. Within this model structure that is not correct at steady state: LDLR uptake is a read-out, not a net mass-balance channel. Only synthesis and biliary/bile-acid output compensate. The function `compensation_table()` and plot `p6` show the balance ΔA + ΔS_hep = ΔB + ΔBA explicitly.

## Expected results

These come from a Python re-implementation of the steady state and PK. They are model predictions with illustrative parameters, not clinical estimates.

Reference patient (10 mg doses, diet 300 mg/d):

| Treatment | LDL-C change |
|---|---|
| Rosuvastatin 10 mg | −47 % |
| Ezetimibe alone | −20 % (hepatic synthesis +29 %) |
| Ezetimibe added to rosuvastatin | −25 % on top |
| Combination vs none | −60 % |

Each doubling of the rosuvastatin dose adds about 5–8 percentage points (5 / 10 / 20 / 40 mg give −38 / −47 / −54 / −59 %).

Ezetimibe alone versus diet and HMGCR capacity:

| S_max factor \ diet [mg/d] | 0 | 300 | 900 |
|---|---|---|---|
| 0.5 | −16 | −23 | −37 |
| 1 | −15 | −20 | −31 |
| 2 | −13 | −17 | −25 |

Ezetimibe alone, one patient parameter changed at a time:

| Parameter | Low | High |
|---|---|---|
| NPC1L1 absorption capacity ×0.4 / ×2 | −11 | −27 |
| LDLR elasticity m_R 0.5 / 1.5 (PCSK9) | −13 | −28 |
| OATP1B1 activity ×0.5 / ×1.5 (add-on effect) | −24.7 | −24.4 |

What these numbers show:

1. **Diet matters more than a pure "diet share of lumen" argument suggests.** Going from 0 to 900 mg/d nearly doubles the effect: the absorbed flux grows, and the higher hepatic pool flattens the SREBP2 response.
2. **Statin background makes ezetimibe stronger** (−20 % alone, −25 % as add-on), because I_S shrinks the synthesis buffer.
3. **The liver parameters with the largest leverage** are the LDLR elasticity (m_R, net of PCSK9) and the HMGCR capacity. A low-synthesizer liver responds more. The SREBP2 set point and ABCG5/G8 activity act through the same loop gain.
4. **OATP1B1 (SLCO1B1) barely changes efficacy** in this model. It mainly changes plasma rosuvastatin exposure, which is relevant for myopathy, not for LDL lowering.
5. **Ezetimibe acts in pulses.** It is recirculated through the gallbladder and released at meals, so I_E is high around meals (plot p9). Meal timing relative to the dose therefore modulates the daily-averaged inhibition.

## Assumptions and limitations

- One lumped hepatic regulatory pool. Real SREBP2 senses ER cholesterol, a small, fast pool. This matters for why statins work so well when hepatic synthesis is only part of whole-body synthesis.
- Peripheral synthesis S_P is held constant (not regulated). Ezetimibe-induced synthesis in the intestine is not modelled.
- Biliary cholesterol is secreted continuously (not stored in the gallbladder); only ezetimibe follows the gallbladder.
- VLDL → LDL conversion is instantaneous. There are no IDL/HDL pools and no explicit PCSK9 state (it is lumped into m_R).
- Steady-state scans use the time-averaged inhibition over a dosing interval. This is justified because cholesterol pools respond much more slowly than PK fluctuates, but it is an approximation. `simulate_dynamic()` keeps the full within-day PK.
- PK parameters are illustrative. For quantitative rosuvastatin work, couple the Open Systems Pharmacology PBPK model via `ospsuite`.

## How published models formulate this differently

| Model | Regulation | Consequence for the ezetimibe question |
|---|---|---|
| van de Pas et al. 2012 (J Lipid Res) | mostly linear first-order fluxes | no compensation, so it overestimates ezetimibe effects |
| Mc Auley et al. 2012 (BMC Syst Biol) | Michaelis–Menten, feedback on synthesis, ageing | has compensation, but LDLR regulation is simpler |
| Gadkar et al. 2014 (CPT:PSP) | SREBP2 co-regulates HMGCR, LDLR and PCSK9 | closest to this model; m_R here lumps their PCSK9 arm |
| Faltaos et al. 2006 | empirical indirect response | cannot answer state-dependence questions |
| Ezzet et al. 2001 | ezetimibe population PK with EHC, gallbladder emptying at fixed times | the EHC logic used here (here with smooth meal pulses) |

## References (PubMed searches; I did not check these against the literature for this file)

- Mc Auley et al. 2012, whole-body cholesterol model: https://pubmed.ncbi.nlm.nih.gov/?term=Mc+Auley+whole-body+mathematical+model+cholesterol+metabolism
- van de Pas et al. 2012: https://pubmed.ncbi.nlm.nih.gov/?term=van+de+Pas+physiologically+based+in+silico+kinetic+model+plasma+cholesterol
- Gadkar et al. 2014: https://pubmed.ncbi.nlm.nih.gov/?term=Gadkar+mechanistic+systems+pharmacology+model+LDL+PCSK9
- Paalvast et al. 2015, review comparing cholesterol models: https://pubmed.ncbi.nlm.nih.gov/?term=Paalvast+evaluating+computational+models+cholesterol+metabolism
- Ezzet et al. 2001, ezetimibe EHC PK: https://pubmed.ncbi.nlm.nih.gov/?term=Ezzet+ezetimibe+population+pharmacokinetic+enterohepatic
- Faltaos et al. 2006, statin indirect response: https://pubmed.ncbi.nlm.nih.gov/?term=Faltaos+indirect+effect+model+LDL+statins
- Open Systems Pharmacology (rosuvastatin PBPK): https://github.com/Open-Systems-Pharmacology

## Glossary

| Abbreviation | Meaning |
|---|---|
| NPC1L1 | Niemann-Pick C1-Like 1, intestinal sterol transporter, the ezetimibe target |
| HMGCR | HMG-CoA reductase, rate-limiting enzyme of synthesis, the statin target |
| SREBP2 | sterol regulatory element-binding protein 2, ER sterol sensor and transcription factor |
| LDLR | LDL receptor |
| PCSK9 | proprotein convertase subtilisin/kexin 9, degrades LDLR and is co-induced by SREBP2 |
| OATP1B1 (SLCO1B1) | hepatic uptake transporter of rosuvastatin |
| BCRP (ABCG2) | efflux transporter (gut absorption, biliary excretion) |
| ABCG5/G8 | sterol efflux into bile and gut lumen |
| CYP7A1 | rate-limiting enzyme of bile-acid synthesis |
| EHC | enterohepatic circulation |
| RCT | reverse cholesterol transport (periphery to liver via HDL) |
| FCR | fractional catabolic rate [1/d] |
| PBPK | physiologically based pharmacokinetics |
| QSP | quantitative systems pharmacology |
| PD | pharmacodynamics |
