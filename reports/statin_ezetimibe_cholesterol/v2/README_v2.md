# v2: meal pattern, meal composition and pill timing

This extends the v1 model (`../R`, `../README.md`) with bile-acid enterohepatic circulation, micelle-gated cholesterol absorption, meal composition and NPC1L1 binding kinetics. Its purpose is to address these questions:

- Does the liver stop secreting cholesterol while fasting?
- Does ezetimibe need meals?
- Do meal pattern (intermittent fasting), meal composition and pill timing matter?

The R code has **not been run** (by design). No numbers are given below, only the expected direction of effects with their mechanism, so that you can check them against the simulation.

## Files

| File | Content |
|---|---|
| `R/01_parameters_v2.R` | `make_parms_v2()`, `derive_parms_v2()` (mean-field calibration), state order |
| `R/02_model_v2.R` | meal and dosing pulses, gut + PK sub-model, full ODE `chol_v2_ode()`, simulation, PD calibration |
| `R/03_scenarios_v2.R` | meal schedules, scenario table (experiments A–D), runs and summaries |
| `R/04_plots_v2.R` | ggplot2 functions only |
| `run_all_v2.R` | `run_all_v2()` returns `list(pList, results)` |
| `statin_ezetimibe_v2.dot` | structure diagram |

To view the diagram:

```r
DiagrammeR::grViz(readLines("statin_ezetimibe_v2.dot"))
```

To run:

```r
setwd("reports/statin_ezetimibe_cholesterol/v2"); source("run_all_v2.R")
```

The run takes several minutes: about 35 runs of 40 simulated days each.

## What is new compared with v1

The model has 22 states. Units: [mmol] for cholesterol and bile acids (BA); [g] for fat and fibre; [mg] for drugs; [d] for time.

| Process | v1 | v2 |
|---|---|---|
| Biliary cholesterol | B = k_B·C_L, continuous, straight to the lumen | B = B0·(C_L/C_L0)·(J_sec/J_BA0)^nu_B, coupled to BA secretion J_sec. With nu_B < 1 the cholesterol/BA ratio of bile rises when BA flux is low (fasting) |
| Bile routing | none | fasting: a fraction phi_GB = phi_fast·(1 − a_CCK) of hepatic bile (BA, cholesterol and ezetimibe) is stored in the gallbladder. Emptying k_gb = k_gb_mmc + k_gb_max·a_CCK |
| CCK | none | driven by fat in the small intestine: dCCK/dt = (Fat_J/Fat_ref − CCK)/tau_cck |
| Bile acids | none | gallbladder, jejunum and ileum pools; ASBT reabsorption 97 % per pass; the hepatocyte is quasi-steady, so secretion = portal return + synthesis |
| CYP7A1 | k_BA·C_L | k_syn·C_L·h(Z), where Z is the FXR/FGF19 signal following portal BA return (time constant 3 h) |
| Absorption | k_abs·(1 − I_E)·C_lum | k_abs·(1 − I_E)·C_mic, with C_mic = C_lum·cap/(C_lum + a_ps·PS + cap) (smooth gate) and capacity cap = r_mic·BA_J·(m0 + (1 − m0)·Fat/(K + Fat)) |
| Fibre | none | binds a fraction b_fib of ileal BA, which is lost to faeces instead of reabsorbed |
| Plant sterols | none | compete for micellar space (relative affinity a_ps = 1.5) |
| Ezetimibe action | instantaneous Emax on gut amount | NPC1L1 occupancy state: dO/dt = k_on·E_gut·(1 − O) − k_off·O, with I_E = Emax·O. The "instant equilibrium" switch is the k_off → ∞ limit |
| Hepatic synthesis | constant capacity | optional circadian factor 1 + a_circ·cos(2π(h − h_peak)/24) |
| Peripheral pool | slow state | for the scans, RCT is set to S_P + U_per (quasi-steady). This gives the same steady state while removing the 40-day mode |

**Calibration.** `derive_parms_v2()` uses the same daily flux targets as v1 and adds: BA secretion 40 mmol/d (= synthesis/(1 − 0.97)), micellar solubilisation 60 % of lumen cholesterol, and 80 g/d fat. The targets hold only for constant inputs. With pulsed meals and nonlinear gates (Jensen's inequality), the reference baseline lands near but not on them. `res$results$burn_in_summary` shows the gap. The PD calibration sets IC50/EC50/k_on so that the reference regimen (3 meals, pill at 08:00) gives I_S = 0.70 and I_E = 0.60.

## Experiments and expected outcomes (hypotheses, not results)

**A. Meal pattern at equal daily intake** (3 meals; 16:8 time-restricted eating; one meal a day) × (none, ezetimibe, rosuvastatin, both).

- Fewer meals mean fewer gallbladder emptyings, so fewer BA cycles per day, lower daily BA secretion, and lower biliary cholesterol secretion. Less cholesterol therefore enters the gut, and the absolute effect of ezetimibe should be somewhat smaller.
- Long fasts mean less portal BA return, a lower FXR/FGF19 signal and de-repressed CYP7A1. More cholesterol is converted to bile acids, so drug-free LDL should be slightly lower.
- At a single large meal, micellar capacity may saturate, lowering fractional absorption.
- **Expected net effect:** small (a few percent) changes of LDL and of the drug effects. Plot p7 should show that biliary cholesterol secretion continues overnight and that gallbladder bile becomes more lithogenic during long fasts.

**B. Pill time (08, 14, 22 h) × NPC1L1 binding (kinetic k_off = 1/d versus instant equilibrium) × pattern.**

- With slow binding kinetics, pill timing should barely matter, because occupancy averages over the day.
- With instant equilibrium, the effect follows the ezetimibe pulses. Even then the gallbladder releases stored drug together with the next meal's bile, so the timing penalty is damped.
- The difference between the two binding modes is the test of the "drug and substrate arrive together" hypothesis.

**C. Where cholesterol and fat are eaten:**

- spread over meals;
- all cholesterol at a fatty breakfast;
- all cholesterol at a low-fat breakfast;
- one rich dinner.

Cholesterol eaten with little fat triggers little CCK and forms few micelles, so it is poorly absorbed: drug-free LDL is lower and the ezetimibe effect is smaller.

**D. Viscous fibre 10 g/d and plant sterols 2 g/d, with and without ezetimibe.**

- Fibre interrupts BA reabsorption, so CYP7A1 consumes more hepatic cholesterol. This is roughly additive with ezetimibe, because it acts on a different step.
- Plant sterols act on the same step as ezetimibe (micellar uptake), so they should be sub-additive with it.

## Limitations

- Bile acids are a single species (no glyco/tauro, primary/secondary distinction; no colonic bacterial conversion). Phospholipids are implicit in r_mic.
- The hepatocyte BA pool is quasi-steady, and portal extraction is 100 % on first pass.
- The CCK stimulus is fat only. Protein (also a strong stimulus) can be entered as fat-equivalent grams.
- Small-intestinal transit is constant, independent of meals.
- Trans-intestinal cholesterol excretion (TICE) and the intestinal cholesterol synthesis induced by ezetimibe are not included.
- The NPC1L1 off-rate (k_off = 1/d) is an assumption, not a literature value. It is the key parameter of experiment B.
- No gallstone nucleation. The cholesterol/BA ratio in the gallbladder is only a proxy for the cholesterol saturation index (CSI); phospholipids are ignored.

## Glossary

| Term | Meaning |
|---|---|
| ASBT | apical sodium-dependent bile acid transporter (ileal BA reabsorption) |
| CCK | cholecystokinin (gallbladder contraction, sphincter of Oddi relaxation) |
| CYP7A1 | cholesterol 7α-hydroxylase, the rate-limiting enzyme of BA synthesis |
| FXR / FGF19 | bile-acid receptor / ileal hormone that represses CYP7A1 |
| MMC | migrating motor complex (fasting gut motility, partial gallbladder emptying) |
| TRE / OMAD | time-restricted eating / one meal a day |
| CSI | cholesterol saturation index of bile |
| TICE | trans-intestinal cholesterol excretion |
