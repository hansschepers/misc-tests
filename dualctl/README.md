# dualctl — dual control and active learning for model-rich, data-poor processes

`dualctl` is an R toolbox for **model-based sequential optimisation with active
learning (dual control)**. It targets processes where the model structure is
roughly trusted, the parameters are uncertain and observations are sparse and
noisy. One season (or campaign) is one trajectory to learn from and act on. Each
decision epoch the toolbox

1. **advises** which inputs to change, by how much and in which order;
2. **refits** a hierarchical parameter/state estimator after every change and
   recomputes the plan;
3. **probes** deliberately when the information is worth its cost.

The engine is model-agnostic: a model is one plug-in R file plus one YAML
scenario. The heat-exchanger fouling demo (`inst/models/fouling.R`) and a toy
decay tank used by the tests (`inst/models/toy_decay.R`) run through the same
engine code.

> **Status — please read.** This package was written without being executed
> (on request: no test runs). Nothing here has been run: not the tests, the
> vignette or the demo. Expect some first-run bugs; the tests are the
> place to start (`devtools::test()`). No results, runtimes or figures are
> claimed in this README. `run_demo.R` writes `demo_output/summary.md`,
> which states from the numbers where dual control beat, matched or lost to
> the simpler baselines. **All demo numbers come from a synthetic plant with
> illustrative, unvalidated parameter values.**

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Methods in brief](#methods-in-brief)
- [The fouling demo](#the-fouling-demo)
- [Evaluation design](#evaluation-design)
- [How to add a new model](#how-to-add-a-new-model)
- [How to add a probing strategy](#how-to-add-a-probing-strategy)
- [Assumptions](#assumptions)
- [What to expect, honestly](#what-to-expect-honestly)
- [Tests and milestones](#tests-and-milestones)

---

## Installation

R >= 4.1. Imports: `R6`, `data.table`, `deSolve`, `yaml`, `ggplot2` (plus base
`stats`, `utils`). Suggests: `testthat`, `future`, `future.apply` (parallel
Monte Carlo), `nloptr` (optional BOBYQA optimiser), `knitr`, `rmarkdown`.

```r
# from the repository root
install.packages(c("R6", "data.table", "deSolve", "yaml", "ggplot2",
                   "testthat", "future", "future.apply", "knitr", "rmarkdown"))
devtools::document("dualctl")   # generates man/ from the roxygen comments
devtools::install("dualctl")    # parallel workers need the installed package
devtools::test("dualctl")
```

`NAMESPACE` is written by hand to match the roxygen tags; `man/` is not in the
repository yet (run `devtools::document()`). For a pinned environment run
`renv::init()` then `renv::snapshot()` in `dualctl/`; no lockfile is committed
because the package has not been run here.

## Quick start

```r
library(dualctl)
cfg <- load_scenario("fouling_quick.yaml")          # or a path to your own YAML
res <- run_season(cfg, method = "dual_infompc", seed = 1, advisor = TRUE, keep_obs = TRUE)
res$advisor$sequence                                 # recommended changes, in order
plot_param_convergence(res$posterior, unit_idx = 0)  # shared parameters

out <- run_demo("fouling_quick.yaml", seed = 1)      # all figures, tables and summary.md
```

Full study (50 paired seeds, all baselines and ablations):

```sh
cd dualctl && R CMD INSTALL . && Rscript run_demo.R 20261027 8 fouling_default.yaml
```

---

## Architecture

```
          scenario.yaml ──► load_scenario() ──► cfg (engine defaults ⊕ scenario ⊕ overrides)
                                                 │
 model file (build_model) ──► load_model(cfg, "fitted" | "truth") ──► dc_model (ModelSpec)
                                                 │
   ┌──────────── run_season(cfg, method, seed) ──┴───────────────────────────────────────┐
   │  Plant (truth variant, hidden hierarchical parameters, realized weather, CRN noise)  │
   │  Estimator (EKF | EnKF on [x_1..x_N, phi_shared, b_1..b_N])                         │
   │  per unit and arm: ProbingStrategy$act(state, posterior, forecast, time)             │
   │      └─ Controller (receding horizon, move blocks, event-timing enumeration)        │
   │  Advisor (ranked changes, greedy sequence)    Experiment layer (arms, crossover)     │
   └──────────────────────────────────────────────────────────────────────────────────────┘
        │                                       │
  identifiability_report()               run_monte_carlo() ─► summarize_metrics() ─► plots, summary.md
  (sensitivities, FIM, CRB)              run_ablations()       season_attribution() (Shapley)
```

| File | Content |
|---|---|
| `R/utils.R` | seeding (`with_seed`, `derive_seed`), transforms, `softplus`, linear algebra, config loading (`load_scenario`, scenario `base:` inheritance) |
| `R/model_spec.R` | `model_spec()`, `load_model()`, `check_model()`, `phi_to_parms()` |
| `R/disturbances.R` | interpolation, AR(1)/OU/storm generators, CSV input, climatology-plus-persistence forecaster |
| `R/simulate.R` | RK4 on a fixed grid (deSolve or internal), `simulate_unit()` with piecewise-constant controls, events, cost and constraint integration |
| `R/plant.R` | `Plant` (truth model, hierarchical population, common-random-number noise) |
| `R/estimator.R`, `R/estimator_enkf.R` | hierarchical EKF and EnKF with the common interface `init/predict/update/apply_event/posterior/sample/cov` |
| `R/identifiability.R` | FD and forward-equation sensitivities, FIM, spectrum, Cramér–Rao bounds, expected information gain, identifiability report, profile likelihood |
| `R/schedules.R` | calendar schedule, PRBS, multisine, open-loop dither plans |
| `R/controller.R` | `Controller` (receding horizon, rate-limited decision mapping, event enumeration, random shooting + Nelder–Mead / BOBYQA) |
| `R/strategies.R` | `ProbingStrategy` interface, registry and nine strategies |
| `R/advisor.R` | `advise()` — ranked changes by expected saving per unit cost, greedy re-ranked sequence |
| `R/attribution.R` | Shapley (exact / permutation MC, linear or log space), Aumann–Shapley path attribution |
| `R/experiment.R` | arm assignment, `run_season()` |
| `R/evaluation.R` | Monte Carlo, regret, metrics, ablations, `write_summary()` |
| `R/plots.R`, `R/demo.R` | figures and the end-to-end `run_demo()` |
| `inst/config/engine_defaults.yaml` | every engine number (no magic numbers in engine code) |
| `inst/config/fouling_default.yaml`, `fouling_quick.yaml`, `toy.yaml` | scenarios |
| `inst/models/fouling.R`, `toy_decay.R` | model plug-ins |

## Methods in brief

**Model form.** `dx/dt = rhs(time, x, u, d, parms)`, `y = outputs(time, x, u, d, parms)`.
`outputs()` returns one named vector holding the observables, the cost-rate
components [currency/time] and the constraint margins g (violated if g > 0); the
spec says which name plays which role. Controls are piecewise constant per
decision epoch with bounds and per-epoch rate limits; discrete events (cleaning)
reset states and carry a lump cost.

**Parameters** are estimated in a transformed space (log/logit/identity) with
priors from the config. **Hierarchy:** unit i has effective parameters
`phi_i = phi_shared + M b_i`, `b_i ~ N(0, unit_sd²)` for the parameters with
`unit_sd > 0`. All units sit in one augmented state, so what the probe units
teach about `phi_shared` reaches the production units.

**Estimators.** EKF: finite-difference Jacobians of the transition (one RK4
integration per perturbed coordinate), measurement and event maps; Joseph-form
update; optional parameter random walk to absorb structural mismatch. With
`learn: false` the parameter rows of the gain are zero (states tracked,
parameters frozen at the prior). EnKF: perturbed observations, multiplicative
inflation of the parameter anomalies; a derivative-free robustness check.

**Identifiability.** Output sensitivities to phi (central differences or forward
sensitivity equations `dS/dt = f_x S + f_phi`, with event jumps), Fisher
information `F = Σ S' R⁻¹ S`, its eigen-spectrum, classical CRB (∞ on null
directions) and the Bayesian bound `sqrt(diag((F + P0⁻¹)⁻¹))`. The report
compares one season with (a) disturbances only (calendar recipe, routine
sensors), (b) PRBS dither with routine sensors, (c) dither with probe sensors,
and classifies each parameter.

**Expected information gain** of a plan: `IG = ½ log det(I + L F L')` with the
current posterior covariance `Σ = L'L` and `F` predicted along the planned
trajectory for the unit's sensors (nats).

**Controller.** Receding horizon over `horizon_epochs`; continuous controls in
move blocks mapped from [0,1] into the rate-limited box (always feasible);
events by enumeration over timing (none, epoch 0, …, H−1) respecting minimum
intervals; objective = integrated cost + event costs + constraint penalties +
terminal cost (`terminal_days` × final cost rate). Optimiser: random shooting
with common candidates (including "hold"), Nelder–Mead refinement (or BOBYQA
via nloptr), event timing evaluated on the best continuous plan, then
re-optimised.

**Strategies** (`act(state, posterior, forecast, time)`; the task's `t` is
called `time` to avoid masking `t()`):

| name | behaviour |
|---|---|
| `calendar` | fixed recipe, events every N epochs (baseline) |
| `ce` / `none` | certainty-equivalent MPC with the posterior mean, no probing |
| `dither` | CE plus bounded PRBS or multisine; per-control amplitude ∝ information gain per unit perturbation cost (flat-cost, high-sensitivity directions dither most); annealed |
| `infompc` | min over the CE plan and ±step probe candidates of `J − λ_t · IG`, λ_t = λ0 · anneal(season fraction) |
| `thompson` | CE plan with one posterior draw per epoch |
| `evsi` | heuristic EVSI: EVPI from posterior draws × fraction of variance a candidate resolves (`1 − exp(−2 IG/n_p)`) × remaining-season factor |
| `hillclimb` | naive one-at-a-time on measured cost; threshold rule for events (baseline) |
| `oracle` | MPC with the true model, true parameters and true state (regret reference) |

**Advisor.** For each continuous control, steps of ±k·step within the rate limit,
and each event that is allowed now, the expected saving over the horizon against
"hold the current recipe" is computed under posterior draws (common random
numbers), with 5–95 % range and P(saving > 0). Rank = expected saving / change
cost; the recommended step is the best step per control. The sequence is built
greedily: adopt the top change, re-evaluate the rest given it, re-rank (limiting
factor logic). Called every epoch after the refit.

**Attribution.** The adopted season is replayed open-loop on the true plant
(same weather and initial state) with each change as a 0/1 gate against the
calendar baseline. Shapley values by exact enumeration (≤ 8 gates) or
permutation Monte Carlo; `space: log` plays the game on log cost and maps shares
back. Gates per control/event type or per adopted step change. Aumann–Shapley
integrates finite-difference gradients along the straight path from baseline to
adopted continuous trajectories (events fixed at the adopted schedule).

**Experiment layer.** Production arm (champion: CE on the shared posterior,
routine sensors) and probe arm (probing strategy, intensive sensors: an extra
measured quantity and lower noise), 2–3 units per arm, randomised within
cohorts, optional crossover at a season fraction.

## The fouling demo

*Illustrative SI values, not validated against any plant.* A shell-and-tube
cooler: process stream (C_h = 200 kW/K, 65 °C in, 45 °C target) cooled by
cooling water in 300 tubes (19 mm, 6 m). Scale deposits on the water side.

- **Fitted model** (state `Rf` [m²K/kW]): Ebert–Panchal threshold form
  `dRf/dt = α (Re/Re_ref)^-0.66 (Pr/Pr_ref)^-0.33 exp(−E/R (1/T_film − 1/T_ref)) c_f exp(−k_inh·dose) − γ (τ_w/τ_ref) Rf/(Rf+ε)`,
  re-parameterised at reference conditions to decorrelate α and E. Cleaning:
  `Rf → (1 − η) Rf`. Derived: Dittus–Boelter h_i, clean U0 (× `f_U0`), fouled
  U, ε-NTU duty, outlet temperatures, film temperature, Blasius friction,
  tube narrowing → pressure drop, pump power.
- **Truth** adds deposit hardness `H` (ageing at `k_age`): hard deposit resists
  shear removal and cleaning, scaled by `mismatch` ∈ [0, 1]. The fitted model
  cannot represent this.
- **Estimated parameters:** `alpha_dep`, `E_act`, `gamma_rem`, `k_inh`, `f_U0`,
  `eta_clean` (random effects on `alpha_dep` and `f_U0`).
- **Controls:** tube velocity `v` [m/s] (0.8–2.5, ±0.3 per epoch), inhibitor dose
  [mg/L] (0–10, ±2 per epoch), cleaning (event, at least 3 epochs apart).
  The optional approach-temperature setpoint is not implemented.
- **Disturbances:** cooling-water inlet temperature (seasonal cosine + AR(1) +
  storms), foulant index (log-OU drift + storm spikes), process load (weekly
  cycle); real data via `model: disturbances: {source: csv, path: …}`.
- **Observations:** routine (all units, every 2 d): U from temperatures and flows
  (sd 0.03 kW/m²K), pressure drop (sd 0.8 kPa). Intensive (probe units, daily):
  lower noise plus a direct fouling probe `Rf_probe` (sd 0.01 m²K/kW).
- **Costs [EUR/d]:** cooling shortfall (0.02 EUR/kWh of duty below the
  requirement, smooth max), pumping (0.15 EUR/kWh over exchanger + circuit
  losses), inhibitor (6 EUR/kg on 2000 m³/d treated volume); cleaning 4000 EUR
  labour + 1 d downtime valued as lost duty. Constraints: process outlet ≤ 50 °C,
  exchanger Δp ≤ 30 kPa.

## Evaluation design

- **Paired Monte Carlo** (`evaluation: n_seeds: 50`): the seed fixes weather,
  population, initial states, observation noise and arm assignment.
- **Methods** (`fouling_default.yaml`): `calendar`; `ce_prior` (CE with prior
  means, no learning); `learn_noprobe` (CE + learning, routine sensors);
  `learn_sensors` (as before but the probe arm has intensive sensors and no
  perturbation — separates the value of sensors from the value of probing);
  `hillclimb`; `dual_dither`, `dual_infompc`, `dual_thompson` (probe arm with
  that strategy); `dual_evsi` (configured, not in the default run list because it
  is expensive); `oracle`.
- **Metrics:** cumulative regret vs the oracle with t-intervals; total cost;
  paired differences of every dual method against every simpler baseline with a
  verdict; parameter RMSE and posterior log-det over time; 90 % coverage;
  information gained (nats) and extra information per 1000 EUR of probe cost;
  constraint-violation days; probe-arm extra cost vs production-arm saving
  (paired against `learn_sensors`).
- **Ablations:** λ schedule (constant/linear/exponential/zero), probe-arm size
  (1/2/3), sensor noise (×0.5/1/2), mismatch strength (0/0.5/1).
- **Figures:** regret by strategy (time course and season end), parameter
  convergence, parameter RMSE, log-det, coverage, identifiability spectrum and
  bounds, advisor sequence over a season, controls and fouling per unit, Shapley
  attribution, probe-arm value, ablations.

## How to add a new model

No engine change is needed. Two files:

**1. A model file**, e.g. `models/sei.R`, defining `build_model(cfg, variant)`
that returns `model_spec(...)`. The engine sources it into an environment whose
parent is the dualctl namespace, so helpers such as `softplus()`,
`with_seed()`, `ar1_series()` and `storm_series()` are available.

```r
# models/sei.R -- battery SEI growth (sketch)
sei_rhs <- function(time, x, u, d, parms) {
  list2env(c(parms, as.list(x), as.list(u), as.list(d)), envir = environment())
  arr <- exp(-Ea_sei * 1000 / 8.314 * (1 / (T_cell + 273.15) - 1 / 298.15))  # [-]
  dL <- k_sei * arr * (1 + beta_soc * soc_max) / (1 + L_sei / L_0)            # [nm/d] diffusion-limited
  c(dL)
}
sei_outputs <- function(time, x, u, d, parms) {
  list2env(c(parms, as.list(x), as.list(u), as.list(d)), envir = environment())
  cap <- 100 - q_per_nm * L_sei                                                # [%] capacity
  c(cap_obs = cap, R_obs = R_0 + r_per_nm * L_sei,                            # [%], [mOhm]
    cost_wear = c_cap * q_per_nm * k_sei,                                     # [EUR/d]
    cost_lost_use = c_use * (1 - soc_max),                                    # [EUR/d]
    g_cap = cap_min - cap)                                                    # [%]
}
build_model <- function(cfg, variant = "fitted") {
  mc <- cfg$model
  model_spec(name = mc$name, variant = variant, states = mc$states, params = mc$parameters,
             constants = lapply(mc$constants, as.numeric), controls = mc$controls,
             events = list(),
             disturbances = list(names = "T_cell", units = "degC",
                                 generate = function(times, cfg, seed) with_seed(seed, data.table::data.table(
                                   time = times, T_cell = 25 + 8 * sin(2 * pi * times / 365),
                                   T_cell_clim = 25 + 8 * sin(2 * pi * times / 365)))),
             rhs = sei_rhs, outputs = sei_outputs, sensor_sets = mc$sensor_sets,
             cost_outputs = c("cost_wear", "cost_lost_use"), constraint_outputs = "g_cap")
}
```

**2. A scenario YAML** with a `model:` block (states, parameters with
transform/prior/range/truth spread, constants, controls with bounds/rate/
default/step/change_cost, events, disturbances, sensor sets) plus `methods:`
and any engine overrides. Copy `inst/config/toy.yaml` as a template. A file can
extend another with `base: other.yaml`.

Rules the checker enforces (`check_model()`): unique names across states,
parameters, constants, controls and disturbances; `rhs()` returns `n_x` finite
derivatives; `outputs()` is fully named, contains every sensor/cost/constraint
name and does not reuse state or control names; event `apply()` returns `n_x`
states and `cost()` one number. A model needs at least one continuous control.
If the plant should differ structurally, make `build_model()` return extra
states/constants for `variant = "truth"` (see `fouling.R`). Then:

```r
cfg <- load_scenario("path/to/sei.yaml")
res <- run_season(cfg, method = "dual_infompc", seed = 1)
```

`tests/testthat/test-toy-engine.R` proves the point with `toy_decay.R`: every
method, the EnKF, crossover, the advisor and attribution run on it with no
model-specific engine code.

## How to add a probing strategy

```r
MyStrategy <- R6::R6Class("MyStrategy", inherit = ProbingStrategy, public = list(
  name = "mine",                      # reads cfg$strategies$mine
  act = function(state, posterior, forecast, time) {
    pl <- self$ce_plan(state, phi = posterior$mean, forecast = forecast, time = time)
    self$result(pl$u_first, events = pl$events_first)
  }))
register_strategy("mine", MyStrategy)
# then in the YAML: methods: {my_method: {production: ce, probe: mine, learn: true, n_probe: 2}}
```

## Assumptions

Made to work unattended; each is a config entry or a documented simplification.

1. Time unit is the day; decisions every 14 d; controls piecewise constant per epoch.
2. Both RK4 engines use a fixed grid (plant 0.5 d, inner loops 1 d); the internal
   loop exists only to avoid deSolve call overhead in the many short integrations.
3. Disturbances (weather, load) are measured in real time by the estimator; the
   controller plans with a forecast (climatology + decaying current anomaly). The
   oracle gets the same forecasts, so regret isolates model and parameter error.
4. Observation noise is Gaussian, additive, independent, with known sd; the
   estimator knows the sensor noise level (also in the noise ablation).
5. Random-effect sds of the estimator are fixed from the config, not estimated.
6. The EKF's parameter random walk (`param_process_sd`) is the only allowance for
   structural mismatch; coverage below nominal is expected when `mismatch > 0`.
7. Cleaning is instantaneous in the dynamics; its downtime is a lump cost (lost
   required duty for `downtime_d`).
8. Terminal cost = `terminal_days` × cost rate at the horizon end (no explicit
   cost-to-go model).
9. The controller considers at most one event per horizon; continuous controls
   use move blocking [1, 3] epochs.
10. Information value is computed on the unit's effective parameters over a short
    window (`info_window_epochs`); transfer to other units is not valued explicitly.
11. EVSI is a heuristic proxy (EVPI × variance fraction resolved), not a
    preposterior analysis.
12. The hill climber sees the metered operating cost of the last epoch (event
    costs excluded) and the latest measured outputs.
13. Shapley attribution replays open-loop; feedback that would have followed a
    switched-off change is ignored. Replays use the inner integration step.
14. The truth population mean is drawn around the prior (`truth_pop_sd: 1`), i.e.
    the prior is calibrated; set `plant: truth_spread_mult` to stress it.
15. Fouling: water properties from simple fits (0–60 °C); property temperature =
    cooling-water inlet; film temperature from the bulk mean plus 0.55 × q/h_i;
    deposit thickness = Rf × λ_dep; minor and circuit losses as fixed coefficients;
    inhibitor acts multiplicatively on deposition; dosing cost scales with a fixed
    treated volume, not with circulation flow.
16. PRBS is a seeded random ±1 sequence with a minimum hold (not a maximum-length
    LFSR sequence).
17. Parallel runs use `future.apply`; workers need the installed package.

## What to expect, honestly

No results are reported because nothing was run. Some points can be stated in advance
and should be checked against `summary.md`:

- **Learning versus probing.** With seasonal weather, the identifiability
  report is designed to show which parameters weather alone identifies (expected:
  `f_U0` after cleanings, `alpha_dep`, partly `E_act`) and which need velocity or
  dose changes (`gamma_rem`, `k_inh`). If the CE controller already moves `v` and
  `dose` enough for its own reasons, deliberate probing adds little. In that case
  `dual_*` will **not** beat `learn_noprobe` / `learn_sensors` significantly.
  That would be a legitimate finding, not a bug.
- **Sensors versus perturbation.** `learn_sensors` separates the value of the
  probe sensors from the value of perturbing. Expect a sizeable part of any
  probe-arm benefit to come from the direct fouling probe.
- **Mismatch.** With the hardness effect on, the fitted model is wrong in a way
  more data cannot fix. Learning may then over-trust a mis-specified model
  (coverage drops below 0.9), and the gap to the oracle cannot close.
- **Small cohorts.** With 2–3 probe units and one season, the probe-arm cost is
  certain but its value is spread over only 2–3 production units. A negative net
  value is quite possible.
- **Baselines can be strong.** A well-tuned calendar schedule is hard to beat in
  a mild season. The hill climber is deliberately naive: weather confounds its
  comparisons.

`write_summary()` lists every pairing where a dual method was *not*
significantly better than a simpler baseline, with the paired 95 % CI.

## Tests and milestones

The work was built (but, on request, never run) in milestones, one commit each:
(1) model spec, plant, integrators, fouling and toy plug-ins; (2) estimators;
(3) identifiability; (4) controller, strategies, advisor; (5) experiment layer
and attribution; (6) evaluation, figures, demo, docs.

| Test file | Checks |
|---|---|
| `test-ode-analytic.R` | both RK4 engines vs the analytic solution of the toy model; cost integration; events |
| `test-sensitivities.R` | forward sensitivity equations vs central finite differences (toy with an event, fouling); FIM symmetric PSD; probing raises expected information |
| `test-estimator.R` | EKF recovers parameters from noise-free data; EnKF moves towards the truth; hierarchical transfer between units; frozen parameters |
| `test-strategies.R` | every registered strategy conforms to `act()`; controller respects bounds and rate limits and is no worse than "hold"; advisor ranking |
| `test-toy-engine.R` | the toy model runs through every method, EnKF, crossover and attribution; paired seeds; Shapley exact vs permutation; arm randomisation |
| `test-evaluation.R` | Monte Carlo, regret, metrics, summary, scenario inheritance, figures |
| `test-fouling-model.R`, `test-plant.R`, `test-transforms.R` | fouling magnitudes, velocity effect, mismatch-0 equivalence; plant reproducibility; transforms and seeding |

Runtime of the full study has not been measured. A rough estimate from the
operation counts: a few minutes per seed per method on one core, so plan on
several workers for 50 seeds × 9 methods plus ablations. The knobs are
`controller$optimizer`, `solver$dt_inner` and `evaluation$n_seeds`.
