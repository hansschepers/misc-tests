# The second model (toy_decay) is a plug-in file plus a config only. These
# tests run it through every part of the engine without any engine change.

test_that("the toy model runs through every method of the engine unchanged", {
  cfg <- toy_cfg()
  models <- list(fit = load_model(cfg, variant = "fitted"), truth = load_model(cfg, variant = "truth"))
  K <- floor(cfg$season$length_days / cfg$season$epoch_days)
  m <- names(cfg$methods)[1]                                 # debugging line
  for (m in names(cfg$methods)) {
    res <- run_season(cfg, method = m, seed = 1, models = models, advisor = (m == "dual_infompc"),
                      keep_obs = TRUE)
    expect_equal(nrow(res$epochs), K * cfg$experiment$n_units, info = m)
    expect_true(all(is.finite(res$epochs$cost_total)), info = m)
    expect_true(all(c("u_u_in", "comp_cost_track", "x_conc") %in% names(res$epochs)), info = m)
    if (isTRUE(cfg$methods[[m]]$learn) || m == "ce_prior") {
      expect_true(data.table::is.data.table(res$posterior), info = m)
      expect_equal(max(res$posterior$epoch), K, info = m)
    }
    if (m == "dual_infompc") {
      expect_true(nrow(res$advisor$ranking) > 0)
      expect_equal(sum(res$arms == "probe"), cfg$methods[[m]]$n_probe)
    }
  }
})

test_that("weather and plant are paired across methods", {
  cfg <- toy_cfg()
  a <- run_season(cfg, method = "calendar", seed = 5)
  b <- run_season(cfg, method = "oracle", seed = 5)
  expect_equal(a$dist, b$dist)
  expect_equal(a$plant$unit_phi, b$plant$unit_phi)
})

test_that("the ensemble filter can replace the EKF in a season", {
  cfg <- toy_cfg(list(estimator = list(type = "enkf", n_ensemble = 15)))
  res <- run_season(cfg, method = "learn_sensors", seed = 2)
  expect_true(all(is.finite(res$posterior$mean)))
})

test_that("crossover swaps the arms half way", {
  cfg <- toy_cfg(list(experiment = list(crossover = TRUE)))
  res <- run_season(cfg, method = "learn_sensors", seed = 3)
  first <- res$epochs[res$epochs$epoch == 0, ]
  last <- res$epochs[res$epochs$epoch == max(res$epochs$epoch), ]
  expect_equal(last$arm, ifelse(first$arm == "probe", "production", "probe"))
})

test_that("Shapley values of an additive game equal the contributions", {
  contrib <- c(a = 3, b = -1, c = 2)
  vf <- function(S) sum(contrib[S])
  ex <- shapley_values(vf, players = names(contrib), exact_max_players = 8)
  mc <- shapley_values(vf, players = names(contrib), n_perm = 30, exact_max_players = 0)
  expect_equal(ex$phi, unname(contrib))
  expect_equal(mc$phi, unname(contrib))
})

test_that("Shapley values are efficient for an interacting game", {
  vf <- function(S) if (all(S)) 10 else if (S[1]) 4 else 0
  sh <- shapley_values(vf, players = c("x", "y"))
  expect_equal(sum(sh$phi), 10)
  expect_equal(sh$phi, c(7, 3))
})

test_that("season attribution runs on the toy model", {
  cfg <- toy_cfg()
  res <- run_season(cfg, method = "learn_noprobe", seed = 1)
  at <- season_attribution(res, unit_idx = 1, cfg = cfg)
  expect_equal(sum(at$shapley$saving), at$total_saving, tolerance = 1e-8)
  at_log <- season_attribution(res, unit_idx = 1, cfg = cfg, space = "log")
  expect_equal(sum(at_log$shapley$saving), at_log$total_saving, tolerance = 1e-8)
  at_ch <- season_attribution(res, unit_idx = 1, cfg = cfg, gate_by = "change")
  expect_true(nrow(at_ch$shapley) >= 1)
  expect_equal(nrow(at$aumann), 1)
})

test_that("arm assignment is randomised within cohorts", {
  a <- assign_arms(6, n_probe = 2, seed = 1, cohorts = c(1, 1, 1, 2, 2, 2))
  expect_equal(sum(a == "probe"), 2)
  expect_equal(sum(a[1:3] == "probe"), 1)
  expect_equal(assign_arms(5, n_probe = 0, seed = 1), rep("production", 5))
})
