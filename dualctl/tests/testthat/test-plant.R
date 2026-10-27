test_that("plant is reproducible from its seed", {
  cfg <- toy_cfg()
  tru <- load_model(cfg, variant = "truth")
  p1 <- Plant$new(tru, cfg = cfg, seed = 3, n_units = 3)
  p2 <- Plant$new(tru, cfg = cfg, seed = 3, n_units = 3)
  p3 <- Plant$new(tru, cfg = cfg, seed = 4, n_units = 3)
  expect_equal(p1$dist, p2$dist)
  expect_equal(p1$unit_phi, p2$unit_phi)
  expect_false(isTRUE(all.equal(p1$unit_phi, p3$unit_phi)))
  y_true <- c(conc_obs = 3)
  expect_equal(p1$observe(1, time = 2, sensor_set = "routine", y_true = y_true),
               p2$observe(1, time = 2, sensor_set = "routine", y_true = y_true))
})

test_that("plant advances and records cost", {
  cfg <- toy_cfg()
  tru <- load_model(cfg, variant = "truth")
  pl <- Plant$new(tru, cfg = cfg, seed = 1, n_units = 2)
  res <- pl$advance(1, t0 = 0, t1 = 5, u = c(u_in = 1), out_times = 1:5)
  expect_equal(nrow(res$y_out), 5)
  expect_true(is.finite(res$cost))
  expect_equal(pl$x[[1]], res$x_end)
})
