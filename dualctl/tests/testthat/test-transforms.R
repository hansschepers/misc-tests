test_that("parameter transforms round-trip", {
  theta <- c(0.004, 50, 0.9, -1.2)
  tr <- c("log", "log", "logit", "identity")
  phi <- transform_vec(theta, transform = tr, lo = c(0, 0, 0, 0), hi = c(1, 1, 1, 1))
  back <- inverse_transform_vec(phi, transform = tr, lo = c(0, 0, 0, 0), hi = c(1, 1, 1, 1))
  expect_equal(back, theta, tolerance = 1e-12)
})

test_that("logit transform respects general bounds", {
  phi <- transform_vec(3, transform = "logit", lo = 2, hi = 6)
  expect_equal(phi, 0 + stats::qlogis(0.25))
  expect_equal(inverse_transform_vec(phi, transform = "logit", lo = 2, hi = 6), 3)
})

test_that("phi_to_parms and parms_to_phi are inverse", {
  model <- load_model(toy_cfg(), variant = "fitted")
  phi <- c(k_el = log(0.3), b_in = log(1.7))
  parms <- phi_to_parms(model, phi = phi)
  expect_equal(parms$k_el, 0.3)
  expect_equal(parms$b_in, 1.7)
  expect_equal(parms_to_phi(model, parms = parms), phi)
})

test_that("seeding is local and reproducible", {
  set.seed(99)
  before <- stats::runif(1)
  set.seed(99)
  a <- with_seed(5, stats::rnorm(3))
  after <- stats::runif(1)
  expect_equal(after, before)
  expect_equal(a, with_seed(5, stats::rnorm(3)))
  expect_equal(derive_seed(1, "weather"), derive_seed(1, "weather"))
  expect_false(derive_seed(1, "weather") == derive_seed(2, "weather"))
})
