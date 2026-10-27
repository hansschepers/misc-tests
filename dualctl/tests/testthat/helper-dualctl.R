# shared helpers for the dualctl tests

toy_cfg <- function(overrides = list()) load_scenario("toy.yaml", overrides = overrides)

fouling_cfg <- function(overrides = list()) load_scenario("fouling_default.yaml", overrides = overrides)

# disturbance function returning constant values
const_dfun <- function(...) {
  v <- c(...)
  function(time) v
}
