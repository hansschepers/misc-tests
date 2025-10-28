# ==============================================================================
# R CODING PREFERENCES - Mathematical Biology / ODE Modeling Projects
# ==============================================================================
# Author background: PhD Mathematical Biology, ODE specialist, 25 years data science
# Primary domain: Life sciences, animal physiology
# 
# Place this header at the top of scripts to guide Claude Code with your preferences
# ==============================================================================

# === CORE R STYLE ===
# - Use `<-` for assignment, NOT `=`
# - NEVER use `<<-` unless absolutely necessary (explain if used)
# - Don't use `T` or `t` as object names (T = TRUE, t = transpose)
# - Write out Greek symbols: use `alpha_0` not `α`, `mu` not `μ`
# - For micro units: use `u` prefix not `\mu` (e.g., `ug` not `μg`)

# === PACKAGE PREFERENCES ===
# - Prefer data.table over tidyverse for data manipulation
# - ggplot2 is excellent for plotting (use `linewidth` not deprecated `size` for lines)
# - For ODEs: use deSolve package with explicit method = "rk4"

# === ODE MODEL STRUCTURE ===
# Typical model form:
#   dX/dt = f(X, p, u(t), w(t))  - Main dynamics
#   Y(t) = g(X, p, u(t), w(t))   - Outputs
# where:
#   X = state variables
#   p = parameters
#   u(t) = control inputs (with cost function)
#   w(t) = uncontrollable drivers (weather, time-varying parameters)
#
# Control u(t) can be:
#   - Rule-based: u(t) = h_r(X, p_u, w(t))
#   - Observer-based with own states:
#       dX_u/dt = f_u(X_u, X, Y, p_ux, w(t))
#       u(t) = h_d(X_u, X, Y, p_u, w(t))

# === UNITS & DIMENSIONS ===
# ALWAYS add units as comments for ALL variables:
#   state <- 10.5        # [kg/m³] - density
#   rate <- 0.05         # [1/day] - specific growth rate
#   flow <- 3.2          # [L/min] - volumetric flow rate
#   biomass <- 450       # [g] - total biomass
#   temperature <- 15.3  # [°C] - water temperature

# === PARAMETERS ===
# - Use `parms` as parameter list name (not `par` or `pars`)
# - Inside ODE functions, unpack with:
#     list2env(c(parms, as.list(state)), envir = environment())
#   NOT parms$ParameterName (gives NULL as silent error)
# - For deSolve::ode(), ensure initial state names match state names in model
# - Add initial states to the parameter list with units in comments

# === DATA.TABLE SPECIFIC ===
# - Avoid matrices when possible; use data.tables with proper column names
# - Time column should be named `time`, not `time_d` etc.
# - To assign a variable `field` to column `field`, use:
#     dt[, field := ..field]
#   NOT dt[, field := field] (meaningless self-reference)

# === FUNCTION CALLING CONVENTIONS ===
# IMPORTANT naming rules:
# 1. If object name EXACTLY matches parameter name: naming optional
#      f(data, method = "rk4")  # if object is named 'data'
# 2. If object name differs: argument MUST be named
#      f(data = my_data, method = "rk4")  # object is 'my_data'
# 3. Once ANY argument is named, name all subsequent arguments
# 4. Exception: trailing arguments with defaults can remain unnamed
#
# Example function: create_matrix(sensitivity_dt, normalization = "parameter_relative")
#   CORRECT:   create_matrix(sensitivity_dt = my_results, normalization = "parameter_relative")
#   INCORRECT: create_matrix(my_results, normalization = "parameter_relative")

# === FOR LOOPS ===
# Always initialize loop variable for debugging BEFORE the loop:
#   pattern <- "three_meals"  # <-- debugging line
#   for (pattern in c("three_meals", "one_meal")) {
#     factor_idx <- 1  # <-- debugging line
#     for (factor_idx in 1:length(factors)) {
#       # loop body
#     }
#   }

# === deSolve SPECIFIC ===
# - Always use method = "rk4" explicitly
# - Ensure initial state vector `y` has:
#   a) Same names as states in model
#   b) Same order as derivatives returned in model
#   c) Unnamed values (use unname() if needed to avoid name corruption)
# - Example:
#     initial_states <- c(N = unname(parms$N_0), 
#                        W = unname(parms$W_0))

# === PLOTTING WITH GGPLOT2 ===
# - For multiple plots, create list: pList <- list()
# - Add plots to list: pList$p1 <- p1, pList$p2 <- p2, etc.
# - Initialize pList at start of run script
# - Return pList in results for easy access
# - Separate simulation code from visualization code into different functions

# === SHINY APPS ===
# Build smart, not complex:
# 1. Extract computational logic into standalone testable functions:
#      my_computation <- function(input_params) { ... }
# 2. Wrap in thin reactive layers:
#      my_reactive <- reactive({ my_computation(input) })
# 3. Start with static plots (ggplot OR plotly, not both)
# 4. Add manual controls first (sliders/buttons)
# 5. NEVER attempt animations until manual control works
# 6. When broken, SIMPLIFY - strip back to last working version
# 7. Use minimal data for prototyping (10-20 frames, 50 points max)
# 8. Test each feature independently

# === EXCEL OUTPUT (openxlsx) ===
# - For borders: use c("top", "bottom", "left", "right") NOT c("all") (deprecated)
# - addStyle() doesn't take ranges for both rows and cols:
#     WRONG: addStyle(wb, sheet, style, rows = 2:3, cols = 1:5)
#     RIGHT: lapply(2:3, \(rr) addStyle(wb, sheet, style, rows = rr, cols = 1:5))
# - Don't overwrite headers when writing data.tables

# === COMMON PITFALLS TO AVOID ===
# - Greek symbols in code (hard to type/send)
# - Using `t` as object name (transpose function)
# - Using `T` as object name (TRUE constant)
# - Not checking object names are passed as arguments (unless explicitly global)
# - Matrix objects without column names (prefer data.table)
# - Missing units in comments
# - Using parms$parameter inside ODE (use list2env instead)

# === DOCUMENTATION REQUIREMENTS ===
# When discussing life sciences be rigorous:
# - Mention feedbacks, time scales, units, dimensions
# - Provide references when possible
# - Cross-check logic between different sources
# - Highlight differences if articles formulate models differently
# - Use real-world examples from life sciences/animal physiology

# === EXAMPLE STRUCTURE ===
# Use this structure for ODE models:
#
# model_function <- function(time_d, state, parms) {
#   # Unpack
#   list2env(c(parms, as.list(state)), envir = environment())
#   
#   # Forcing functions
#   temperature <- temp_forcing(time_d, parms)  # [°C]
#   
#   # Rates
#   growth_rate <- mu_max * (temperature - T_min) / (T_opt - T_min)  # [1/day]
#   
#   # State derivatives
#   dX_dt <- growth_rate * X  # [kg/day]
#   
#   # Return
#   list(c(dX_dt = dX_dt))
# }
#
# # Parameters with units
# parms <- list(
#   mu_max = 0.5,      # [1/day] - maximum specific growth rate
#   T_min = 4,         # [°C] - minimum temperature
#   T_opt = 18,        # [°C] - optimal temperature
#   X_0 = 10           # [kg] - initial biomass
# )
#
# # Initial states (unnamed!)
# initial_states <- c(X = unname(parms$X_0))  # [kg]
#
# # Simulate
# times <- seq(0, 100, by = 1)  # [days]
# out <- ode(y = initial_states, times = times, func = model_function, 
#           parms = parms, method = "rk4")

# ==============================================================================
# END OF PREFERENCES HEADER
# ==============================================================================
