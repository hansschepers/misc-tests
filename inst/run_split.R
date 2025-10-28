# Source the splitter function
source("split_r_file.R")

# Split your monolithic file
result <- split_r_file(
  input_file = "fish_farm_observer_model.R",
  subdir = "functions",
  overwrite = TRUE
)

# Create index
create_index_file(result, subdir = "functions")

# Now load just what you need:
source("functions/fun_create_feeding_table.R")
source("functions/fun_run_fish_farm_simulation.R")

# Or load everything:
source("functions/index_functions.R")
