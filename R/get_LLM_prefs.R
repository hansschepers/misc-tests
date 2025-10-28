#' get_LLM_prefs.R
#' @export
get_LLM_prefs <- function(wdout = "."
                            , overwrite = TRUE) {
  input_file <- system.file("R_PREFERENCES_HEADER.R", "misc_tests")
  output_file <- file.path(wdout, basename(input_file))
  if (!file.exists(input_file)) {
    stop(sprintf("Input file does not exist: %s", input_file))
  }
  
  file.copy(input_file, output_file, overwrite = overwrite)
}
