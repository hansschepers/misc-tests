# ==============================================================================
# FUNCTION: split_r_file
# ==============================================================================
# Split a monolithic R file into separate files, one per function
# Functions saved as "fun_<function_name>.R"
# Demo/run blocks saved as "run_<block_name>.R"
# ==============================================================================

#TODO file name per function should be the name of the funciton with 'fun_' before it, nothing behind (now it contains all characters up till the `{`)

#' split_r_file
#' @export
split_r_file <- function(input_file, subdir = ".", overwrite = TRUE, verbose = TRUE) {
  # Split an R file into individual function files and demo scripts
  #
  # Args:
  #   input_file: path to input R file [character]
  #   subdir: subdirectory for output files (created if needed) [character]
  #   overwrite: whether to overwrite existing files [logical]
  #   verbose: print progress messages [logical]
  #
  # Returns:
  #   list with components:
  #     functions: character vector of function file paths created
  #     demos: character vector of demo file paths created
  #     skipped: character vector of files skipped (already exist, overwrite=FALSE)
  
  if (!file.exists(input_file)) {
    stop(sprintf("Input file does not exist: %s", input_file))
  }
  
  # Create subdirectory if needed
  if (subdir != ".") {
    if (!dir.exists(subdir)) {
      dir.create(subdir, recursive = TRUE)
      if (verbose) cat(sprintf("Created directory: %s\n", subdir))
    }
  }
  
  # Read file
  lines <- readLines(input_file, warn = FALSE)
  n_lines <- length(lines)
  
  if (verbose) {
    cat(sprintf("Read %d lines from %s\n", n_lines, input_file))
  }
  
  # Initialize tracking
  functions_created <- character(0)
  demos_created <- character(0)
  skipped_files <- character(0)
  
  # State machine variables
  in_function <- FALSE
  in_demo <- FALSE
  current_name <- ""
  current_lines <- character(0)
  brace_depth <- 0
  comment_block <- character(0)  # Preceding comment lines
  
  i <- 1  # Line counter
  while (i <= n_lines) {
    line <- lines[i]
    trimmed <- trimws(line)
    
    # ========================================================================
    # DETECT FUNCTION DEFINITIONS
    # ========================================================================
    # Pattern: name <- function(...) OR name = function(...)
    func_pattern <- "^([a-zA-Z_][a-zA-Z0-9_\\.]*)[[:space:]]*(<-|=)[[:space:]]*function[[:space:]]*\\("
    
    if (!in_function && !in_demo && grepl(func_pattern, trimmed)) {
      # Extract function name
      func_name <- sub(func_pattern, "\\1", trimmed)
      
      in_function <- TRUE
      current_name <- func_name
      current_lines <- c(comment_block, line)  # Include preceding comments
      comment_block <- character(0)
      
      # Count braces on this line
      brace_depth <- count_braces(line)
      
      if (verbose) {
        cat(sprintf("  Found function: %s (line %d)\n", func_name, i))
      }
      
      i <- i + 1
      next
    }
    
    # ========================================================================
    # DETECT DEMO BLOCKS
    # ========================================================================
    # Pattern 1: if (FALSE) { ... } blocks
    # Pattern 2: Sections with specific markers like "# EXAMPLE" or "# MAIN EXECUTION"
    
    demo_start_pattern <- "^if[[:space:]]*\\([[:space:]]*FALSE[[:space:]]*\\)[[:space:]]*\\{"
    demo_marker_pattern <- "^#.*\\b(EXAMPLE|MAIN EXECUTION|RUN|DEMO)\\b"
    
    if (!in_function && !in_demo) {
      # Check for if(FALSE) block
      if (grepl(demo_start_pattern, trimmed)) {
        in_demo <- TRUE
        current_name <- "demo"
        current_lines <- c(comment_block, line)
        comment_block <- character(0)
        brace_depth <- count_braces(line)
        
        if (verbose) {
          cat(sprintf("  Found demo block (line %d)\n", i))
        }
        
        i <- i + 1
        next
      }
      
      # Check for demo marker comment
      if (grepl(demo_marker_pattern, trimmed, ignore.case = TRUE)) {
        # Extract a name from the comment if possible
        marker_match <- regmatches(trimmed, regexpr("\\b(EXAMPLE|MAIN EXECUTION|RUN|DEMO)\\b", 
                                                     trimmed, ignore.case = TRUE))
        demo_name <- tolower(gsub(" ", "_", marker_match))
        
        in_demo <- TRUE
        current_name <- demo_name
        current_lines <- line
        brace_depth <- 0  # Will accumulate as we see braces
        
        if (verbose) {
          cat(sprintf("  Found demo section: %s (line %d)\n", demo_name, i))
        }
        
        i <- i + 1
        next
      }
    }
    
    # ========================================================================
    # ACCUMULATE LINES WITHIN FUNCTION OR DEMO
    # ========================================================================
    
    if (in_function || in_demo) {
      current_lines <- c(current_lines, line)
      brace_depth <- brace_depth + count_braces(line)
      
      # Check if function/demo is complete
      if (brace_depth == 0 && length(current_lines) > 1) {
        # Function or demo is complete
        
        if (in_function) {
          # Save function file
          output_file <- file.path(subdir, sprintf("fun_%s.R", current_name))
          
          if (file.exists(output_file) && !overwrite) {
            if (verbose) {
              cat(sprintf("    Skipping (exists): %s\n", output_file))
            }
            skipped_files <- c(skipped_files, output_file)
          } else {
            writeLines(current_lines, output_file)
            functions_created <- c(functions_created, output_file)
            if (verbose) {
              cat(sprintf("    Wrote: %s (%d lines)\n", 
                         output_file, length(current_lines)))
            }
          }
          
          in_function <- FALSE
        }
        
        if (in_demo) {
          # Save demo file
          output_file <- file.path(subdir, sprintf("run_%s.R", current_name))
          
          if (file.exists(output_file) && !overwrite) {
            if (verbose) {
              cat(sprintf("    Skipping (exists): %s\n", output_file))
            }
            skipped_files <- c(skipped_files, output_file)
          } else {
            writeLines(current_lines, output_file)
            demos_created <- c(demos_created, output_file)
            if (verbose) {
              cat(sprintf("    Wrote: %s (%d lines)\n", 
                         output_file, length(current_lines)))
            }
          }
          
          in_demo <- FALSE
        }
        
        # Reset
        current_name <- ""
        current_lines <- character(0)
        brace_depth <- 0
      }
      
      i <- i + 1
      next
    }
    
    # ========================================================================
    # ACCUMULATE COMMENT BLOCKS (potential documentation)
    # ========================================================================
    
    if (grepl("^#", trimmed)) {
      comment_block <- c(comment_block, line)
    } else if (trimmed == "") {
      # Keep blank lines in comment blocks
      if (length(comment_block) > 0) {
        comment_block <- c(comment_block, line)
      }
    } else {
      # Non-comment, non-blank line - reset comment block
      comment_block <- character(0)
    }
    
    i <- i + 1
  }
  
  # ========================================================================
  # HANDLE INCOMPLETE BLOCKS (shouldn't happen with well-formed code)
  # ========================================================================
  
  if (in_function || in_demo) {
    warning(sprintf("Incomplete %s block at end of file: %s", 
                   ifelse(in_function, "function", "demo"), 
                   current_name))
  }
  
  # ========================================================================
  # SUMMARY
  # ========================================================================
  
  if (verbose) {
    cat("\n=== SUMMARY ===\n")
    cat(sprintf("Functions extracted: %d\n", length(functions_created)))
    cat(sprintf("Demo blocks extracted: %d\n", length(demos_created)))
    cat(sprintf("Files skipped: %d\n", length(skipped_files)))
  }
  
  invisible(list(
    functions = functions_created,
    demos = demos_created,
    skipped = skipped_files
  ))
}

# ==============================================================================
# HELPER FUNCTION: Count braces
# ==============================================================================

count_braces <- function(line) {
  # Count net brace depth change in a line
  # Returns: integer (positive = more open, negative = more close, 0 = balanced)
  
  # Remove strings and comments to avoid counting braces in them
  line_clean <- remove_strings_and_comments(line)
  
  n_open <- nchar(gsub("[^{]", "", line_clean))
  n_close <- nchar(gsub("[^}]", "", line_clean))
  
  return(n_open - n_close)
}

# ==============================================================================
# HELPER FUNCTION: Remove strings and comments
# ==============================================================================

remove_strings_and_comments <- function(line) {
  # Remove quoted strings and comments from a line to avoid false brace detection
  # This is a simple heuristic - not perfect but good enough for most code
  
  # Remove comments (everything after unquoted #)
  # Simple approach: remove from first # to end (may have false positives)
  line_no_comment <- sub("#.*$", "", line)
  
  # Remove double-quoted strings
  line_no_strings <- gsub('"([^"\\\\]|\\\\.)*"', '""', line_no_comment)
  
  # Remove single-quoted strings
  line_no_strings <- gsub("'([^'\\\\]|\\\\.)*'", "''", line_no_strings)
  
  return(line_no_strings)
}

# ==============================================================================
# HELPER FUNCTION: Create an index/master file
# ==============================================================================

create_index_file <- function(split_result, output_file = "index_functions.R", 
                              subdir = ".", overwrite = TRUE) {
  # Create an index file that sources all split files
  #
  # Args:
  #   split_result: result from split_r_file() [list]
  #   output_file: name for index file [character]
  #   subdir: subdirectory (same as used in split_r_file) [character]
  #   overwrite: whether to overwrite existing index [logical]
  #
  # Returns:
  #   path to created index file [character]
  
  output_path <- file.path(subdir, output_file)
  
  if (file.exists(output_path) && !overwrite) {
    message(sprintf("Index file exists (overwrite=FALSE): %s", output_path))
    return(invisible(output_path))
  }
  
  # Create header
  lines <- c(
    "# ==============================================================================",
    "# AUTO-GENERATED INDEX FILE",
    sprintf("# Created: %s", Sys.time()),
    "# ==============================================================================",
    "# Source this file to load all functions and run demos",
    "# ==============================================================================",
    "",
    "# === LOAD FUNCTIONS ===",
    ""
  )
  
  # Add source commands for functions
  if (length(split_result$functions) > 0) {
    for (func_file in split_result$functions) {
      rel_path <- basename(func_file)
      lines <- c(lines, sprintf('source("%s")', rel_path))
    }
  } else {
    lines <- c(lines, "# No functions extracted")
  }
  
  lines <- c(lines, "", "# === RUN DEMOS (commented out by default) ===", "")
  
  # Add source commands for demos (commented out)
  if (length(split_result$demos) > 0) {
    for (demo_file in split_result$demos) {
      rel_path <- basename(demo_file)
      lines <- c(lines, sprintf('# source("%s")', rel_path))
    }
  } else {
    lines <- c(lines, "# No demo blocks extracted")
  }
  
  lines <- c(lines, "", "# ==============================================================================")
  
  # Write file
  writeLines(lines, output_path)
  message(sprintf("Created index file: %s", output_path))
  
  invisible(output_path)
}

# ==============================================================================
# EXAMPLE USAGE (commented out)
# ==============================================================================

if (FALSE) {
  # Example: Split the fish farm model
  getwd()
  result <- split_r_file(
    input_file = "fish_farm_observer_model_1.R",
    subdir = "fish_farm_functions",
    overwrite = TRUE,
    verbose = TRUE
  )
  
  # Create index file to source all functions
  create_index_file(result, subdir = "fish_farm_functions")
  
  # Now you can:
  # source("fish_farm_functions/index_functions.R")
  # Or source individual functions:
  # source("fish_farm_functions/fun_create_feeding_table.R")
  # source("fish_farm_functions/fun_run_fish_farm_simulation.R")
}
