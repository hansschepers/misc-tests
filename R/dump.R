#' dump
#' @export
dump <- function(object, fftxt = "object.txt", wd = NULL, split = TRUE){
  
  if(is.null(wd)){
    if (exists("wd", .GlobalEnv)){
      wd <- get("wd", .GlobalEnv)
    } else {
      wd <- "."
    }
  }
  ff <- file.path(wd, fftxt)
  sink(ff, split = split)
  cmd <- as.character(substitute(object))
  print(cmd)
  print(object)
  sink(NULL)
  ff
}

