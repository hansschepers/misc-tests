setwd("C:/Users/Lenovo/Documents/misc-tests/muse/winterreise")
sink("tests/work_probe7_output.txt")
suppressPackageStartupMessages({ library(httr2); library(jsonlite); library(glue) })
api <- function(...) {
  request("https://de.wikisource.org/w/api.php") |> req_url_query(..., format = "json") |>
    req_user_agent("song-cycle-semantic-path/0.1 (R httr2)") |> req_perform() |> resp_body_json()
}
for (pg in c("Wanderschaft", "Thränenregen", "Der Jäger (Müller)")) {
  wt <- api(action = "parse", page = pg, prop = "wikitext", redirects = 1)$parse$wikitext[[1]]
  cat(glue("\n===== {pg} ====="), "\n", wt, "\n")
}
sink()
