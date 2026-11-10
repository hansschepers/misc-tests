setwd("C:/Users/Lenovo/Documents/misc-tests/muse/winterreise")
sink("tests/work_probe5_output.txt")
suppressPackageStartupMessages({ library(httr2); library(jsonlite); library(glue) })
api <- function(...) {
  request("https://de.wikisource.org/w/api.php") |> req_url_query(..., format = "json") |>
    req_user_agent("winterreise-semantic-path/0.1 (R httr2)") |> req_perform() |> resp_body_json()
}
wt <- api(action = "parse", page = "Buch der Lieder", prop = "wikitext")$parse$wikitext[[1]]
writeLines(wt, "cache/probe_buch_der_lieder.txt", useBytes = TRUE)
i <- regexpr("Intermezzo", wt)
cat(substr(wt, max(1, i - 300), i + 6000), "\n")
for (pg in c("Im wunderschönen Monat Mai", "Ich grolle nicht, und wenn das Herz auch bricht")) {
  cat(glue("\n===== {pg} ====="), "\n")
  cat(api(action = "parse", page = pg, prop = "wikitext", redirects = 1)$parse$wikitext[[1]], "\n")
}
sink()
