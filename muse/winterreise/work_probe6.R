setwd("C:/Users/Lenovo/Documents/misc-tests/muse/winterreise")
sink("work_probe6_output.txt")
suppressPackageStartupMessages({ library(httr2); library(jsonlite); library(glue) })
api <- function(host, ...) {
  request(glue("https://{host}/w/api.php")) |> req_url_query(..., format = "json") |>
    req_user_agent("song-cycle-semantic-path/0.1 (R httr2)") |> req_perform() |> resp_body_json()
}
# Wikipedia Dichterliebe: sections + the Liedertexte section
secs <- api("de.wikipedia.org", action = "parse", page = "Dichterliebe", prop = "sections")$parse$sections
for (s in secs) cat(s$index, s$line, "\n")
i <- which(vapply(secs, function(s) grepl("Liedertext", s$line), logical(1)))
if (length(i)) {
  wt <- api("de.wikipedia.org", action = "parse", page = "Dichterliebe", prop = "wikitext",
            section = secs[[i[1]]]$index)$parse$wikitext[[1]]
  cat("\n--- Liedertexte section:", nchar(wt), "chars ---\n", substr(wt, 1, 1500), "\n")
}
# Wikisource Die schöne Müllerin
wt <- api("de.wikisource.org", action = "parse", page = "Die schöne Müllerin", prop = "wikitext", redirects = 1)$parse$wikitext[[1]]
con <- file("cache/probe_schoene_muellerin.txt", open = "w", encoding = "UTF-8"); writeLines(wt, con); close(con)
cat("\n===== Die schöne Müllerin:", nchar(wt), "chars =====\n")
cat(substr(wt, 1, 5000), "\n")
sink()
