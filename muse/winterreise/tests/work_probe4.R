setwd("C:/Users/Lenovo/Documents/misc-tests/muse/winterreise")
sink("tests/work_probe4_output.txt")
suppressPackageStartupMessages({ library(httr2); library(jsonlite); library(glue) })
api <- function(...) {
  request("https://de.wikisource.org/w/api.php") |> req_url_query(..., format = "json") |>
    req_user_agent("winterreise-semantic-path/0.1 (R httr2)") |> req_perform() |> resp_body_json()
}
for (q in c("Im wunderschönen Monat Mai", "Ich grolle nicht", "Die alten, bösen Lieder", "Allnächtlich im Traume")) {
  s <- api(action = "query", list = "search", srsearch = glue('"{q}"'), srlimit = 10)
  cat(glue("\n--- search: {q}"), "\n")
  for (h in s$query$search) cat(" ", h$title, "\n")
}
# intermezzo HTML structure around poem I: is there stanza markup?
html <- readLines("cache/lyrisches_intermezzo.html", encoding = "UTF-8", warn = FALSE) |> paste(collapse = "\n")
i <- regexpr("Aus meinen Thr", html)
cat("\n--- raw html around poem I ---\n", substr(html, i - 600, i + 900), "\n")
cat("\n'Im wunderschönen' in 1823 text:", grepl("wundersch", html), "\n")
sink()
