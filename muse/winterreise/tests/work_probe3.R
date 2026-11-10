setwd("C:/Users/Lenovo/Documents/misc-tests/muse/winterreise")
sink("tests/work_probe3_output.txt")
suppressPackageStartupMessages({ library(httr2); library(jsonlite); library(glue) })
r <- request("https://de.wikisource.org/w/api.php") |>
  req_url_query(action = "parse", page = "Lyrisches Intermezzo", prop = "text", format = "json") |>
  req_user_agent("winterreise-semantic-path/0.1 (R httr2)") |> req_perform()
html <- resp_body_json(r)$parse$text[[1]]
cat(nchar(html), "chars html\n")
writeLines(html, "cache/lyrisches_intermezzo.html", useBytes = TRUE)
txt <- gsub("<br */?>", "\n", html)
txt <- gsub("</(p|div|h[1-6]|center)>", "\n", txt)
txt <- gsub("<[^>]+>", "", txt)
txt <- gsub("&#160;|&nbsp;", " ", txt)
ln <- trimws(strsplit(txt, "\n")[[1]])
writeLines(ln, "cache/lyrisches_intermezzo_lines.txt", useBytes = TRUE)
cat(length(ln), "lines\n")
cat(head(ln[ln != ""], 160), sep = "\n")
sink()
