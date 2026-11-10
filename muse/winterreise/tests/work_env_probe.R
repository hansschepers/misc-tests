sink("C:/Users/Lenovo/Documents/misc-tests/muse/winterreise/tests/work_env_probe_output.txt")
pk <- c("httr2", "jsonlite", "stringr", "data.table", "uwot", "ggplot2", "ggrepel",
        "viridisLite", "reticulate", "text", "fastTextR", "word2vec", "irlba", "ggforce")
print(sapply(pk, requireNamespace, quietly = TRUE))
r <- try(httr2::req_perform(httr2::request("http://localhost:11434/api/tags")), silent = TRUE)
print(if (inherits(r, "try-error")) "no ollama server" else httr2::resp_body_string(r))
r2 <- try(httr2::req_perform(httr2::req_url_query(httr2::request("https://de.wikisource.org/w/api.php"),
  action = "parse", page = "Winterreise_(Schubert/Friedlaender)", prop = "wikitext", format = "json")), silent = TRUE)
if (!inherits(r2, "try-error")) {
  wt <- httr2::resp_body_json(r2)$parse$wikitext[[1]]
  cat(nchar(wt), "chars\n")
  writeLines(wt, "C:/Users/Lenovo/Documents/misc-tests/muse/winterreise/wikitext_raw.txt")
  cat(substr(wt, 1, 3000))
} else print(r2)
Sys.getenv(c("OPENAI_API_KEY", "VOYAGE_API_KEY", "ANTHROPIC_API_KEY")) |> nzchar() |> print()
sink()
