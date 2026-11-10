setwd("C:/Users/Lenovo/Documents/misc-tests/muse/winterreise")
sink("work_probe2_output.txt")
suppressPackageStartupMessages({ library(httr2); library(jsonlite); library(glue) })
key <- Sys.getenv("VOYAGE_API_KEY")
cat("VOYAGE_API_KEY set:", nzchar(key), " length:", nchar(key), "\n")
if (nzchar(key)) {
  r <- request("https://api.voyageai.com/v1/embeddings") |>
    req_auth_bearer_token(key) |>
    req_body_json(list(input = list("Fremd bin ich eingezogen.", "I came here as a stranger."),
                       model = "voyage-4-large", input_type = "document")) |>
    req_error(is_error = function(resp) FALSE) |>
    req_perform()
  cat("status", resp_status(r), "\n")
  b <- resp_body_json(r, simplifyVector = TRUE)
  if (!is.null(b$data)) {
    e <- do.call(rbind, b$data$embedding)
    cat("dim", dim(e), " cos DE/EN =", sum(e[1, ] * e[2, ]) / sqrt(sum(e[1, ]^2) * sum(e[2, ]^2)), "\n")
    print(b$usage)
  } else print(b)
}
wiki <- function(host, page) {
  r <- request(glue("https://{host}/w/api.php")) |>
    req_url_query(action = "parse", page = page, prop = "wikitext", format = "json", redirects = 1) |>
    req_user_agent("winterreise-semantic-path/0.1 (R httr2)") |>
    req_error(is_error = function(resp) FALSE) |> req_perform()
  b <- resp_body_json(r)
  if (!is.null(b$error)) return(paste("ERROR", b$error$info))
  b$parse$wikitext[[1]]
}
for (pg in list(c("de.wikipedia.org", "Dichterliebe"), c("de.wikisource.org", "Dichterliebe"),
                c("de.wikisource.org", "Lyrisches Intermezzo"))) {
  wt <- wiki(pg[1], pg[2])
  cat(glue("\n===== {pg[1]} / {pg[2]}: {nchar(wt)} chars ====="), "\n")
  writeLines(wt, glue("cache/probe_{gsub('[^A-Za-z]', '_', pg[2])}_{substr(pg[1], 1, 2)}{substr(pg[1], 4, 7)}.txt"), useBytes = TRUE)
  cat(substr(wt, 1, 2500), "\n")
}
sink()
