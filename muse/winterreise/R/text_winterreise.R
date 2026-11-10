# Winterreise text: fetch (Wikisource, Schubert/Friedlaender edition) + parse into
# song x stanza units. Public domain (Müller 1794-1827).

fetch_winterreise_wikitext <- function(cache_file) {
  if (file.exists(cache_file)) return(readLines(cache_file, encoding = "UTF-8", warn = FALSE))
  resp <- request("https://de.wikisource.org/w/api.php") |>
    req_url_query(action = "parse", page = "Winterreise_(Schubert/Friedlaender)",
                  prop = "wikitext", format = "json") |>
    req_user_agent("winterreise-semantic-path/0.1 (R httr2)") |>
    req_perform()
  wt <- resp_body_json(resp)$parse$wikitext[[1]]
  dir.create(dirname(cache_file), showWarnings = FALSE, recursive = TRUE)
  con <- file(cache_file, open = "w", encoding = "UTF-8")
  writeLines(wt, con)
  close(con)
  strsplit(wt, "\n")[[1]]
}

# Returns list(lines = one row per sung line, stanzas = one row per DEDUPLICATED stanza).
# Dedup: Schubert repeats lines/stanzas musically; the embedding unit is Müller's stanza
# (each distinct line once, in first-occurrence order). n_lines_sung keeps the repetition count.
parse_winterreise <- function(wikitext_lines) {
  txt <- paste(wikitext_lines, collapse = "\n")
  head_re <- "(?m)^== *([0-9]+)\\. *(.+?) *== *$"
  heads <- regmatches(txt, gregexpr(head_re, txt, perl = TRUE))[[1]]
  bodies <- strsplit(txt, head_re, perl = TRUE)[[1]][-1]
  stopifnot(length(heads) == 24, length(bodies) == 24)

  lines_dt <- rbindlist(lapply(seq_along(heads), function(i) {
    title <- sub("^== *[0-9]+\\. *(.+?) *== *$", "\\1", heads[i], perl = TRUE)
    poem <- sub("(?s).*?<poem>(.*?)</poem>.*", "\\1", bodies[i], perl = TRUE)
    ln <- strsplit(poem, "\n")[[1]]
    ln <- trimws(gsub("[\\h]+", " ", ln, perl = TRUE))
    is_tempo <- grepl("^''.*''$", ln)
    tempo <- gsub("''", "", ln[is_tempo][1])
    ln <- ln[!is_tempo]
    stanza <- cumsum(ln == "" & c(TRUE, head(ln, -1) != ""))
    keep <- ln != ""
    data.table(song = i, title = title, tempo = tempo,
               stanza = match(stanza[keep], unique(stanza[keep])), line = ln[keep])
  }))
  lines_dt[, key := tolower(gsub("[[:punct:]–’ ]+", " ", line))]
  lines_dt[, key := trimws(key)]

  stanzas <- lines_dt[, .(text = paste(line[!duplicated(key)], collapse = " ")
                          , first_line = line[1]
                          , n_lines_sung = .N
                          , n_lines_unique = sum(!duplicated(key))
                          , skey = paste(unique(key), collapse = "|")
                          ), by = .(song, title, tempo, stanza)]
  # drop a stanza whose distinct lines are all already present in an earlier stanza of that song
  stanzas[, dup := {
    seen <- character(0)
    out <- logical(.N)
    for (j in seq_len(.N)) {
      k <- strsplit(skey[j], "|", fixed = TRUE)[[1]]
      out[j] <- all(k %in% seen)
      seen <- c(seen, k)
    }
    out
  }, by = song]
  stanzas <- stanzas[dup == FALSE][, c("dup", "skey") := NULL]
  stanzas[, stanza := seq_len(.N), by = song]
  stanzas[, unit_id := paste0(song, ".", stanza)]  # glue() cannot see data.table columns in j
  list(lines = lines_dt, stanzas = stanzas)
}

# Müller's final 1824 order, expressed in Schubert numbers (from memory of the 1824
# "Gedichte aus den hinterlassenen Papieren eines reisenden Waldhornisten" -- VERIFY).
mueller_order_in_schubert_numbers <- function() {
  c(1, 2, 3, 4, 5, 13, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 9, 10, 23, 11, 12, 22, 24)
}
